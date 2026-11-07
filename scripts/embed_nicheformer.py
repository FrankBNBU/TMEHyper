"""Frozen Nicheformer (theislab; Hugging Face weights) embedding of every cell of a mouse h5ad.
Context tokens [SPECIES_mouse] [TECH_x] [MODALITY_spatial]; the panel genes are mapped to the model vocabulary (human orthologue Ensembl ids)
by scripts/map_panel_to_nicheformer.py.  Tokenisation as in the Nicheformer release: raw counts -> CP10k -> / technology mean (per-gene mean over the
cells of this data set) -> log1p -> genes ranked by value (non-zero only) -> vocabulary ids.  Embedding = masked mean of the gene-token outputs of the
last layer (512-d).  Sequencing-based spatial platforms (Stereo-seq, Visium, Visium HD) have no technology token in the vocabulary; [TECH_10x_3_v3]
(3' poly-A capture chemistry) is used for them.
    python scripts/embed_nicheformer.py --h5ad GSE274447.h5ad --mapping embeddings/GSE274447_to_nicheformer.json --out embeddings/GSE274447.X_niche.npy --hf_dir <nicheformer_hf>
Writes <out> (cells x 512 float32) and <out stem>.obs_names.txt (read by --cell_embedding_file)."""
import argparse, os, json, time, numpy as np, anndata as ad, scipy.sparse as sp, torch
ap = argparse.ArgumentParser(); ap.add_argument("--h5ad", required=True); ap.add_argument("--mapping", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--counts_layer", default=None, help="layer with raw counts when X is not raw (GSE193460: counts)")
ap.add_argument("--tech", default="[TECH_10x_3_v3]"); ap.add_argument("--species", default="[SPECIES_mouse]")
ap.add_argument("--hf_dir", required=True, help="directory of the Nicheformer Hugging Face checkpoint (config, weights, vocab.json)")
ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--batch", type=int, default=128)
a = ap.parse_args()
from transformers import AutoModelForMaskedLM
vocab = json.load(open(os.path.join(a.hf_dir, "vocab.json"))); CTX = [vocab[a.species], vocab[a.tech], vocab["[MODALITY_spatial]"]]; PAD = 0
dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
model = AutoModelForMaskedLM.from_pretrained(a.hf_dir, trust_remote_code=True).to(dev).eval(); m = model.nicheformer; cfg = model.config
A = ad.read_h5ad(a.h5ad); mapping = json.load(open(a.mapping))
genes = [g for g in A.var_names if mapping.get(g, {}).get("token") is not None]; tok = np.array([mapping[g]["token"] for g in genes])
print(f"{a.h5ad}: {A.n_obs} cells; panel genes with a vocabulary token {len(genes)}/{A.n_vars}; context {a.species} {a.tech} [MODALITY_spatial]", flush=True)
Xs = A[:, genes].layers[a.counts_layer] if a.counts_layer else A[:, genes].X
Xs = sp.csr_matrix(Xs, dtype=np.float64); assert np.allclose(Xs.data[:1000], np.round(Xs.data[:1000])), "expects raw counts"
lib = np.asarray(A.X.sum(1) if not a.counts_layer else A.layers[a.counts_layer].sum(1)).ravel()   # CP10k on the cell's counts in the file
Xn = sp.diags(1e4 / np.maximum(lib, 1)) @ Xs; tech_mean = np.maximum(np.asarray(Xn.mean(0)).ravel(), 1e-6)

@torch.no_grad()
def embed(R):
    R = np.log1p(R / tech_mean); L = int((R > 0).sum(1).max()) + 3; ids = np.zeros((len(R), L), dtype=np.int64); ids[:, :3] = CTX
    for i, row in enumerate(R):
        nz = np.nonzero(row)[0]; order = nz[np.argsort(-row[nz])]; ids[i, 3:3 + len(order)] = tok[order]
    ids = torch.as_tensor(ids, device=dev); mask = ids != PAD
    emb = m.embeddings(ids)
    emb = emb + m.positional_embedding(torch.arange(L, device=dev))[None] if cfg.learnable_pe else m.positional_embedding(emb)
    for layer in m.encoder.layers: emb = layer(emb, src_key_padding_mask=~mask, is_causal=False)
    gm = mask[:, 3:].float()[..., None]; return ((emb[:, 3:] * gm).sum(1) / gm.sum(1).clip(1.0)).float().cpu().numpy()

t0 = time.time(); out = []
for s in range(0, A.n_obs, a.batch): out.append(embed(Xn[s:s + a.batch].toarray()))
E = np.concatenate(out).astype(np.float32); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); np.save(a.out, E)
open(os.path.splitext(a.out)[0] + ".obs_names.txt", "w").write("\n".join(map(str, A.obs_names)))
print(f"done {E.shape} in {time.time() - t0:.0f}s | mean norm {np.linalg.norm(E, axis=1).mean():.2f} | cells with no expressed mapped gene {(np.asarray((Xs > 0).sum(1)).ravel() == 0).sum()}", flush=True)
