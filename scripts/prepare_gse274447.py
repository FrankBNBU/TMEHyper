"""GSE274447 (spatial Perturb-seq, Stereo-seq cellbin, mouse brain) in the layout the loader expects.  Every segmented cell stays in the file:
  * edited cells (dominant sgrna_* feature)            -> their perturbation (18 labels including the safe-targeting guide 'msafe')
  * --n_control unedited cells drawn with the edited cells' chip x cell-type x sequencing-depth composition (uniform within a stratum) -> 'Control'
  * all other unedited cells                            -> 'None' (environment only)
Panel: 500 genes = the targeted genes present in the matrix + the top highly variable genes (scanpy seurat_v3 on counts, batch = chip).
    python scripts/prepare_gse274447.py --src GSE274447_cellbin.h5ad --out GSE274447.h5ad
The source h5ad holds the cellbin count matrix with obs 'sample' (chip), 'cell_type', 'perturbation' (dominant sgRNA feature, 'Control' = unedited),
'x', 'y', 'area', 'sg_total'."""
import argparse, numpy as np, pandas as pd, anndata as ad, scanpy as sc, scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n_control", type=int, default=20000); ap.add_argument("--n_genes", type=int, default=500)
a = ap.parse_args()

A = ad.read_h5ad(a.src); A.X = sp.csr_matrix(A.X)
print(f"source {A.shape}; X integer counts: {np.allclose(A.X.data[:100000], np.round(A.X.data[:100000]))}")
# sgRNA feature names -> gene symbols where the target is unambiguous; the rest keep the authors' label
REN = {"clu": "Clu", "fasn": "Fasn", "flcn": "Flcn", "gfap": "Gfap", "lrrk2": "Lrrk2", "oligo2": "Olig2", "rraga": "Rraga",
       "sh3gl2": "Sh3gl2", "srf": "Srf", "stk39": "Stk39", "tbk1": "Tbk1", "trem2": "Trem2"}
p = A.obs["perturbation"].astype(str).map(lambda x: REN.get(x, x)).values.astype(object)
unedited = np.where(p == "Control")[0]
rng = np.random.default_rng(0)
tc = np.asarray(A.X.sum(1)).ravel(); dq = pd.qcut(np.log1p(tc), 10, labels=False, duplicates="drop").astype(int)
ct = (A.obs["sample"].astype(str) + "|" + A.obs["cell_type"].astype(str) + "|d" + pd.Series(dq).astype(str).values).values
edited = (p != "Control")
frac = pd.Series(ct[edited]).value_counts(normalize=True)
n_k = (frac * a.n_control).round().astype(int)
ctrl = np.concatenate([rng.choice(unedited[ct[unedited] == k], min(n, int((ct[unedited] == k).sum())), replace=False) for k, n in n_k.items()])
print(f"control pool: {len(ctrl)} cells; by stratum:", pd.Series(ct[ctrl]).value_counts(normalize=True).round(3).head(8).to_dict())
p[unedited] = "None"; p[ctrl] = "Control"
A.obs["perturbation"] = pd.Categorical(p)
print("labels:", pd.Series(p).value_counts().to_dict())

genes = pd.Index(A.var_names.astype(str)); low = {g.lower(): g for g in genes}
tg = [low[t.lower()] for t in ["C9orf72", "Cfap410"] + list(REN.values()) if t.lower() in low]
print(f"targeted genes in the matrix: {len(tg)} {tg}")
n_hvg = a.n_genes - len(tg)
hv = sc.pp.highly_variable_genes(A, flavor="seurat_v3", n_top_genes=a.n_genes + 50, batch_key="sample", inplace=False)   # needs scikit-misc
hv.index = genes
rank = hv["highly_variable_rank"]
order = [g for g in rank.sort_values().index if g not in tg]
panel = tg + order[:n_hvg]
assert len(panel) == a.n_genes and len(set(panel)) == a.n_genes
B = A[:, panel].copy()
B.obs = B.obs[[c for c in ["sample", "cell_type", "perturbation", "x", "y", "area", "sg_total"] if c in B.obs.columns]]
B.obsm["spatial"] = B.obs[["x", "y"]].to_numpy(np.float32)
B.uns["source"] = (f"GSE274447 Stereo-seq cellbin; perturbation = dominant sgRNA feature; Control = {len(ctrl)} unedited cells drawn with the edited cells' "
                   f"chip x type x depth composition; None = other unedited cells (environment only); panel = {len(tg)} targeted genes + {n_hvg} HVG (seurat_v3, batch = chip)")
B.write_h5ad(a.out)
print(f"wrote {a.out} {B.shape}; panel first 20: {panel[:20]}")
