"""Data: h5ad loading, cell-level splits, the locally paired control, the calibrated reference offset, and the batches.

h5ad layout: obs['perturbation'] ('Control' = control cells; a gene name = knock-out cells; 'None' / 'Multiple' /
missing = environment-only cells), obs['cell_type'], obs['sample'] (section id, optional), obsm['spatial'], and either log-normalised
X (uns['log1p'] present) or raw counts (X, or layers['counts'] with the library size in obs)."""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors


def lognorm_expression(adata):
    """log1p(1e4 x / library size) without transforming an already log-normalised matrix twice."""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    if "log1p" in adata.uns or ("log_normalized" in adata.layers and X.max() < 50):
        print("  expression: adata.X is already log-normalised, used as is")
        return X.astype(np.float32)
    if "counts" in adata.layers:
        C = adata.layers["counts"]; C = C.toarray() if hasattr(C, "toarray") else np.asarray(C)
        lib, src = None, None
        for col in ["nCount_Spatial", "total_counts", "n_counts"]:
            if col in adata.obs.columns:
                lib = adata.obs[col].values.astype(np.float64)[:, None]; src = f"obs['{col}']"; break
        if lib is None: lib = C.sum(axis=1, keepdims=True); src = "row sums of the counts layer"
        lib[lib == 0] = 1
        print(f"  expression: log1p(1e4 counts / {src})")
        return np.log1p(C / lib * 1e4).astype(np.float32)
    lib = X.sum(axis=1, keepdims=True); lib[lib == 0] = 1
    print("  expression: log1p(1e4 X / row sums)")
    return np.log1p(X / lib * 1e4).astype(np.float32)


def select_genes(adata, n_genes):
    """The gene panel: var['highly_variable'] (ranked when a rank is stored) if present, else seurat_v3 on counts."""
    if "highly_variable" in adata.var.columns:
        if "highly_variable_rank" in adata.var.columns:
            rank = adata.var["highly_variable_rank"].values.copy(); rank[np.isnan(rank)] = np.inf
            idx = np.argsort(rank)[:n_genes]
        else:
            hv = np.where(adata.var["highly_variable"].values)[0]; idx = hv[:n_genes] if len(hv) >= n_genes else hv
        return adata[:, idx].copy()
    print("  computing highly variable genes (seurat_v3)")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_genes, flavor="seurat_v3", layer="counts" if "counts" in adata.layers else None)
    return adata[:, adata.var["highly_variable"]].copy()


def separate_slides(coords, sample_labels=None, gap_factor=1.5):
    """Shift the sections along x so that no spatial neighbourhood crosses a section."""
    coords = np.asarray(coords, np.float64).copy()
    if sample_labels is None: return coords
    sample_labels = np.asarray(sample_labels); uniq = list(np.unique(sample_labels))
    if len(uniq) <= 1: return coords
    step = float(coords[:, 0].max() - coords[:, 0].min()) * gap_factor
    for i, s in enumerate(uniq): coords[sample_labels == s, 0] += i * step
    print(f"  {len(uniq)} sections shifted apart along x by {step:.0f}")
    return coords


def gene_id_mapping(gene_names, vocab):
    ids = []
    for g in gene_names:
        g = str(g)
        for cand in (g, g.upper(), g.lower()):
            if cand in vocab: ids.append(vocab[cand]); break
        else: ids.append(-1)
    ids = np.asarray(ids, np.int64)
    print(f"  scGPT vocabulary: {(ids >= 0).sum()}/{len(ids)} panel genes have a token")
    return ids


def load_data(config, vocab=None):
    path = os.path.expanduser(config.data_file)
    print(f"loading {path}")
    adata = sc.read_h5ad(path)
    ext = None
    if config.cell_embedding_file:
        f = os.path.expanduser(config.cell_embedding_file)
        E = np.load(f).astype(np.float32); names = [l.rstrip("\n") for l in open(os.path.splitext(f)[0] + ".obs_names.txt")]
        ix = pd.Index(names).get_indexer(adata.obs_names.astype(str))
        if (ix < 0).any(): raise ValueError(f"{f}: {(ix < 0).sum()} cells of the h5ad have no embedding")
        ext = E[ix]; print(f"  cell embedding file: {f} {E.shape}")
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")
    if adata.n_vars > int(config.n_genes):
        adata = select_genes(adata, int(config.n_genes)); print(f"  gene panel: {adata.n_vars} genes")
    config.n_genes = int(adata.n_vars)
    coords_raw = np.asarray(adata.obsm["spatial"], np.float64)
    expression_all = lognorm_expression(adata)
    cell_types_all = adata.obs["cell_type"].values.astype(str) if "cell_type" in adata.obs else None
    sections = None
    for col in ["sample", "batch", "slice"]:
        if col in adata.obs.columns:
            sections = adata.obs[col].values; print(f"  section label obs['{col}'] ({len(np.unique(sections))} sections)"); break
    coords_all = separate_slides(coords_raw, sections)
    pert = adata.obs["perturbation"]
    has = (pert.notna()) & (pert.astype(str) != "None") & (pert.astype(str) != "Multiple")
    has = np.asarray(has.values, bool)
    pi = np.where(has)[0]; obs_p = adata.obs.iloc[pi]
    pert_p = obs_p["perturbation"]
    if isinstance(pert_p.dtype, pd.CategoricalDtype): pert_p = pert_p.cat.remove_unused_categories()   # 'None' / 'Multiple' are not labels
    names = list(pert_p.value_counts().index)
    to_idx = {n: i for i, n in enumerate(names)}
    labels = np.array([to_idx[p] for p in obs_p["perturbation"].values])
    ctrl_indices = np.where(obs_p["perturbation"].values == "Control")[0]; ko_indices = np.where(obs_p["perturbation"].values != "Control")[0]
    expression = expression_all[pi]; coords = coords_all[pi]
    ctrl_mean = expression[ctrl_indices].mean(0)
    gene_names = adata.var["gene_name"].values if "gene_name" in adata.var else adata.var_names.values
    print(f"  perturbed set: {len(pi)} cells ({len(ctrl_indices)} control, {len(ko_indices)} knock-out), {len(names)} labels")
    return {"expression": expression.astype(np.float32), "coords": coords.astype(np.float32), "perturb_labels": labels, "perturb_names": names,
            "ctrl_indices": ctrl_indices, "ko_indices": ko_indices, "ctrl_mean": ctrl_mean.astype(np.float32),
            "cell_emb": (ext[pi] if ext is not None else None), "gene_names": gene_names,
            "gene_ids": (gene_id_mapping(gene_names, vocab) if vocab is not None else None),
            "cell_types": (cell_types_all[pi] if cell_types_all is not None else None), "obs_names": np.asarray(adata.obs_names.astype(str))[pi],
            "expression_all": expression_all.astype(np.float32), "coords_all": coords_all.astype(np.float32),
            "cell_types_all": cell_types_all, "perturbed_indices_in_all": pi, "n_cells_all": int(len(expression_all))}


def make_split(data, config):
    """random: knock-out cells shuffled by RandomState(seed), first train_ratio for training | file: indices from an npz."""
    ko = np.asarray(data["ko_indices"]).copy(); names = data["perturb_names"]; lab = data["perturb_labels"]
    if str(config.split_mode) == "file":
        s = np.load(os.path.expanduser(config.split_file), allow_pickle=True); tr, te = np.asarray(s["train_ko_indices"]), np.asarray(s["test_ko_indices"])
        print(f"split from {config.split_file}")
    else:
        np.random.RandomState(int(config.seed)).shuffle(ko); n = int(len(ko) * float(config.train_ratio)); tr, te = ko[:n], ko[n:]
        print(f"random cell split (train_ratio {config.train_ratio}, seed {config.seed})")
    cnt = lambda idx: {names[p]: int(c) for p, c in zip(*np.unique(lab[idx], return_counts=True))}
    print(f"  train {len(tr)} cells {cnt(tr)}\n  test {len(te)} cells {cnt(te)}")
    data["train_ko_indices"] = tr; data["test_ko_indices"] = te
    return tr, te


def attach_reference(data, config, train_idx):
    """Environment-matched reference profile r_i of every perturbed-set cell (scripts/build_reference.py), calibrated per gene on
    the training knock-out cells.  The offset alpha_g (r_ig - c_g) is added to the prediction."""
    from .reference import calibrate_offset
    R = np.load(os.path.expanduser(config.ref_file)).astype(np.float32)
    if R.shape != data["expression"].shape: raise ValueError(f"{config.ref_file}: shape {R.shape} != expression {data['expression'].shape}")
    data["ref_cell"] = R; print(f"  reference: {config.ref_file}")
    data["ref_alpha"] = calibrate_offset(data["expression"], data["ctrl_mean"], data["perturb_labels"], train_idx, R)


class PerturbationDataset(Dataset):
    """One item = a knock-out cell i: its paired control cell (input), its own response Delta_i = x_i - c (target), its
    perturbation, its index among the environment-encoder nodes, and the calibrated reference offset."""

    def __init__(self, data, config, indices, vocab=None, use_scgpt_input=False, seed=0):
        self.expression = data["expression"]; self.perturb_labels = data["perturb_labels"]; self.ctrl_mean = data["ctrl_mean"]
        self.ctrl_indices = np.asarray(data["ctrl_indices"]); self.cell_emb = data.get("cell_emb"); self.config = config
        self.indices = np.asarray(indices); self.rng = np.random.default_rng(seed)
        self.index_mode = data.get("index_mode", "global"); self.perturbed_indices_in_all = data["perturbed_indices_in_all"]
        if "ref_alpha" not in data: raise ValueError("call attach_reference() first")
        self.ref_cell = data["ref_cell"]; self.ref_alpha = data["ref_alpha"]
        k = int(config.pair_k); XY = np.asarray(data["coords"], np.float64); ctrl = self.ctrl_indices
        ct = np.asarray(data["cell_types"]).astype(str) if data.get("cell_types") is not None else np.array(["all"] * len(XY))
        self._pair_cands = np.zeros((len(XY), k), np.int64); n_fb = 0
        for t in np.unique(ct):
            rows = np.where(ct == t)[0]; cg = ctrl[ct[ctrl] == t]
            if len(cg) < 5: cg = ctrl; n_fb += len(rows)
            _, j = NearestNeighbors(n_neighbors=min(k, len(cg))).fit(XY[cg]).kneighbors(XY[rows]); jj = cg[j]
            if jj.shape[1] < k: jj = np.concatenate([jj] * (k // jj.shape[1] + 1), axis=1)[:, :k]
            self._pair_cands[rows] = jj
        print(f"  paired control: one of the {k} nearest same-type control cells (type with < 5 controls falls back to any type: {n_fb} cells)")
        self.n_bins = int(config.scgpt_n_bins)
        self.gene_ids = data.get("gene_ids"); self.vocab = vocab
        self.use_scgpt_input = bool(use_scgpt_input and self.gene_ids is not None and vocab is not None)
        if self.use_scgpt_input:
            self.valid_gene_mask = self.gene_ids >= 0; self.valid_gene_ids = self.gene_ids[self.valid_gene_mask]

    def __len__(self):
        return len(self.indices)

    def _tokenize(self, expr):
        v = expr[self.valid_gene_mask]; nz = np.where(v > 0)[0]
        if len(nz) == 0: nz = np.arange(len(self.valid_gene_ids))
        max_len = int(self.config.max_seq_len) - 1
        if len(nz) > max_len: nz = nz[np.argsort(v[nz])[::-1][:max_len]]
        ids = self.valid_gene_ids[nz]; vals = v[nz].astype(np.float64)
        if len(vals) > 0 and vals.max() > 0:
            bins = np.quantile(vals, np.linspace(0, 1, self.n_bins - 1)); vals = np.clip(np.digitize(vals, bins), 1, self.n_bins - 1).astype(np.float32)
        ids = np.insert(ids, 0, self.vocab["<cls>"]); vals = np.insert(vals, 0, 0)
        return torch.LongTensor(ids), torch.FloatTensor(vals), len(ids)

    def __getitem__(self, idx):
        i = int(self.indices[idx]); p = int(self.perturb_labels[i])
        c = int(self.rng.choice(self._pair_cands[i]))
        out = {"delta_x": torch.FloatTensor(self.expression[i] - self.ctrl_mean), "c_idx": p,
               "center_idx": int(self.perturbed_indices_in_all[i]) if self.index_mode == "global" else i,
               "ref_off": torch.FloatTensor(self.ref_alpha * (self.ref_cell[i] - self.ctrl_mean))}
        if self.cell_emb is not None: out["ctrl_emb"] = torch.FloatTensor(self.cell_emb[c])
        if self.use_scgpt_input:
            out["gene_ids"], out["values"], out["seq_len"] = self._tokenize(self.expression[c])
        return out


def collate(batch, pad_token_id=0, pad_value=0.0):
    out = {"delta_x": torch.stack([b["delta_x"] for b in batch]), "c_idx": torch.LongTensor([b["c_idx"] for b in batch]),
           "center_indices": torch.LongTensor([b["center_idx"] for b in batch]), "ref_off": torch.stack([b["ref_off"] for b in batch])}
    if "ctrl_emb" in batch[0]: out["ctrl_emb"] = torch.stack([b["ctrl_emb"] for b in batch])
    if "gene_ids" in batch[0]:
        L = max(b["seq_len"] for b in batch); ids, vals, masks = [], [], []
        for b in batch:
            n = b["seq_len"]; pad = L - n
            ids.append(torch.cat([b["gene_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]) if pad else b["gene_ids"])
            vals.append(torch.cat([b["values"], torch.full((pad,), pad_value, dtype=torch.float)]) if pad else b["values"])
            m = torch.zeros(L, dtype=torch.bool); m[n:] = True; masks.append(m)
        out["gene_ids"] = torch.stack(ids); out["values"] = torch.stack(vals); out["padding_mask"] = torch.stack(masks)
    return out


def make_collate(vocab=None, pad_value=0.0):
    pad_id = 0
    if vocab is not None:
        try: pad_id = vocab["<pad>"]
        except KeyError: pad_id = 0
    return lambda batch: collate(batch, pad_token_id=pad_id, pad_value=pad_value)


def create_loaders(data, config, train_idx, test_idx, vocab=None, use_scgpt_input=False):
    """Training and test data sets; their control pairings are drawn from their own generators (seed, seed + 1)."""
    train_ds = PerturbationDataset(data, config, train_idx, vocab, use_scgpt_input, seed=int(config.seed))
    test_ds = PerturbationDataset(data, config, test_idx, vocab, use_scgpt_input, seed=int(config.seed) + 1)
    cf = make_collate(vocab, pad_value=float(config.scgpt_pad_value)) if use_scgpt_input else collate
    train_loader = DataLoader(train_ds, batch_size=int(config.batch_size), shuffle=True, collate_fn=cf, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(config.batch_size), shuffle=False, collate_fn=cf, num_workers=0)
    return train_loader, test_loader
