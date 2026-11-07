"""Description of the observed micro-environment of every cell, built from unperturbed cells only.

Only unperturbed cells contribute expression (every cell that is not a knock-out cell: environment cells and non-targeting
controls), and a cell never contributes to its own description.  Scales are Gaussian kernels in units of the median
nearest-neighbour distance d (1.5 d contact, 5 d paracrine, 15 d tissue), over the 200 nearest unperturbed cells.  Four layers:
  A composition  fractions of fine unperturbed-cell types (k-means on expression PCs) at the three scales, and their entropy
  B state        kernel-weighted mean of unperturbed-cell expression PCs at the three scales
  C signal       paracrine-scale exposure to every ligand of the panel, times the receptor competence of the cell's type
  D structure    local density at the three scales, log distance to the nearest unperturbed cell of each fine type, and the
                 spatial domain (k-means on the tissue-scale composition), one-hot
Every column is standardised.  Returns (features [n_all, D], layer slices)."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ligand -> receptor pairs (CellChatDB / CellPhoneDB; human symbols, matched case-insensitively)
LR = {'TNF': ['TNFRSF1A', 'TNFRSF1B'], 'IL1B': ['IL1R1', 'IL1RAP'], 'IL1A': ['IL1R1', 'IL1RAP'], 'IL6': ['IL6R', 'IL6ST'], 'IFNG': ['IFNGR1', 'IFNGR2'],
      'CXCL9': ['CXCR3'], 'CXCL10': ['CXCR3'], 'CXCL11': ['CXCR3'], 'CCL2': ['CCR2'], 'CCL5': ['CCR5', 'CCR1'], 'CXCL12': ['CXCR4', 'ACKR3'],
      'TGFB1': ['TGFBR1', 'TGFBR2'], 'TGFB2': ['TGFBR1', 'TGFBR2'], 'TGFB3': ['TGFBR1', 'TGFBR2'], 'IL10': ['IL10RA', 'IL10RB'],
      'CSF1': ['CSF1R'], 'VEGFA': ['KDR', 'FLT1'], 'SPP1': ['CD44', 'ITGB1', 'ITGAV'], 'CD274': ['PDCD1'], 'EGF': ['EGFR'], 'AREG': ['EGFR'],
      'WNT5A': ['ROR2', 'FZD5'], 'TNFSF10': ['TNFRSF10A', 'TNFRSF10B'], 'IL4': ['IL4R'], 'IL13': ['IL13RA1', 'IL4R'], 'IL18': ['IL18R1'],
      'IL33': ['IL1RL1'], 'BMP2': ['BMPR1A', 'BMPR2'], 'LBP': ['CD14', 'TLR4'], 'C3': ['C3AR1', 'ITGAM'], 'APOE': ['LRP1', 'TREM2']}


def build_env_features(data, n_types=12, n_pcs=20, n_domains=8, k=200, scales=(1.5, 5.0, 15.0), seed=0, verbose=True):
    X = np.asarray(data["expression_all"], np.float32); XY = np.asarray(data["coords_all"], np.float64); N = len(X)
    pi = np.asarray(data["perturbed_indices_in_all"]); ctrl = pi[np.asarray(data["ctrl_indices"])]
    ko = np.setdiff1d(pi, ctrl); donor = np.ones(N, bool); donor[ko] = False; D_idx = np.where(donor)[0]
    genes = [str(g).upper() for g in data["gene_names"]]; gi = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(seed)
    # expression PCs and fine types of the unperturbed cells
    fit = D_idx if len(D_idx) <= 50000 else rng.choice(D_idx, 50000, replace=False)
    pca = PCA(min(n_pcs, X.shape[1] - 1), random_state=seed).fit(X[fit]); P = pca.transform(X[D_idx]).astype(np.float32)
    nt = min(n_types, max(2, len(D_idx) // 20))
    km = KMeans(nt, random_state=seed, n_init=4).fit(P[np.isin(D_idx, fit)] if len(D_idx) > 50000 else P); T = km.predict(P)
    # neighbours among the unperturbed cells (self excluded)
    kk = min(k + 1, len(D_idx)); nn = NearestNeighbors(n_neighbors=kk).fit(XY[D_idx]); dist, nb = nn.kneighbors(XY)
    self_hit = D_idx[nb] == np.arange(N)[:, None]; dist = np.where(self_hit, np.inf, dist)
    d1 = NearestNeighbors(n_neighbors=2).fit(XY).kneighbors(XY)[0][:, 1]; d = float(np.median(d1[d1 > 0]))
    W = {s: np.exp(-(dist / (s * d)) ** 2 / 2) for s in scales}
    Wn = {s: w / np.maximum(w.sum(1, keepdims=True), 1e-12) for s, w in W.items()}
    T_nb = T[nb]; XD = X[D_idx]
    # A composition
    A = []
    for s in scales:
        comp = np.stack([(Wn[s] * (T_nb == t)).sum(1) for t in range(nt)], 1); A += [comp, -(comp * np.log(comp + 1e-10)).sum(1, keepdims=True)]
    A = np.concatenate(A, 1)
    # B state
    B = np.concatenate([np.stack([(Wn[s] * P[nb, c]).sum(1) for c in range(P.shape[1])], 1) for s in scales], 1)
    # C signal exposure at the paracrine scale times the receptor competence of the cell's type
    ligs = [l for l in LR if l in gi]; C = np.zeros((N, len(ligs)), np.float32)
    ctype = np.asarray(data["cell_types_all"]).astype(str) if data.get("cell_types_all") is not None else np.array(['all'] * N)
    for j, l in enumerate(ligs):
        expo = (Wn[scales[1]] * XD[nb, gi[l]]).sum(1)
        recs = [gi[r] for r in LR[l] if r in gi]
        if recs:
            comp_t = {t: XD[ctype[D_idx] == t][:, recs].mean() if (ctype[D_idx] == t).any() else XD[:, recs].mean() for t in np.unique(ctype)}
            expo = expo * np.array([comp_t[t] for t in ctype])
        C[:, j] = expo
    # D structure
    dens = np.stack([W[s].sum(1) for s in scales], 1)
    dist_t = np.stack([np.where(T_nb == t, dist, np.inf).min(1) for t in range(nt)], 1)
    dist_t = np.log1p(np.where(np.isfinite(dist_t), dist_t, dist[np.isfinite(dist)].max()) / d)
    dom = KMeans(n_domains, random_state=seed, n_init=4).fit_predict(A[:, -(nt + 1):-1]); Dm = np.eye(n_domains, dtype=np.float32)[dom]
    Dd = np.concatenate([dens, dist_t, Dm], 1)
    blocks = {'A': A, 'B': B, 'C': C, 'D': Dd}; F, sl, o = [], {}, 0
    for name, M in blocks.items():
        M = np.asarray(M, np.float32); sd = M.std(0); M = (M - M.mean(0)) / np.where(sd > 1e-8, sd, 1.0); F.append(M); sl[name] = slice(o, o + M.shape[1]); o += M.shape[1]
    F = np.concatenate(F, 1)
    if verbose:
        print(f"  environment description: {N} cells, {len(D_idx)} unperturbed, d = {d:.1f}, scales {scales} x d, {nt} fine types, {len(ligs)} ligands "
              f"-> A {A.shape[1]} + B {B.shape[1]} + C {C.shape[1]} + D {Dd.shape[1]} = {F.shape[1]} dimensions")
    return F, sl
