"""Group hyperedges with edge-dependent vertex weights, and the hypergraph convolution that splits the cell description into a
low-frequency and a high-frequency component under every family.

A hyperedge is a group of cells (a tissue domain, the cells of one type inside one domain, a ligand source with its diffusion
range) and the weight gamma(v, e) of a cell depends on the hyperedge.  One layer of hypergraph convolution, per family k:
    y_e      = rho_k( sum_v gamma(v,e) phi(F_v) / sum_v gamma(v,e) )   vertex -> hyperedge, over the UNPERTURBED members of e
    low_v    = sum_e gamma~(v,e) y_e                                   hyperedge -> vertex, gamma~ = gamma normalised per cell (32 strongest)
    high_v   = phi(F_v) - low_v
rho_k is a learned layer per family.  With rho_k the identity, low = P_k Phi and high = (I - P_k) Phi, the low-pass (response lambda) and
high-pass (response 1 - lambda) filters of the description over the hypergraph, P_k being the random walk with edge-dependent vertex
weights.
Families (every weight comes from the data; the kernel scales are those of env_features):
  domain   one hyperedge per spatial domain; gamma = tissue-scale (15 d) kernel fraction of the 200 nearest unperturbed cells in it
  niche    one per (cell type, domain); gamma = domain weight x [type(v) = t]; needs 20 unperturbed members
  source   one per spatial cluster (DBSCAN, 3 d) of unperturbed cells in the top decile of a ligand's expression; gamma = paracrine
           (5 d) kernel-weighted ligand mass reaching v (within 12 d) x receptor competence of v's type
A family the data cannot define gives every cell a learned null low-frequency component."""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from .env_features import LR

FAMILIES = ("domain", "niche", "source")


def nn_scale(XY):
    d1 = NearestNeighbors(n_neighbors=2).fit(XY).kneighbors(XY)[0][:, 1]
    return float(np.median(d1[d1 > 0])) if (d1 > 0).any() else 1.0


def build_group_hyperedges(data, F_env, slices, config):
    """-> (families: {name: csr [N_all, E] with gamma >= 0}, donor mask [N_all] of the unperturbed cells)."""
    XY = np.asarray(data["coords_all"], np.float64); N = len(XY)
    pi = np.asarray(data["perturbed_indices_in_all"]); ctrl = pi[np.asarray(data["ctrl_indices"])]
    ko = np.setdiff1d(pi, ctrl); donor = np.ones(N, bool); donor[ko] = False; D_idx = np.where(donor)[0]
    d = nn_scale(XY)
    ctype = np.asarray(data["cell_types_all"]).astype(str) if data.get("cell_types_all") is not None else np.array(["all"] * N)
    nD = int(config.env_n_domains); dom = np.argmax(F_env[:, slices["D"]][:, -nD:], 1)
    # soft domain membership: tissue-scale kernel fraction of the 200 nearest unperturbed cells in each domain (self excluded)
    kk = min(201, len(D_idx)); dist, nb = NearestNeighbors(n_neighbors=kk).fit(XY[D_idx]).kneighbors(XY)
    dist = np.where(D_idx[nb] == np.arange(N)[:, None], np.inf, dist)
    w = np.exp(-(dist / (15.0 * d)) ** 2 / 2); w /= np.maximum(w.sum(1, keepdims=True), 1e-12); dom_nb = dom[D_idx[nb]]
    M_dom = np.stack([(w * (dom_nb == e)).sum(1) for e in range(nD)], 1).astype(np.float32)
    out = {}; info = []
    # domain
    out["domain"] = sp.csr_matrix(np.where(M_dom >= 0.02, M_dom, 0.0).astype(np.float32)); info.append(f"domain {nD}")
    # niche
    types = np.unique(ctype); T = (ctype[:, None] == types[None, :]).astype(np.float32)
    M = (M_dom[:, :, None] * T[:, None, :]).reshape(N, -1); M = sp.csr_matrix(np.where(M >= 0.02, M, 0.0).astype(np.float32))
    keep = np.asarray((M[D_idx] > 0).sum(0)).ravel() >= 20
    out["niche"] = M[:, np.where(keep)[0]].tocsr(); info.append(f"niche {int(keep.sum())} of {len(types)} types x {nD} domains")
    # source
    X = np.asarray(data["expression_all"], np.float32); genes = [str(g).upper() for g in data["gene_names"]]; gi = {g: i for i, g in enumerate(genes)}
    ligs = [l for l in LR if l in gi]; max_src = int(config.group_max_sources); max_mem = 150
    tree = NearestNeighbors(radius=12.0 * d).fit(XY); rng = np.random.default_rng(0)
    rows, cols, vals, e, used = [], [], [], 0, set()
    for l in ligs:
        x = X[D_idx, gi[l]]
        if (x > 0).sum() < 20: continue
        src = D_idx[x >= np.quantile(x[x > 0], 0.90)]
        if len(src) < 3: continue
        lab = DBSCAN(eps=3.0 * d, min_samples=3).fit_predict(XY[src])
        recs = [gi[r] for r in LR[l] if r in gi]; comp = np.ones(N, np.float32)
        if recs:
            XD = X[D_idx][:, recs].mean(1)
            comp_t = {t: float(XD[ctype[D_idx] == t].mean()) if (ctype[D_idx] == t).any() else float(XD.mean()) for t in np.unique(ctype)}
            comp = np.array([comp_t[t] for t in ctype], np.float32)
        clusters = sorted([np.where(lab == c)[0] for c in np.unique(lab) if c >= 0], key=len, reverse=True)[:max_src]
        for mem in clusters:
            cells = src[mem]; scale = 1.0
            if len(cells) > max_mem: scale = len(cells) / max_mem; cells = rng.choice(cells, max_mem, replace=False)
            dl, il = tree.radius_neighbors(XY[cells], return_distance=True)
            xl = X[cells, gi[l]]
            r_ = np.concatenate(il).astype(np.int64); dd_ = np.concatenate(dl); m_ = np.repeat(xl, [len(a) for a in il])
            v_ = np.exp(-(dd_ / (5.0 * d)) ** 2 / 2) * m_ * scale * comp[r_]
            rows.append(r_); cols.append(np.full(len(r_), e)); vals.append(v_.astype(np.float32)); used.add(l); e += 1
    if e:
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, e)); M.sum_duplicates()
        M.data[M.data < 1e-5 * M.data.max()] = 0; M.eliminate_zeros()
        out["source"] = M; info.append(f"source {e} clusters of {len(used)} ligands")
    else:
        info.append("source 0 (no ligand with enough expressing unperturbed cells)")
    cov = {f: float((np.asarray(M.sum(1)).ravel() > 0).mean()) for f, M in out.items()}
    print(f"  group hyperedges: {N} cells, {len(D_idx)} unperturbed, d = {d:.1f}; " + "; ".join(info) + "; cells with a membership: " + ", ".join(f"{f} {c:.2f}" for f, c in cov.items()))
    return out, donor


def normalise_memberships(M, donor):
    """-> (Agg csr [E, N]: rows over unperturbed members sum to 1, hyperedges without members dropped;
            Mem csr [N, E]: rows sum to 1 or 0; kept hyperedge ids)"""
    M = M.tocsr(); A = M.T.tocsr().multiply(donor[None, :].astype(np.float32)).tocsr()
    rs = np.asarray(A.sum(1)).ravel(); keep = np.where(rs > 0)[0]
    A = sp.diags(1.0 / rs[keep]) @ A[keep]; Mk = M[:, keep].tocsr(); ms = np.asarray(Mk.sum(1)).ravel()
    Mem = sp.diags(np.where(ms > 0, 1.0 / np.maximum(ms, 1e-12), 0.0)) @ Mk
    return A.tocsr(), Mem.tocsr(), keep


class GroupHypergraphEncoder(nn.Module):
    """Hypergraph convolution on the group hyperedges.  last_comps [B, 2 * n_families, 64] holds the (low, high) frequency components
    of the batch cells, read by the response head."""

    def __init__(self, config, input_dim):
        super().__init__()
        te = int(config.env_embed_dim); dropout = float(config.dropout)
        self.families = list(FAMILIES)
        self.hidden = 64; nf = len(self.families); self.n_comps = 2 * nf
        self.phi = nn.Sequential(nn.Linear(input_dim, self.hidden), nn.LayerNorm(self.hidden), nn.GELU())
        self.rho = nn.ModuleDict({f: nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.LayerNorm(self.hidden), nn.GELU()) for f in self.families})
        self.null = nn.ParameterDict({f: nn.Parameter(torch.zeros(self.hidden)) for f in self.families})
        self.fusion = nn.Sequential(nn.Linear(self.n_comps * self.hidden, te), nn.LayerNorm(te), nn.GELU(), nn.Dropout(dropout), nn.Linear(te, te))
        self.F_all = None; self.agg = {}; self.mem_idx = {}; self.mem_w = {}; self.has = {}; self.node_to_all = None
        self.last_comps = None
        print(f"environment encoder: group hypergraph convolution, families {self.families}, {input_dim} -> {self.hidden}, {self.n_comps} frequency components -> {te}")

    def set_group_env(self, F_all, fams, donor, node_to_all, device, k_mem=32):
        self.F_all = torch.as_tensor(np.asarray(F_all, np.float32), device=device)
        self.node_to_all = torch.as_tensor(np.asarray(node_to_all), dtype=torch.long, device=device); N = self.F_all.shape[0]
        for f in self.families:
            if f not in fams or fams[f].shape[1] == 0:
                self.agg[f] = torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long), torch.zeros(0), size=(1, N), device=device).coalesce()
                self.mem_idx[f] = torch.zeros((N, k_mem), dtype=torch.long, device=device); self.mem_w[f] = torch.zeros((N, k_mem), device=device)
                self.has[f] = torch.zeros(N, device=device)
                print(f"  family {f}: no hyperedges in this data set (null component for every cell)"); continue
            A, Mem, _ = normalise_memberships(fams[f], donor); A = A.tocoo()
            self.agg[f] = torch.sparse_coo_tensor(torch.as_tensor(np.vstack([A.row, A.col]), dtype=torch.long), torch.as_tensor(A.data, dtype=torch.float32), size=A.shape, device=device).coalesce()
            Mem = Mem.tocsr(); idx = np.zeros((N, k_mem), np.int64); w = np.zeros((N, k_mem), np.float32)
            for i in range(N):
                a, b = Mem.indptr[i], Mem.indptr[i + 1]
                if b == a: continue
                c, v = Mem.indices[a:b], Mem.data[a:b]
                if len(c) > k_mem: o = np.argsort(-v)[:k_mem]; c, v = c[o], v[o]
                idx[i, :len(c)] = c; w[i, :len(c)] = v / v.sum()
            self.mem_idx[f] = torch.as_tensor(idx, device=device); self.mem_w[f] = torch.as_tensor(w, device=device)
            self.has[f] = (self.mem_w[f].sum(1) > 0).float()
            print(f"  family {f}: {A.shape[0]} hyperedges, cells with a membership {float(self.has[f].mean()):.2f}")

    def _edges(self):
        """Phi = phi(F) of every cell and, per family, the hyperedge embeddings y_e = rho_k(mean of the unperturbed members) (vertex -> hyperedge)."""
        H = self.phi(self.F_all); means = {f: torch.sparse.mm(self.agg[f], H) for f in self.families}
        return H, {f: self.rho[f](m) for f, m in means.items()}

    def _comps(self, g, H, he):
        """hyperedge -> vertex: low = sum_e gamma~(v,e) y_e and high = phi(F_v) - low for the cells g."""
        comps = []
        for f in self.families:
            low = (he[f][self.mem_idx[f][g]] * self.mem_w[f][g][..., None]).sum(1) + (1.0 - self.has[f][g])[:, None] * self.null[f]
            comps += [low, H[g] - low]
        return comps

    def forward(self, center_indices, return_all=False):
        if self.F_all is None: raise ValueError("call set_group_env() first")
        H, he = self._edges(); g = self.node_to_all[center_indices.long()]
        comps = self._comps(g, H, he); self.last_comps = torch.stack(comps, 1)
        z = self.fusion(torch.cat(comps, -1))
        if return_all:
            return z, self.fusion(torch.cat(self._comps(self.node_to_all, H, he), -1))
        return z, None
