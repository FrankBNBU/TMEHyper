"""Hyperedge regularisers on the environment embedding.

Cohesion hyperedges: for every cell, its k spatial neighbours weighted by the product of a spatial fall-off
g_ij = exp(-d_ij^2 / 2 (sigma d_cell)^2) and a feature similarity s_ij = exp(-||F_i - F_j||^2 / 2 tau_i^2), tau_i the median
feature distance to the neighbours; the members with g_ij s_ij >= 0.5 (plus the cell itself) form one hyperedge, kept when it
has at least five members.  Cohesion pulls the environment embeddings of the members of a hyperedge towards their mean;
separation keeps the means of different hyperedges at least a margin apart.  A random subset of the hyperedges is used in every batch."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


def build_cohesion_hyperedges(coords, features, k=30, sigma=5.0, min_cells=5, n_query=200):
    """The k nearest spatial neighbours are taken from a query of the n_query nearest (the order of equidistant cells follows that query)."""
    N = len(coords); coords = np.asarray(coords, np.float64); Fz = np.asarray(features, np.float64)
    nn = NearestNeighbors(n_neighbors=max(k, n_query) + 1).fit(coords)
    dist_all, idx_all = nn.kneighbors(coords); dist_all, idx_all = dist_all[:, 1:], idx_all[:, 1:]
    d1 = dist_all[:, 0]; d_cell = float(np.median(d1[d1 > 0])) if (d1 > 0).any() else 1.0
    Fz = (Fz - Fz.mean(0)) / (Fz.std(0) + 1e-6)
    nb = idx_all[:, :k]; dn = dist_all[:, :k]
    dF = np.empty((N, k))
    for a in range(0, N, 4096):
        b = min(a + 4096, N); dF[a:b] = np.linalg.norm(Fz[a:b, None, :] - Fz[nb[a:b]], axis=2)
    tau = np.median(dF, axis=1, keepdims=True) + 1e-6
    sim = np.exp(-dF ** 2 / (2 * tau ** 2)); g = np.exp(-dn ** 2 / (2 * (sigma * d_cell) ** 2))
    ws = g * sim
    edges = [set(nb[i][ws[i] >= 0.5].tolist()) | {i} for i in range(N)]
    edges = [s for s in edges if len(s) >= min_cells]
    print(f"  cohesion hyperedges: {N} cells, k = {k}, d_cell = {d_cell:.1f}, {len(edges)} hyperedges with >= {min_cells} members")
    return edges


class HyperedgeRegulariser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cohesion_weight = float(config.cohesion_weight); self.separation_weight = float(config.separation_weight)
        self.margin = float(config.separation_margin); self.max_edges = int(config.regulariser_max_edges)

    def _sample(self, edges):
        if self.max_edges and len(edges) > self.max_edges:
            idx = torch.randperm(len(edges))[:self.max_edges].tolist()
            return [edges[i] for i in idx]
        return edges

    def cohesion(self, z, edges):
        total, n = 0.0, 0
        for e in self._sample(edges):
            idx = [i for i in e if i < z.size(0)]
            if len(idx) < 2: continue
            emb = z[idx]; total = total + (emb - emb.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1).mean(); n += 1
        return total / n if n else torch.tensor(0.0, device=z.device)

    def separation(self, z, edges, max_pairs=500):
        means = [z[[i for i in e if i < z.size(0)]].mean(dim=0) for e in edges if any(i < z.size(0) for i in e)]
        if len(means) < 2: return torch.tensor(0.0, device=z.device)
        embs = torch.stack(means); n = embs.size(0)
        if n * (n - 1) // 2 > max_pairs:
            i1 = torch.randint(0, n, (max_pairs,), device=z.device); i2 = torch.randint(0, n, (max_pairs,), device=z.device)
            same = i1 == i2; i2[same] = (i2[same] + 1) % n
        else:
            i1, i2 = zip(*[(i, j) for i in range(n) for j in range(i + 1, n)])
            i1 = torch.tensor(i1, device=z.device); i2 = torch.tensor(i2, device=z.device)
        dist = (embs[i1] - embs[i2]).norm(dim=-1)
        return (F.relu(self.margin - dist) ** 2).mean()

    def forward(self, z, edges):
        sep_edges = self._sample(edges)   # the two terms use independent random subsets of the hyperedges
        out = {"cohesion": self.cohesion(z, edges), "separation": self.separation(z, sep_edges)}
        out["total"] = self.cohesion_weight * out["cohesion"] + self.separation_weight * out["separation"]
        return out
