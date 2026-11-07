"""Environment-matched reference profile by semi-relaxed entropic optimal transport, and its calibration.

Per cell type: sources = every perturbed-set cell of the type (knock-out and control), targets = the type's control cells; each
source is connected to its K = 300 nearest controls in space (a control is never matched to itself).
    cost  C_ij = |s_i - s_j|^2 / sigma_s^2 + w_e |F_i - F_j|^2 / sigma_e^2
          s = coordinates (sections separated), F = PCA-20 of the environment description (built from unperturbed cells only),
          sigma_s = median distance to the K-th control, sigma_e = median environment distance
    plan  rows fixed (every cell fully matched), columns KL-relaxed with strength tau; log-domain Sinkhorn
    local mean l_i = sum_j pi_ij x_j, then per-gene shrinkage to the leave-one-out type mean, x~_i = mu_t(i) + lambda_g (l_i - mu_t(i)),
    with lambda_g in {0, 0.25, 0.5, 0.75, 1} chosen by the control leave-one-out error.
The reference of every data set (scripts/build_reference.py) is the mean of the shrunk references at eps = 0.05, 0.1, 0.2 and 0.5 with
tau = 1 and w_e = 1, the same setting for every data set; no knock-out expression is used.  The offset added to the prediction is
alpha_g (x~_ig - c_g), with alpha_g fitted per gene on the training knock-out cells of the split (calibrate_offset)."""
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

EPS_LIST = (0.05, 0.1, 0.2, 0.5); TAU = 1.0; W_E = 1.0; K_CONTROLS = 300


def _sinkhorn(C, col, n_col, eps, tau, iters, dev):
    """semi-relaxed entropic OT on a sparse support; C [n, K] (inf = forbidden), col [n, K] column ids -> row-normalised plan [n, K]"""
    C = torch.as_tensor(C, dtype=torch.float64, device=dev); col = torch.as_tensor(col, dtype=torch.long, device=dev); n = C.shape[0]
    logK = -C / eps; loga = torch.full((n,), -np.log(n), dtype=torch.float64, device=dev); logb = torch.full((n_col,), -np.log(n_col), dtype=torch.float64, device=dev)
    f = torch.zeros(n, dtype=torch.float64, device=dev); g = torch.zeros(n_col, dtype=torch.float64, device=dev)
    rho = tau / (tau + eps)                                                             # KL relaxation of the column marginal
    for _ in range(iters):
        f = loga - torch.logsumexp(logK + g[col], dim=1)
        M = logK + f[:, None]
        mx = torch.full((n_col,), -np.inf, dtype=torch.float64, device=dev).scatter_reduce(0, col.reshape(-1), M.reshape(-1), reduce="amax")
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        s = torch.zeros(n_col, dtype=torch.float64, device=dev).index_add_(0, col.reshape(-1), torch.exp(M - mx[col]).reshape(-1))
        g = rho * (logb - (torch.log(s + 1e-300) + mx))
    return torch.softmax(logK + g[col], dim=1)


def _weighted_mean(P, E, cand, dev, chunk=2048):
    Et = torch.as_tensor(E, dtype=torch.float32, device=dev); out = []
    for s in range(0, P.shape[0], chunk):
        c = torch.as_tensor(cand[s:s + chunk], dtype=torch.long, device=dev)
        out.append(torch.einsum("nk,nkg->ng", P[s:s + chunk].float(), Et[c]).cpu())
    return torch.cat(out).numpy()


class OTMatcher:
    def __init__(self, E, ctrl, ct, XY, Z, K=K_CONTROLS, device=None):
        self.E = np.asarray(E, np.float32); self.ctrl = np.asarray(ctrl); self.ct = np.asarray(ct).astype(str); n = len(self.E)
        self.dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.types = []
        isc = np.zeros(n, bool); isc[self.ctrl] = True; self.isc = isc
        for t in np.unique(self.ct):
            rows = np.where(self.ct == t)[0]; cg = self.ctrl[self.ct[self.ctrl] == t]
            if len(cg) < 25: cg = self.ctrl                                              # a type with few controls matches against every control
            k = min(K + 1, len(cg)); d, j = NearestNeighbors(n_neighbors=k).fit(XY[cg]).kneighbors(XY[rows]); cand = cg[j]
            selfm = cand == rows[:, None]
            Ds = d ** 2 / (np.median(d[:, -1]) ** 2 + 1e-9)
            dz = ((Z[cand] - Z[rows][:, None, :]) ** 2).sum(-1); De = dz / (np.median(dz) + 1e-9)
            S = self.E[cg].sum(0); mu = np.zeros((len(rows), self.E.shape[1]), np.float32)
            for a, i in enumerate(rows): mu[a] = (S - (self.E[i] if isc[i] else 0)) / (len(cg) - int(isc[i]))   # leave-one-out type mean
            self.types.append(dict(rows=rows, cg=cg, cand=cand, col=j, selfm=selfm, Ds=Ds, De=De, mu=mu))

    def local_means(self, eps, iters=150):
        L = np.zeros_like(self.E); MU = np.zeros_like(self.E)
        for T in self.types:
            C = T["Ds"] + W_E * T["De"]; C = np.where(T["selfm"], np.inf, C)
            P = _sinkhorn(C, T["col"], len(T["cg"]), eps, TAU, iters, self.dev)
            L[T["rows"]] = _weighted_mean(P, self.E, T["cand"], self.dev); MU[T["rows"]] = T["mu"]
        return L, MU

    def shrink(self, L, MU, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
        """per-gene lambda_g on the grid with the smallest control leave-one-out error"""
        c = self.ctrl; dev = L - MU
        errs = np.stack([((self.E[c] - (MU + lam * dev)[c]) ** 2).mean(0) for lam in grid])
        lam = np.array(grid)[errs.argmin(0)]; return (MU + lam[None, :] * dev).astype(np.float32)

    def multiscale_reference(self):
        """mean of the shrunk references at every eps of EPS_LIST"""
        return np.mean([self.shrink(*self.local_means(e)) for e in EPS_LIST], axis=0).astype(np.float32)


def calibrate_offset(E, ctrl_mean, labels, train_idx, R):
    """Per-gene scale of the reference offset on the training knock-out cells only.
    y_i = x_i - c - m_p (m_p = mean training response of perturbation p), o_i = r_i - c,
    alpha_g = clip( sum_i y_ig o_ig / sum_i o_ig^2 , 0, 1 ).  Test cells are never used."""
    E = np.asarray(E, np.float32); C = np.asarray(ctrl_mean, np.float32)[None, :]; lab = np.asarray(labels); tr = np.asarray(train_idx)
    m = np.zeros((int(lab.max()) + 1, E.shape[1]), np.float32)
    for p in np.unique(lab[tr]): m[p] = (E[tr[lab[tr] == p]] - C).mean(0)
    Y = E[tr] - C - m[lab[tr]]; O = np.asarray(R, np.float32)[tr] - C
    alpha = np.clip((Y * O).sum(0) / ((O ** 2).sum(0) + 1e-9), 0.0, 1.0).astype(np.float32)
    print(f"  reference offset calibrated on training knock-out cells: mean alpha {alpha.mean():.2f}, genes with alpha = 0: {(alpha == 0).sum()}/{len(alpha)}")
    return alpha
