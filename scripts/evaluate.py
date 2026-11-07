"""The seven reported metrics of a TMEHyper run on one data set and one split.

Units = perturbations with >= min_test test cells and >= min_train training cells, minus --exclude.
Metrics per unit (pr = predicted mean response over the unit's test cells, t = observed mean response, r = mean of the training per-perturbation means):
  Pearson delta                 Pearson(pr, t)
  DEG PCC@100 / @20             Pearson(pr - r, t - r) on the 100 / 20 genes with the largest |t - r|  (shared response removed)
  DEG recall@50                 overlap of the 50 genes with the largest |pr| / s_ctrl with the 50 top genes of a Wilcoxon (Mann-Whitney) test of
                                the test cells against the control cells
  MSE (all genes / top-20 DEG)  mean (pr - t)^2 over all genes / the 20 genes with the largest |t|
  MMD                           Gaussian-kernel MMD between the test cells and pr + within-perturbation noise of the training cells
Paper settings: --min_test 10 (5 for SPAC-seq and GSE193460), --min_train 10, --exclude msafe on GSE274447 (safe-targeting guide).
    python scripts/evaluate.py --data GSE274447.h5ad --name GSE274447 --split_name block_d0 --run runs/GSE274447_block_d0 --min_test 10 --exclude msafe --out results
Writes <out>/units_<name>_<split_name>.csv (one row per unit) and <out>/mean_<name>_<split_name>.csv (mean over the units)."""
import os, sys, argparse, numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tmehyper.config import Config
from tmehyper.data import load_data
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--name', required=True, help='data set label')
ap.add_argument('--split_name', required=True, help='split label, e.g. random_s42 or block_d0')
ap.add_argument('--run', required=True, help='TMEHyper run directory (pred_cells.npz, split.npz)')
ap.add_argument('--min_test', type=int, default=10); ap.add_argument('--min_train', type=int, default=10)
ap.add_argument('--exclude', default='', help='comma-separated perturbations left out of the units'); ap.add_argument('--out', default='results')
a = ap.parse_args()
CAP = 300; K = 50; EXC = set(x for x in a.exclude.split(',') if x)
cfg = Config(); cfg.data_file = a.data; cfg.device = 'cpu'; dd = load_data(cfg)
D = dd['expression'] - dd['ctrl_mean']; lab = np.asarray(dd['perturb_labels']); names = [str(n) for n in dd['perturb_names']]; ctrl = np.asarray(dd['ctrl_indices'])
sp = np.load(os.path.join(a.run, 'split.npz'), allow_pickle=True); tr, te = sp['train_ko_indices'], sp['test_ko_indices']
w = np.load(os.path.join(a.run, 'pred_cells.npz'), allow_pickle=True); nm = [str(n) for n in w['names']]
n_tr = {names[p]: int((lab[tr] == p).sum()) for p in range(len(names))}
units = [nm[p] for p in np.unique(w['labels']) if nm[p] != 'Control' and nm[p] not in EXC and (w['labels'] == p).sum() >= a.min_test and n_tr.get(nm[p], 0) >= a.min_train]
pred = {p: w['pred'][w['labels'] == nm.index(p)].mean(0) for p in units}
T = {p: D[te[lab[te] == names.index(p)]] for p in units}; Tm = {p: T[p].mean(0) for p in units}
for p in units:   # the observed responses rebuilt from the data file must be those stored with the predictions
    assert np.allclose(Tm[p], w['true'][w['labels'] == nm.index(p)].mean(0), atol=1e-4), f'truth mismatch {p}'
r = np.mean([D[tr[lab[tr] == p]].mean(0) for p in range(len(names)) if names[p] != 'Control' and (lab[tr] == p).sum()], axis=0)
ctrlD = D[ctrl]; s_ctrl = ctrlD.std(0, ddof=1); n2 = len(ctrlD)
s_ctrl = np.maximum(s_ctrl, np.percentile(s_ctrl[s_ctrl > 0], 10)) + 1e-6   # floor: genes never seen in a small control pool
def wz(X):
    u = mannwhitneyu(X, ctrlD, axis=0).statistic; n1 = len(X); return np.nan_to_num((u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12))
deg = {p: set(np.argsort(-np.abs(wz(T[p])))[:K]) for p in units}
def mmd2(S, Tt):
    sig = np.median(cdist(Tt, Tt)[np.triu_indices(len(Tt), 1)]); g = 1.0 / (2 * sig ** 2 + 1e-12)
    k = lambda A, B: np.exp(-g * cdist(A, B, 'sqeuclidean')).mean(); return float(k(S, S) + k(Tt, Tt) - 2 * k(S, Tt))
rows = []
for p in units:
    pr, t = np.asarray(pred[p], float), Tm[p]; o = np.argsort(np.abs(t - r)); oo = np.argsort(np.abs(t)); g = lambda kk: pearsonr((pr - r)[o[-kk:]], (t - r)[o[-kk:]])[0]
    rng = np.random.RandomState(1000 + units.index(p)); Tc = T[p]; Tc = Tc[rng.choice(len(Tc), CAP, replace=False)] if len(Tc) > CAP else Tc
    pool = D[tr[lab[tr] == names.index(p)]]; pool = pool - pool.mean(0); eps = pool[rng.choice(len(pool), len(Tc), replace=len(pool) < len(Tc))]
    top = set(np.argsort(-np.abs(pr) / s_ctrl)[:K]); e2 = (pr - t) ** 2
    rows.append({'dataset': a.name, 'split': a.split_name, 'pert': p, 'n_test': len(T[p]), 'Pearson delta': pearsonr(pr, t)[0],
                 'DEG PCC@100': g(100), 'DEG PCC@20': g(20), 'DEG recall@50': len(top & deg[p]) / K,
                 'MSE (all genes)': float(e2.mean()), 'MSE (top-20 DEG)': float(e2[oo[-20:]].mean()), 'MMD': mmd2(pr[None, :] + eps, Tc)})
U = pd.DataFrame(rows); MET = ['Pearson delta', 'DEG PCC@100', 'DEG PCC@20', 'DEG recall@50', 'MSE (all genes)', 'MSE (top-20 DEG)', 'MMD']
os.makedirs(a.out, exist_ok=True); tag = f'{a.name}_{a.split_name}'
U.to_csv(f'{a.out}/units_{tag}.csv', index=False)
M = U[MET].mean().to_frame().T; M.insert(0, 'n_units', len(U)); M.insert(0, 'split', a.split_name); M.insert(0, 'dataset', a.name)
M.to_csv(f'{a.out}/mean_{tag}.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(f'\n{a.name} / {a.split_name}: {len(units)} units (>= {a.min_test} test cells, >= {a.min_train} training cells, excluded {sorted(EXC)})\n{units}')
print(U.set_index('pert')[MET].round(3).to_string()); print('\nmean over units'); print(M[MET].round(3).to_string(index=False))
print(f'\nsaved {a.out}/units_{tag}.csv, {a.out}/mean_{tag}.csv')
