"""Environment-matched reference profile of every perturbed-set cell (tmehyper/reference.py), the same setting for every data set.

1. environment description of every cell from the unperturbed cells, PCA-20 on the perturbed set;
2. per cell type, semi-relaxed entropic OT from every perturbed-set cell to its 300 nearest control cells, cost = spatial + environment
   distance (w_e = 1), tau = 1, at eps 0.05, 0.1, 0.2 and 0.5, each followed by the per-gene shrinkage to the type mean;
3. the reference = mean of the four shrunk references.  No knock-out expression is used; the reference is the same for every split.
Writes <out>.npy (rows = the perturbed set in loader order), <out>.obs_names.txt and <out>.json.
    python scripts/build_reference.py --data GSE274447.h5ad --out references/ref_GSE274447 [--gpu 0]"""
import os, sys, json, argparse, numpy as np, torch
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tmehyper.config import Config
from tmehyper.data import load_data
from tmehyper.env_features import build_env_features
from tmehyper.reference import OTMatcher, EPS_LIST, TAU, W_E, K_CONTROLS
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--gpu', type=int, default=None)
a = ap.parse_args()
cfg = Config(); cfg.data_file = a.data; cfg.device = 'cpu'
dev = torch.device(f'cuda:{a.gpu}' if a.gpu is not None and torch.cuda.is_available() else 'cpu')
dd = load_data(cfg); F, sl = build_env_features(dd, n_domains=int(cfg.env_n_domains))
E = np.asarray(dd['expression'], np.float32); ctrl = np.asarray(dd['ctrl_indices']); pi = np.asarray(dd['perturbed_indices_in_all'])
ct = np.asarray(dd['cell_types']).astype(str) if dd.get('cell_types') is not None else np.array(['all'] * len(E))
XY = np.asarray(dd['coords'], np.float64); Z = PCA(20, random_state=0).fit_transform(F[pi]).astype(np.float32)
M = OTMatcher(E, ctrl, ct, XY, Z, K=K_CONTROLS, device=dev)
R = M.multiscale_reference()
MU = M.local_means(EPS_LIST[0])[1]
loo = ((E[ctrl] - R[ctrl]) ** 2).mean() / ((E[ctrl] - MU[ctrl]) ** 2).mean()
print(f'reference: eps {EPS_LIST}, tau {TAU}, w_e {W_E}, K {K_CONTROLS}; control leave-one-out error {loo:.4f} x the type mean')
out = os.path.splitext(a.out)[0] if a.out.endswith('.npy') else a.out
os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
np.save(out + '.npy', R.astype(np.float32)); open(out + '.obs_names.txt', 'w').write('\n'.join(map(str, dd['obs_names'])))
json.dump({'eps_list': list(EPS_LIST), 'tau': TAU, 'w_e': W_E, 'K': K_CONTROLS, 'control_loo_vs_type_mean': float(loo)}, open(out + '.json', 'w'), indent=1)
print(f'saved {out}.npy {R.shape}')
