"""Spatial block hold-out.  The perturbed-set cells are tiled into spatial blocks (k-means on their coordinates); whole blocks are held
out as the test set until the requested fraction of the knock-out cells is in it, and every training knock-out cell closer than
buffer x d_cell to a test knock-out cell is dropped (neither train nor test).  Blocks are re-drawn (different seeds) until every
perturbation has enough test and training cells.
    python scripts/make_block_split.py --data GSE274447.h5ad --out splits/GSE274447_block_d0.npz --n_blocks 60 --buffer 3 --min_test 5 --start_seed 0
Paper settings: 60 blocks, test fraction 0.2, buffer 3 d_cell (2 for SPAC-seq and GSE193460), min_test 5 (10 for Perturb-FISH), min_train 10,
draws d0 / d1 / d2 = --start_seed 0 / 1000 / 2000."""
import os, sys, argparse, numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tmehyper.config import Config
from tmehyper.data import load_data
TEST_FRAC = 0.2; MIN_TRAIN = 10; TRIES = 300
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--n_blocks', type=int, default=60)
ap.add_argument('--buffer', type=float, default=3.0, help='in units of d_cell')
ap.add_argument('--min_test', type=int, default=5); ap.add_argument('--start_seed', type=int, default=0, help='first draw seed (0, 1000, 2000 for the three draws)')
a = ap.parse_args()
cfg = Config(); cfg.data_file = a.data; cfg.device = 'cpu'
dd = load_data(cfg)
coords = np.asarray(dd['coords'], np.float64); lab = np.asarray(dd['perturb_labels']); names = [str(n) for n in dd['perturb_names']]; ko = np.asarray(dd['ko_indices'])
d1 = NearestNeighbors(n_neighbors=2).fit(coords).kneighbors(coords)[0][:, 1]; d_cell = float(np.median(d1[d1 > 0])); radius = a.buffer * d_cell
blk = KMeans(n_clusters=a.n_blocks, random_state=0, n_init=3).fit_predict(coords)
perts = [p for p in range(len(names)) if names[p] != 'Control' and (lab[ko] == p).sum() > 0]
best = None
for seed in range(a.start_seed, a.start_seed + TRIES):
    rng = np.random.RandomState(seed); order = rng.permutation(a.n_blocks); test_blocks = []; n_test = 0
    for b in order:
        if n_test >= TEST_FRAC * len(ko): break
        test_blocks.append(b); n_test += int(np.isin(blk[ko], [b]).sum())
    te = ko[np.isin(blk[ko], test_blocks)]; rest = np.setdiff1d(ko, te)
    near = NearestNeighbors(radius=radius).fit(coords[te]).radius_neighbors(coords[rest], return_distance=False)
    tr = rest[np.array([len(x) == 0 for x in near])]
    cnt_te = {p: int((lab[te] == p).sum()) for p in perts}; cnt_tr = {p: int((lab[tr] == p).sum()) for p in perts}
    ok = sum(cnt_te[p] >= a.min_test and cnt_tr[p] >= MIN_TRAIN for p in perts)
    score = (ok, -abs(len(te) / len(ko) - TEST_FRAC))
    if best is None or score > best[0]: best = (score, seed, te, tr, test_blocks, cnt_te, cnt_tr)
    if ok == len(perts): break
(score, seed, te, tr, test_blocks, cnt_te, cnt_tr) = best
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez(a.out, train_ko_indices=np.sort(tr), test_ko_indices=np.sort(te), dropped_ko_indices=np.setdiff1d(ko, np.concatenate([tr, te])), block_of_cell=blk,
         d_cell=d_cell, buffer_radius=radius, seed=seed, test_blocks=np.array(test_blocks))
print(f'{a.data}: {len(ko)} knock-out cells, d_cell = {d_cell:.1f}, buffer {a.buffer} x d_cell = {radius:.1f}, {a.n_blocks} blocks, draw seed {seed}')
print(f'  train {len(tr)}  test {len(te)}  dropped by the buffer {len(ko) - len(tr) - len(te)}  ({len(te) / len(ko) * 100:.1f}% test)')
print(f'  perturbations with >= {a.min_test} test and >= {MIN_TRAIN} training cells: {score[0]}/{len(perts)}')
print('  test cells per perturbation:', {names[p]: cnt_te[p] for p in perts})
