"""SPAC-seq lung metastasis (Visium HD, 8 um bins) pooled into k x k-bin pseudo-spots (k = 6 -> 48 um).
Only bins with the authors' cell-type annotation are used.  Per spot: counts = sum over its bins; perturbation = the authors' clone label that
covers >= --maj of the spot's bins ('sgnon-targeting' -> Control, 'sgX' -> X); spots without a clone bin -> None (environment); spots with clone
bins but no majority -> Multiple (kept in the tissue, excluded from the model).  Spots with fewer than
--min_bins tissue bins (tissue edge) are dropped.  Panel: 500 highly variable genes (seurat_v3 on spot counts).
cell_type: Tumor for clone-bearing spots, otherwise the authors' majority bin type (T_cell, B_cell, Macrophage, Neutrophil, DC, Stromal_lung);
the structural type never comes from the spot's own expression.  Perturbations with fewer than --min_spots spots -> Multiple.
    python scripts/prepare_spac_seq.py --sample <Lung_Metastasis_M003> --out SPAC_LM_M003.h5ad
<sample>/transcriptome/filtered_gene_bc_matrix.h5, <sample>/annotation/*.csv (index, cell_type), <sample>/clone/*.csv (index, clone)."""
import os, glob, argparse, numpy as np, pandas as pd, anndata as ad, scanpy as sc, scipy.sparse as sp
ap = argparse.ArgumentParser()
ap.add_argument('--sample', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--k', type=int, default=6); ap.add_argument('--maj', type=float, default=0.5); ap.add_argument('--min_bins', type=int, default=18)
ap.add_argument('--n_genes', type=int, default=500); ap.add_argument('--min_spots', type=int, default=10)
a = ap.parse_args(); S = a.sample
T = ad.read_h5ad(f'{S}/transcriptome/filtered_gene_bc_matrix.h5')
ct = pd.read_csv(glob.glob(f'{S}/annotation/*.csv')[0]).set_index('index').cell_type
cl = pd.read_csv(glob.glob(f'{S}/clone/*.csv')[0]).set_index('index').clone
keep = T.obs_names.isin(ct.index); T = T[keep].copy(); X = sp.csr_matrix(T.X)
typ = ct.reindex(T.obs_names).values; lab = cl.reindex(T.obs_names).fillna('').values
r, c = T.obs.array_row.values // a.k, T.obs.array_col.values // a.k
sid = pd.Index(r.astype(np.int64) * 100000 + c); uniq, inv = np.unique(sid, return_inverse=True)
M = sp.csr_matrix((np.ones(len(inv), np.float32), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
C = (M @ X).tocsr(); n_bins = np.asarray(M.sum(1)).ravel()
df = pd.DataFrame({'s': inv, 't': typ, 'l': lab})
cell_type = df.groupby('s').t.agg(lambda x: x.value_counts().index[0]).reindex(range(len(uniq))).values
tum_frac = df.assign(tu=df.t == 'Non-immune cell').groupby('s').tu.mean().reindex(range(len(uniq))).values
lb = df[df.l != '']; cnt = lb.groupby(['s', 'l']).size().reset_index(name='n').sort_values('n', ascending=False).drop_duplicates('s').set_index('s')
n_lab = lb.groupby('s').size()
pert = np.array(['None'] * len(uniq), dtype=object)
has = n_lab.reindex(range(len(uniq))).fillna(0).values > 0; pert[has] = 'Multiple'
top = cnt.reindex(range(len(uniq))); ok = (top.n.fillna(0).values / np.maximum(n_bins, 1)) >= a.maj
pert[ok] = [('Control' if x == 'sgnon-targeting' else x[2:]) for x in top.l.values[ok]]
sr, scn = (uniq // 100000), (uniq % 100000)
good = n_bins >= a.min_bins
print(f'{len(uniq)} spots, {int(good.sum())} with >= {a.min_bins} tissue bins; labels:', pd.Series(pert[good]).value_counts().head(6).to_dict())
A = ad.AnnData(X=C[good].astype(np.float32), obs=pd.DataFrame({'sample': os.path.basename(S).replace('Lung_Metastasis_', ''), 'cell_type': cell_type[good], 'tumour_frac': tum_frac[good],
               'n_bins': n_bins[good], 'perturbation': pd.Categorical(pert[good]), 'array_row': sr[good], 'array_col': scn[good]},
               index=[f's{u}' for u in uniq[good]]), var=pd.DataFrame(index=T.var_names.astype(str)))
A.var_names_make_unique()
A.obsm['spatial'] = np.c_[scn[good], sr[good]].astype(np.float32) * a.k * 8.0          # micrometres
A.layers['counts'] = A.X.copy()
hv = sc.pp.highly_variable_genes(A, flavor='seurat_v3', n_top_genes=a.n_genes, inplace=False)
A = A[:, hv.highly_variable.values].copy(); del A.layers['counts']
A.obs['bin_major_type'] = A.obs.cell_type.values
_MAP = {'T': 'T_cell', 'B': 'B_cell', 'Macrophage': 'Macrophage', 'Neutrophil': 'Neutrophil', 'DC': 'DC', 'Non-immune cell': 'Stromal_lung'}
A.obs['cell_type'] = pd.Categorical(np.where(A.obs.perturbation.astype(str).values != 'None', 'Tumor', pd.Series(A.obs.bin_major_type.astype(str).values).map(_MAP).fillna('Stromal_lung').values))
pc0 = A.obs.perturbation.astype(str).value_counts(); rare = [p for p, n in pc0.items() if p not in ('None', 'Multiple', 'Control') and n < a.min_spots]
A.obs['perturbation'] = pd.Categorical(np.where(A.obs.perturbation.astype(str).isin(rare), 'Multiple', A.obs.perturbation.astype(str)))
print(f'{len(rare)} perturbations with < {a.min_spots} spots -> Multiple')
pc = A.obs.perturbation.value_counts(); kos = pc.drop(['None', 'Multiple', 'Control'], errors='ignore')
print(f'genes {A.n_vars}; Control {int(pc.get("Control", 0))}; perturbations {len(kos)}, with >= 20 spots {int((kos >= 20).sum())}')
print('cell types:', A.obs.cell_type.value_counts().to_dict())
A.uns['source'] = (f'SPAC-seq lung metastasis {os.path.basename(S)} (Visium HD 8 um bins, authors\' cell-type and clone annotation) pooled into '
                   f'{8 * a.k} um pseudo-spots; perturbation = clone label covering >= {a.maj:.0%} of the spot; Control = non-targeting clone; 500 HVG')
A.write_h5ad(a.out); print('wrote', a.out, A.shape)
