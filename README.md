# TMEHyper

Spatial perturbation-response prediction with a group hypergraph of the tumour microenvironment. Python 3.10, PyTorch 2.4.

```
pip install -r requirements.txt
pip install --no-deps scgpt==0.2.4
pip install -e .
```

## Data

`data/` holds the input files of GSE274447, SPAC-seq M003 and GSE193460. The Perturb-FISH file (390 MB) is attached to the `data-v1` release:
```
gh release download data-v1 -p PerturbFISH.h5ad -D data
```

## Input

One h5ad file per data set, in `data/`: `obs['perturbation']` (`Control`, the target gene, or `None` for environment-only cells), `obs['cell_type']`, `obs['sample']` (optional section id), `obsm['spatial']`, and `X` as log-normalised expression or raw counts.
`scripts/prepare_gse274447.py` and `scripts/prepare_spac_seq.py` build the GSE274447 and SPAC-seq files from the published matrices.

## Steps

1. Cell embeddings (mouse data sets; Perturb-FISH uses `--scgpt_dir` with the scGPT whole-human checkpoint instead)
   ```
   python scripts/map_panel_to_nicheformer.py panel.txt embeddings/GSE274447_to_nicheformer.json <nicheformer_hf>/vocab.json
   python scripts/embed_nicheformer.py --h5ad data/GSE274447.h5ad --mapping embeddings/GSE274447_to_nicheformer.json --out embeddings/GSE274447.X_niche.npy --hf_dir <nicheformer_hf>
   ```
2. Reference profile
   ```
   python scripts/build_reference.py --data data/GSE274447.h5ad --out references/ref_GSE274447 --gpu 0
   ```
3. Spatial block hold-out (random splits need no file)
   ```
   python scripts/make_block_split.py --data data/GSE274447.h5ad --out splits/GSE274447_block_d0.npz --n_blocks 60 --buffer 3 --min_test 5 --start_seed 0
   ```
   (`--start_seed 1000` and `2000` for the second and third draw)
4. Train and predict
   ```
   python -m tmehyper.main --data data/GSE274447.h5ad --out runs/GSE274447_block_d0 --split file --split_file splits/GSE274447_block_d0.npz \
       --cell_embedding_file embeddings/GSE274447.X_niche.npy --ref_file references/ref_GSE274447.npy --seed 42 --gpu 0
   ```
   `--split random --seed <s>` for the random split. `--epochs 2` runs a quick check of the whole pipeline.
5. Evaluate
   ```
   python scripts/evaluate.py --data data/GSE274447.h5ad --name GSE274447 --split_name block_d0 --run runs/GSE274447_block_d0 --min_test 10 --exclude msafe --out results
   ```
   Writes the seven reported metrics per unit (`units_*.csv`) and their mean over the units (`mean_*.csv`).

Model and training hyper-parameters are in `tmehyper/config.py`.

License: MIT (see `LICENSE`).
