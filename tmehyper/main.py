"""Train TMEHyper on one data set and one split, and write the test-cell predictions.

    python -m tmehyper.main --data GSE274447.h5ad --out runs/GSE274447_random_s42 --split random --seed 42 \
        --cell_embedding_file embeddings/GSE274447.X_niche.npy --ref_file references/ref_GSE274447.npy
    python -m tmehyper.main --data GSE274447.h5ad --out runs/GSE274447_block_d0 --split file --split_file splits/GSE274447_block_d0.npz ..."""
import argparse
import json
import os
import numpy as np
import torch
from scipy.stats import pearsonr

from .config import Config
from .data import load_data, make_split, attach_reference, create_loaders
from .env_features import build_env_features
from .hyperedges import build_group_hyperedges
from .regulariser import build_cohesion_hyperedges
from .model import TMEHyper, count_parameters
from .train import train, predict, export_predictions


def parse_args():
    ap = argparse.ArgumentParser(description="TMEHyper")
    ap.add_argument("--data", required=True, help="h5ad file"); ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--split", choices=["random", "file"], default="random"); ap.add_argument("--split_file", default=None)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--cell_embedding_file", default=None, help="frozen foundation-model embedding of every cell (.npy + .obs_names.txt)")
    ap.add_argument("--scgpt_dir", default=None, help="scGPT checkpoint directory (vocab.json, args.json, best_model.pt)")
    ap.add_argument("--ref_file", required=True, help="environment-matched reference of the perturbed set (scripts/build_reference.py)")
    ap.add_argument("--epochs", type=int, default=None, help="default 100 as in the paper; a small value gives a quick check of the pipeline")
    return ap.parse_args()


def main():
    a = parse_args(); cfg = Config()
    cfg.data_file = os.path.expanduser(a.data); cfg.seed = a.seed; cfg.split_mode = a.split
    if a.split == "file":
        if not a.split_file: raise SystemExit("--split file needs --split_file")
        cfg.split_file = os.path.expanduser(a.split_file)
    if a.cell_embedding_file: cfg.cell_embedding_file = os.path.expanduser(a.cell_embedding_file)
    if a.scgpt_dir: cfg.scgpt_model_dir = os.path.expanduser(a.scgpt_dir)
    cfg.ref_file = os.path.expanduser(a.ref_file)
    if a.epochs is not None: cfg.epochs = a.epochs
    if a.gpu is not None: cfg.device = f"cuda:{a.gpu}"
    os.makedirs(a.out, exist_ok=True)
    print("configuration:", json.dumps(cfg.to_dict(), indent=1, default=str))
    with open(os.path.join(a.out, "run_config.json"), "w") as f: json.dump(cfg.to_dict(), f, indent=1, default=str)
    torch.manual_seed(int(cfg.seed)); np.random.seed(int(cfg.seed))

    # ---- cell encoder input: a precomputed embedding of every cell, or scGPT on the panel ----
    scgpt_model = vocab = None; use_scgpt = False
    if not cfg.cell_embedding_file:
        if not cfg.scgpt_model_dir: raise SystemExit("give --cell_embedding_file or --scgpt_dir")
        from .encoders import load_scgpt
        scgpt_model, vocab = load_scgpt(cfg.scgpt_model_dir, cfg); use_scgpt = True

    # ---- data, environment description, group hyperedges ----
    data = load_data(cfg, vocab=vocab)
    F_env, slices = build_env_features(data, n_domains=int(cfg.env_n_domains))
    N = data["n_cells_all"]; pi = data["perturbed_indices_in_all"]
    fams, donor = build_group_hyperedges(data, F_env, slices, cfg)
    # the hyperedge regulariser is built on all cells when they fit, else on the perturbed set only
    if N <= int(cfg.max_regulariser_nodes):
        data["index_mode"] = "global"; node_to_all = np.arange(N); hg_coords, hg_feats = data["coords_all"], F_env
    else:
        data["index_mode"] = "local"; node_to_all = pi; hg_coords, hg_feats = data["coords_all"][pi], F_env[pi]
        print(f"  {N} cells > {cfg.max_regulariser_nodes}: regulariser hyperedges on the {len(pi)} perturbed-set cells")
    hyperedges = build_cohesion_hyperedges(hg_coords, hg_feats, k=int(cfg.regulariser_k), sigma=float(cfg.regulariser_sigma))

    # ---- split, reference offset, loaders ----
    train_idx, test_idx = make_split(data, cfg)
    attach_reference(data, cfg, train_idx)
    train_loader, test_loader = create_loaders(data, cfg, train_idx, test_idx, vocab=vocab, use_scgpt_input=use_scgpt)

    # ---- model ----
    emb_dim = int(data["cell_emb"].shape[1]) if data.get("cell_emb") is not None else 0
    model = TMEHyper(cfg.n_genes, len(data["perturb_names"]), cfg, cell_embedding_dim=emb_dim, scgpt_model=scgpt_model,
                     env_input_dim=F_env.shape[1]).to(cfg.device)
    model.env_encoder.set_group_env(F_env, fams, donor, node_to_all, cfg.device)
    pc = count_parameters(model); print(f"parameters: {pc['total']:,} total, {pc['trainable']:,} trainable")

    # ---- train, predict, save ----
    model = train(model, train_loader, cfg, hyperedges)
    pred, true, lab = predict(model, test_loader, cfg)
    export_predictions(a.out, pred, true, lab, data["perturb_names"], data["gene_names"], data["ctrl_mean"])
    np.savez(os.path.join(a.out, "split.npz"), train_ko_indices=np.asarray(train_idx), test_ko_indices=np.asarray(test_idx), split_mode=str(cfg.split_mode))
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    names = data["perturb_names"]; rows = []
    for p in np.unique(lab):
        k = lab == p
        if names[p] != "Control" and k.sum() >= 5: rows.append((names[p], int(k.sum()), pearsonr(pred[k].mean(0), true[k].mean(0))[0]))
    print("\nper-perturbation Pearson of the mean response on the test cells (every perturbation with >= 5 test cells):")
    for n, c, r in rows: print(f"  {n:20s} n = {c:5d}  r = {r:.4f}")
    if rows: print(f"  mean over {len(rows)} perturbations: {np.mean([r for _, _, r in rows]):.4f}")
    print(f"\nsaved to {a.out}: pred_cells.npz, pred_means.npz, split.npz, model.pt, run_config.json")


if __name__ == "__main__":
    main()
