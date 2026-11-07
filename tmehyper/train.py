"""Training loop and prediction export."""
import os
import numpy as np
import torch
from .losses import TMEHyperLoss
from .model import dann_lambda


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _forward(model, b):
    """Prediction = model output + calibrated reference offset."""
    out = model(b["c_idx"], b["center_indices"], cell_emb=b.get("ctrl_emb"), gene_ids=b.get("gene_ids"), values=b.get("values"), padding_mask=b.get("padding_mask"))
    out["delta"] = out["delta"] + b["ref_off"]
    return out


def train(model, train_loader, config, hyperedges):
    """AdamW, cosine schedule, gradient clipping; the gradient-reversal weight follows the DANN ramp over the epochs."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=float(config.lr), weight_decay=float(config.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=int(config.epochs))
    loss_fn = TMEHyperLoss(config); device = config.device
    print(f"training: {config.epochs} epochs, batch {config.batch_size}, lr {config.lr}, device {device}, {len(hyperedges)} regulariser hyperedges")
    model.train()
    for epoch in range(int(config.epochs)):
        lam = dann_lambda(epoch, int(config.epochs), float(config.adv_lambda_max)); model.set_adv_lambda(lam)
        tot = {}
        for batch in train_loader:
            b = _to_device(batch, device)
            out = _forward(model, b)
            losses = loss_fn(out, b["delta_x"], b["c_idx"], hyperedges)
            optimiser.zero_grad(); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip)); optimiser.step()
            for k, v in losses.items(): tot[k] = tot.get(k, 0.0) + float(v)
        scheduler.step()
        n = max(len(train_loader), 1); avg = {k: v / n for k, v in tot.items()}
        print(f"epoch {epoch + 1:3d}/{config.epochs} | loss {avg['total']:.4f} | recon {avg['recon']:.4f} | adv {avg['adv']:.4f} (lambda {lam:.3f})"
              f" | cohesion {avg['cohesion']:.4f} separation {avg['separation']:.4f} | gate KL {avg['gate_kl']:.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, loader, config):
    """Predictions on the loader's cells in loader order."""
    model.eval(); device = config.device
    P, T, L = [], [], []
    for batch in loader:
        b = _to_device(batch, device); p = _forward(model, b)["delta"]
        P.append(p.cpu().numpy()); T.append(batch["delta_x"].numpy()); L.append(batch["c_idx"].numpy())
    return np.concatenate(P), np.concatenate(T), np.concatenate(L)


def export_predictions(out_dir, pred, true, labels, names, genes, ctrl_mean):
    """pred_cells.npz: per-cell predicted and observed responses of the test cells (the input of scripts/evaluate.py);
    pred_means.npz: per-perturbation means over the test cells."""
    np.savez(os.path.join(out_dir, "pred_cells.npz"), pred=pred.astype("float32"), true=true.astype("float32"),
             labels=labels.astype("int32"), names=np.array([str(n) for n in names]), genes=np.array([str(g) for g in genes]), ctrl_mean=np.asarray(ctrl_mean, "float32"))
    pm, tm, nn_, nm = [], [], [], []
    for p in np.unique(labels):
        k = labels == p
        if k.sum() >= 5: pm.append(pred[k].mean(0)); tm.append(true[k].mean(0)); nn_.append(int(k.sum())); nm.append(str(names[p]))
    if pm: np.savez(os.path.join(out_dir, "pred_means.npz"), names=np.array(nm), genes=np.array([str(g) for g in genes]), pred=np.stack(pm), true=np.stack(tm), n=np.array(nn_))
