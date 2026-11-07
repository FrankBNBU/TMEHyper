"""Training objective.

    L = MSE(Delta_hat, Delta)  +  xi L_adv  +  Omega
    Omega = w_coh L_cohesion + w_sep L_separation (hyperedge regularisers on the environment embedding)
          + w_l1 mean(gate) (sparsity of the gene gate) + w_kl KL(alpha || uniform) (response gate close to uniform)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .regulariser import HyperedgeRegulariser


class TMEHyperLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.regulariser = HyperedgeRegulariser(config)
        self.adv_weight = float(config.adv_weight); self.gate_l1 = float(config.gate_l1); self.gate_kl = float(config.gate_kl)

    def forward(self, out, target, perturb_labels, hyperedges):
        losses = {"recon": F.mse_loss(out["delta"], target), "adv": F.cross_entropy(out["perturb_logits"], perturb_labels)}
        reg = self.regulariser(out["z_nodes"], hyperedges); losses["cohesion"], losses["separation"] = reg["cohesion"], reg["separation"]
        a = out["gate_alpha"]; losses["gate_l1"] = out["gate"].mean(); losses["gate_kl"] = (a * torch.log(a * a.shape[1] + 1e-8)).sum(1).mean()
        losses["total"] = losses["recon"] + reg["total"] + self.adv_weight * losses["adv"] + self.gate_l1 * losses["gate_l1"] + self.gate_kl * losses["gate_kl"]
        return losses
