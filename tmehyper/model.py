"""TMEHyper.

    cell code          z_c = MLP(E(x_c))                      frozen foundation-model embedding of the locally paired control cell
    environment        z_e = Fuse(z_c, GroupHypergraph(F)_i)  low / high frequency components of the cell under the group hyperedges
    perturbation       z_p = Fuse(z_c, LN(Emb[p]))
    decoder            psi(z_c, z_p, z_e)                     bottleneck + FiLM residual blocks
    response head      delta = W [ (U z_p) * (V ctx) ],  ctx = sum_k alpha_k(p) C_k   (gate over the hypergraph components)
    gene gate          chi = sigmoid( q(z_p, z_c) . k_gene / sqrt(d) / H + b )
    prediction         Delta_hat = chi * (psi + delta)  [+ calibrated reference offset r, added by the training loop]
    adversary          a perturbation classifier on z_e through a gradient-reversal layer keeps z_e perturbation-invariant
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from .decoder import ConditionalDecoder
from .encoders import PrecomputedCellEncoder, ScGPTCellEncoder, GatedFusion
from .hyperedges import GroupHypergraphEncoder


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return _GradientReversal.apply(x, self.lambda_)


def dann_lambda(epoch, total_epochs, lambda_max=1.0):
    """Sigmoid ramp of the gradient-reversal weight (Ganin et al.): lambda(p) = lambda_max (2 / (1 + exp(-10 p)) - 1)."""
    p = epoch / max(total_epochs, 1)
    return float(lambda_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


class TMEHyper(nn.Module):
    def __init__(self, n_genes, n_perturbations, config, cell_embedding_dim=0, scgpt_model=None, env_input_dim=None):
        super().__init__()
        # ---- cell encoder (frozen foundation model -> cell code) ----
        self.use_precomputed = int(cell_embedding_dim or 0) > 0
        if self.use_precomputed:
            self.cell_encoder = PrecomputedCellEncoder(int(cell_embedding_dim), config)
            print(f"cell encoder: precomputed embedding ({cell_embedding_dim}-d) -> MLP")
        elif scgpt_model is not None:
            self.cell_encoder = ScGPTCellEncoder(config, scgpt_model)
            for p in self.cell_encoder.scgpt.parameters(): p.requires_grad = False
            print("cell encoder: frozen scGPT -> projection")
        else:
            raise ValueError("give either a cell embedding file (--cell_embedding_file) or an scGPT checkpoint (--scgpt_dir)")
        # ---- environment encoder ----
        self.env_encoder = GroupHypergraphEncoder(config, int(env_input_dim))
        # ---- perturbation code ----
        self.perturb_encoder = nn.Sequential(nn.Embedding(n_perturbations, config.perturb_embed_dim), nn.LayerNorm(config.perturb_embed_dim))
        # ---- gated fusion with the cell code ----
        self.env_fusion = GatedFusion(config.latent_dim, config.env_embed_dim, n_heads=4, dropout=config.dropout)
        self.perturb_fusion = GatedFusion(config.latent_dim, config.perturb_embed_dim, n_heads=4, dropout=config.dropout)
        # ---- decoder ----
        self.decoder = ConditionalDecoder(n_genes, config)
        # ---- response head: perturbation x environment-component interaction, zero at the start ----
        r = int(config.group_rank); nc = int(self.env_encoder.n_comps); hd = int(self.env_encoder.hidden)
        self.gh_U = nn.Linear(config.perturb_embed_dim, r)
        self.gh_V = nn.Sequential(nn.LayerNorm(hd), nn.Linear(hd, r))
        self.group_out = nn.Linear(r, n_genes, bias=False); nn.init.zeros_(self.group_out.weight)
        self.gh_gate = nn.Linear(config.perturb_embed_dim, nc); nn.init.zeros_(self.gh_gate.weight); nn.init.zeros_(self.gh_gate.bias)   # uniform gate at the start
        print(f"response head: rank {r}, {nc} components, per-perturbation softmax gate")
        # ---- gene gate: which genes a perturbation reaches in this cell ----
        d = int(config.gene_attn_dim); H = int(config.gene_attn_heads); self.ga_d, self.ga_h = d, H
        self.ga_gene = nn.Parameter(torch.randn(n_genes, d) * 0.02)
        self.ga_q = nn.Linear(config.perturb_embed_dim + config.latent_dim, d * H)
        self.ga_k = nn.Linear(d, d * H, bias=False)
        self.ga_b = nn.Parameter(torch.full((n_genes,), 2.0))                 # about 88 % open at the start
        nn.init.zeros_(self.ga_q.weight); nn.init.zeros_(self.ga_q.bias)
        print(f"gene gate: [z_p; z_c] -> {H} heads x {d}-d over {n_genes} gene keys")
        # ---- adversary ----
        self.grl = GradientReversal(lambda_=0.0)
        self.perturb_discriminator = nn.Sequential(nn.Linear(config.env_embed_dim, 64), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(64, n_perturbations))

    def set_adv_lambda(self, lambda_):
        self.grl.set_lambda(lambda_)

    def forward(self, perturb_idx, center_indices, cell_emb=None, gene_ids=None, values=None, padding_mask=None):
        """perturb_idx [B]; center_indices [B] indices of the cells among the environment-encoder nodes; cell_emb [B, D] precomputed
        embedding of the paired control cell, or (gene_ids, values, padding_mask) its scGPT tokens."""
        if self.use_precomputed:
            z_cell = self.cell_encoder(cell_emb)
        else:
            z_cell = self.cell_encoder(gene_ids, values, padding_mask)
        z_env_raw, z_nodes = self.env_encoder(center_indices, return_all=self.training)
        z_perturb_raw = self.perturb_encoder(perturb_idx)
        z_env = self.env_fusion(z_cell, z_env_raw)
        z_perturb = self.perturb_fusion(z_cell, z_perturb_raw)
        perturb_logits = self.perturb_discriminator(self.grl(z_env))
        delta = self.decoder(z_cell, z_env, z_perturb)
        C = self.env_encoder.last_comps                                                   # [B, nc, 64]
        gate_alpha = torch.softmax(self.gh_gate(z_perturb_raw), dim=-1)
        ctx = (gate_alpha[..., None] * C).sum(1)
        delta_group = self.group_out(self.gh_U(z_perturb_raw) * self.gh_V(ctx))
        delta = delta + delta_group
        B = z_cell.shape[0]; d, H = self.ga_d, self.ga_h
        q = self.ga_q(torch.cat([z_perturb_raw, z_cell], dim=-1)).view(B, H, d)
        k = self.ga_k(self.ga_gene).view(-1, H, d)
        logit = torch.einsum("bhd,ghd->bg", q, k) / (d ** 0.5) / H + self.ga_b
        gate = torch.sigmoid(logit)
        return {"delta": gate * delta, "gate": gate, "gate_alpha": gate_alpha, "perturb_logits": perturb_logits, "z_nodes": z_nodes}


def count_parameters(model):
    return {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
