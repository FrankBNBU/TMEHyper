"""Conditional decoder: 256-d bottleneck over [cell code, environment, perturbation], re-injection of the environment and
perturbation codes, and residual blocks whose normalised input is scaled and shifted by FiLM parameters generated from the
environment representation."""
import torch
import torch.nn as nn


class FiLMResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, gamma, beta):
        h = gamma * self.norm1(x) + beta
        return self.norm2(x + self.ff(h))


class ConditionalDecoder(nn.Module):
    def __init__(self, n_genes, config):
        super().__init__()
        e = config.env_embed_dim; p = config.perturb_embed_dim; b = config.bottleneck_dim; h = config.decoder_hidden_dim
        self.bottleneck_encoder = nn.Sequential(nn.Linear(config.latent_dim + e + p, b * 2), nn.LayerNorm(b * 2), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(b * 2, b), nn.LayerNorm(b))
        self.film_generator = nn.Sequential(nn.Linear(e, h * 2), nn.GELU(), nn.Linear(h * 2, h * 2))
        self.input_proj = nn.Sequential(nn.Linear(b + e + p, h), nn.LayerNorm(h), nn.GELU())
        self.decoder_blocks = nn.ModuleList([FiLMResidualBlock(h, config.dropout) for _ in range(config.n_decoder_layers)])
        self.output_proj = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, n_genes))

    def forward(self, z_cell, z_env, z_perturb):
        z_latent = self.bottleneck_encoder(torch.cat([z_cell, z_env, z_perturb], dim=-1))
        gamma, beta = self.film_generator(z_env).chunk(2, dim=-1)
        h = self.input_proj(torch.cat([z_latent, z_env, z_perturb], dim=-1))
        for block in self.decoder_blocks:
            h = block(h, gamma, beta)
        return self.output_proj(h)
