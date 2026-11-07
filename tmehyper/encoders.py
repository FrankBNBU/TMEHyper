"""Cell encoders (frozen foundation model -> cell code) and the gated fusion of a small representation with the cell code."""
import os
import sys
import types
import torch
import torch.nn as nn

# scGPT needs torchtext.vocab; provide a minimal shim when torchtext is not installed
try:
    import torchtext  # noqa: F401
except Exception:
    from . import _torchtext_compat
    _tt = types.ModuleType("torchtext"); _tt.vocab = _torchtext_compat
    sys.modules["torchtext"] = _tt; sys.modules["torchtext.vocab"] = _torchtext_compat

try:
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    SCGPT_AVAILABLE = True
except ImportError:
    SCGPT_AVAILABLE = False; TransformerModel = None; GeneVocab = None


def load_scgpt(model_dir, config):
    """Load a pretrained scGPT checkpoint (vocab.json, args.json, best_model.pt).  Returns (model, vocab)."""
    import json
    if not SCGPT_AVAILABLE: raise ImportError("scGPT is not installed (pip install scgpt)")
    from scgpt.utils import load_pretrained
    vocab = GeneVocab.from_file(os.path.join(model_dir, "vocab.json"))
    for token in ["<pad>", "<cls>", "<eoc>"]:
        if token not in vocab: vocab.append_token(token)
    mc = json.load(open(os.path.join(model_dir, "args.json")))
    model = TransformerModel(ntoken=len(vocab), d_model=mc["embsize"], nhead=mc["nheads"], d_hid=mc["d_hid"], nlayers=mc["nlayers"], vocab=vocab,
                             dropout=mc["dropout"], pad_token=mc["pad_token"], pad_value=mc["pad_value"], do_mvc=False, do_dab=False,
                             use_batch_labels=False, input_emb_style="continuous", cell_emb_style="cls")
    load_pretrained(model, torch.load(os.path.join(model_dir, "best_model.pt"), map_location="cpu", weights_only=True), verbose=False)
    config.scgpt_embed_dim = mc["embsize"]; config.scgpt_pad_value = mc.get("pad_value", config.scgpt_pad_value); config.scgpt_n_bins = mc.get("n_bins", config.scgpt_n_bins)
    print(f"scGPT loaded from {model_dir} ({mc['nlayers']} layers, {len(vocab)} tokens)")
    return model, vocab


class ScGPTCellEncoder(nn.Module):
    """Frozen scGPT: tokenised expression -> CLS token -> projection to the cell code."""

    def __init__(self, config, scgpt_model):
        super().__init__()
        self.scgpt = scgpt_model; self.d_model = config.scgpt_embed_dim
        self.proj = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, config.latent_dim), nn.GELU(), nn.LayerNorm(config.latent_dim))

    def forward(self, gene_ids, values, src_key_padding_mask):
        return self.proj(self.scgpt._encode(gene_ids, values, src_key_padding_mask)[:, 0, :])


class PrecomputedCellEncoder(nn.Module):
    """Frozen foundation-model embedding (e.g. Nicheformer) -> three-layer perceptron -> cell code."""

    def __init__(self, emb_dim, config):
        super().__init__()
        h = config.cell_hidden_dim
        self.emb_norm = nn.LayerNorm(emb_dim)
        self.net = nn.Sequential(nn.Linear(emb_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(config.dropout),
                                 nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, config.latent_dim))

    def forward(self, emb):
        return self.net(self.emb_norm(emb))


class GatedFusion(nn.Module):
    """A small representation (32-d) reads the cell code (512-d) through single-key multi-head attention with a sigmoid gate:
    out = W_o LN(y + g * v),  v = attention value derived from the cell code,  g = sigmoid(W_g [y, v])."""

    def __init__(self, cell_dim, small_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.cell_to_kv = nn.Sequential(nn.Linear(cell_dim, cell_dim), nn.GELU(), nn.Linear(cell_dim, small_dim * 2))
        self.query_proj = nn.Linear(small_dim, small_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=small_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(small_dim * 2, small_dim), nn.Sigmoid())
        self.output_proj = nn.Sequential(nn.LayerNorm(small_dim), nn.Linear(small_dim, small_dim))

    def forward(self, z_cell, y):
        k, v = self.cell_to_kv(z_cell).chunk(2, dim=-1)
        attn, _ = self.cross_attn(self.query_proj(y).unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)); attn = attn.squeeze(1)
        return self.output_proj(y + self.gate(torch.cat([y, attn], dim=-1)) * attn)
