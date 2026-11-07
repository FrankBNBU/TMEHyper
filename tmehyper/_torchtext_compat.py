"""
Minimal torchtext.vocab compatibility shim for scGPT.

torchtext was deprecated after PyTorch 2.3 and is not available for PyTorch 2.5+.
scGPT's GeneVocab only uses torchtext.vocab.Vocab as a base class with basic
token<->index mapping. This module provides that minimal interface.

Usage (auto-injected by encoders.py before scGPT import):
    import sys, types
    from tmehyper import _torchtext_compat
    torchtext_mod = types.ModuleType('torchtext')
    torchtext_mod.vocab = _torchtext_compat
    sys.modules['torchtext'] = torchtext_mod
    sys.modules['torchtext.vocab'] = _torchtext_compat
"""

from collections import OrderedDict
from typing import Dict, Optional


class Vocab:
    """Minimal replacement for torchtext.vocab.Vocab."""

    def __init__(self, vocab_obj=None):
        """
        Args:
            vocab_obj: one of:
                - None -> empty vocab
                - dict (token->index) -> direct init
                - another Vocab -> copy
        """
        if vocab_obj is None:
            self._stoi: Dict[str, int] = {}
            self._itos: Dict[int, str] = {}
        elif isinstance(vocab_obj, Vocab):
            self._stoi = dict(vocab_obj._stoi)
            self._itos = dict(vocab_obj._itos)
        elif isinstance(vocab_obj, dict):
            self._stoi = dict(vocab_obj)
            self._itos = {v: k for k, v in vocab_obj.items()}
        else:
            raise TypeError(f"Unsupported vocab_obj type: {type(vocab_obj)}")
        self._default_index: Optional[int] = None

    # ------ Property that scGPT GeneVocab accesses: super().__init__(_vocab.vocab) ------
    @property
    def vocab(self):
        return self

    # ------ Token lookup ------
    def __getitem__(self, token: str) -> int:
        if token in self._stoi:
            return self._stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(f"Token '{token}' not found in vocabulary")

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __len__(self) -> int:
        return len(self._stoi)

    # ------ Mutation ------
    def set_default_index(self, idx: int):
        self._default_index = idx

    def insert_token(self, token: str, idx: int):
        self._stoi[token] = idx
        self._itos[idx] = token

    def append_token(self, token: str):
        idx = max(self._itos.keys()) + 1 if self._itos else 0
        self.insert_token(token, idx)

    # ------ Serialization ------
    def get_stoi(self) -> Dict[str, int]:
        return dict(self._stoi)

    def get_itos(self):
        if not self._itos:
            return []
        max_idx = max(self._itos.keys())
        return [self._itos.get(i, "") for i in range(max_idx + 1)]

    def __repr__(self):
        return f"Vocab(size={len(self)})"


def vocab(ordered_dict: OrderedDict, min_freq: int = 1) -> Vocab:
    """Factory function mimicking torchtext.vocab.vocab()."""
    filtered = {k: v for k, v in ordered_dict.items() if v >= min_freq}
    token2idx = {token: i for i, token in enumerate(filtered.keys())}
    return Vocab(token2idx)
