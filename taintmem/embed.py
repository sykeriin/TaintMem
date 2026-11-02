"""Deterministic, dependency-free embedder (hashed uni+bigrams). Swap for a real encoder in evals."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

_STOP = frozenset("a an the of to and or is are was were be been it its in on at for with as by that this i you he she we they".split())
_TOK = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _h(self, s: str) -> tuple[int, float]:
        d = hashlib.blake2b(s.encode(), digest_size=8).digest()
        n = int.from_bytes(d, "little")
        return n % self.dim, 1.0 if (n >> 63) & 1 else -1.0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = [w for w in _TOK.findall(t.lower()) if w not in _STOP]
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                j, sgn = self._h(f)
                out[i, j] += sgn
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out
