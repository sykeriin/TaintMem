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


class SentenceTransformerEmbedder:
    """Optional real encoder (`pip install sentence-transformers`); runs fine on CPU or a small GPU.
    Import is lazy so the core package keeps numpy as its only dependency."""
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("pip install sentence-transformers to use SentenceTransformerEmbedder") from e
        self.model = SentenceTransformer(model, device=device)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)


class NLIVerifier:
    """Optional entailment check (`pip install transformers torch`). entails(premise, hypothesis) -> P(entail)."""
    def __init__(self, model: str = "cross-encoder/nli-deberta-v3-small"):
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError("pip install transformers torch to use NLIVerifier") from e
        self.pipe = pipeline("text-classification", model=model, top_k=None)

    def entails(self, premise: str, hypothesis: str) -> float:
        out = self.pipe({"text": premise, "text_pair": hypothesis})
        out = out[0] if out and isinstance(out[0], list) else out
        return next((d["score"] for d in out if d["label"].lower().startswith("entail")), 0.0)
