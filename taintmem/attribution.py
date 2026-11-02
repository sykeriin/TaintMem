"""Which input spans does an output sentence derive from? Errors must only ever ADD taint."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .embed import Embedder, HashingEmbedder
from .record import Span

_TOKEN_ID = re.compile(r"\b(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{4,}\b")
_GROUPED_ID = re.compile(r"\b[A-Za-z0-9]{2,5}(?:[ -][A-Za-z0-9]{2,5}){2,}\b")


def identifiers(text: str) -> frozenset:
    """Identifier-like strings (IBANs, refs, codes), also when written in spaced groups.

    For a run of short groups we add every contiguous sub-join, so neighbouring words
    ('IBAN GB29 NWBK ...') cannot hide the identifier. Extra candidates can only add taint.
    """
    ids = {m.upper() for m in _TOKEN_ID.findall(text)}
    for m in _GROUPED_ID.findall(text):
        g = re.split(r"[ -]", m)
        for i in range(len(g)):
            for j in range(i + 3, len(g) + 1):
                compact = "".join(g[i:j]).upper()
                if len(compact) >= 8 and sum(c.isdigit() for c in compact) >= 4:
                    ids.add(compact)
    return frozenset(ids)


class Attributor:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or HashingEmbedder()
        self._cache: dict[str, np.ndarray] = {}
        self._ids: dict[str, frozenset] = {}

    def _vec(self, texts: Sequence[str]) -> np.ndarray:
        need = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if need:
            for t, v in zip(need, self.embedder.embed(need)):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def _id(self, t: str) -> frozenset:
        if t not in self._ids:
            self._ids[t] = identifiers(t)
        return self._ids[t]

    def scores(self, sentence: str, inputs: Sequence[Span]) -> np.ndarray:
        """Similarity in [0,1]. Sharing an identifier with an input counts as full attribution."""
        if not inputs:
            return np.zeros(0)
        sims = np.clip(self._vec([i.text for i in inputs]) @ self._vec([sentence])[0], 0.0, 1.0)
        mine = self._id(sentence)
        if mine:
            for j, i in enumerate(inputs):
                if mine & self._id(i.text):
                    sims[j] = 1.0
        return sims
