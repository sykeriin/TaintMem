"""Which input spans does an output sentence derive from? Errors must only ever ADD taint."""
from __future__ import annotations

import math
import re
from collections import Counter
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


_WORD = re.compile(r"[a-z0-9]+")
_STOPW = frozenset("a an the of to and or is are was were be been it its in on at for with as by that this i you he she we they".split())


def _features(text: str) -> Counter:
    toks = [w for w in _WORD.findall(text.lower()) if w not in _STOPW]
    return Counter(toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])])


class Attributor:
    """idf=True weights features by inverse frequency ACROSS THE CURRENT INPUTS, so voice boilerplate
    shared by most inputs ('the user's ...') stops creating spurious attribution (label creep)."""

    def __init__(self, embedder: Embedder | None = None, idf: bool = False, min_docs: int = 4):
        self.idf = idf
        self.min_docs = min_docs               # fewer documents than this: IDF is unstable -> plain cosine
        self.background: list[str] = []        # extra corpus (e.g. the whole store) for document frequencies
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

    def _idf_sims(self, sentence: str, inputs: Sequence[Span]) -> np.ndarray:
        docs = [_features(i.text) for i in inputs]
        q = _features(sentence)
        seen = {i.text for i in inputs} | {sentence}
        extra = [_features(t) for t in dict.fromkeys(self.background) if t not in seen]
        n = len(docs) + 1 + len(extra)
        df = Counter()
        for d in docs + [q] + extra:
            df.update(d.keys())
        w = lambda c: {f: (1 + math.log(v)) * math.log((n + 1) / (df[f] + 0.5)) for f, v in c.items()}
        wq = w(q)
        nq = math.sqrt(sum(v * v for v in wq.values())) or 1.0
        out = np.zeros(len(docs))
        for j, d in enumerate(docs):
            wd = w(d)
            nd = math.sqrt(sum(v * v for v in wd.values())) or 1.0
            out[j] = sum(v * wd.get(f, 0.0) for f, v in wq.items()) / (nq * nd)
        return out

    def anchored(self, sentence: str, inputs: Sequence[Span]) -> np.ndarray:
        """True where the sentence shares an identifier with the input (exact anchor)."""
        mine = self._id(sentence)
        return np.array([bool(mine & self._id(i.text)) for i in inputs], dtype=bool)

    def scores(self, sentence: str, inputs: Sequence[Span]) -> np.ndarray:
        """Similarity in [0,1]. Sharing an identifier with an input counts as full attribution."""
        if not inputs:
            return np.zeros(0)
        n_docs = len({i.text for i in inputs} | set(self.background)) + 1
        if self.idf and n_docs >= self.min_docs:
            sims = np.clip(self._idf_sims(sentence, inputs), 0.0, 1.0)
        else:
            sims = np.clip(self._vec([i.text for i in inputs]) @ self._vec([sentence])[0], 0.0, 1.0)
        sims[self.anchored(sentence, inputs)] = 1.0
        return sims
