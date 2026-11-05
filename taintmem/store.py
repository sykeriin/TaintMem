"""TaintMemStore: labelled memory with similarity search. SQLite persistence is optional."""
from __future__ import annotations

import json
import math
import contextlib
import sqlite3
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence

import numpy as np

from .channels import ChannelRegistry
from .embed import _STOP, _TOK, Embedder, HashingEmbedder
from .labels import Confidentiality, Integrity, Label
from .record import MemoryRecord, Span
from .segment import split_sentences

UNTRUSTED_TTL = 30 * 24 * 3600.0


class TamperError(Exception):
    pass


@dataclass
class Retrieved:
    span: Span
    record: MemoryRecord
    score: float


class TaintMemStore:
    def __init__(self, registry: ChannelRegistry | None = None, embedder: Embedder | None = None,
                 path: str | None = None, secret: bytes = b"dev-secret", clock: Callable[[], float] = time.time,
                 untrusted_ttl: Optional[float] = UNTRUSTED_TTL, retriever: str = "dense"):
        assert retriever in ("dense", "bm25", "hybrid")
        self.retriever = retriever            # dense = embedder cosine | bm25 | hybrid = reciprocal-rank fusion
        self.registry = registry or ChannelRegistry()
        self.embedder = embedder or HashingEmbedder()
        self.secret = secret
        self.clock = clock
        self.untrusted_ttl = untrusted_ttl
        self.records: dict[str, MemoryRecord] = {}
        self.audit_log: list[dict] = []
        self._vec: dict[str, np.ndarray] = {}
        self._toks: dict[str, list] = {}
        self._n = 0
        self.path = path
        if path:
            self._load()

    # ---- writes ---------------------------------------------------------------------
    def _new_id(self) -> str:
        self._n += 1
        return f"r{self._n}"

    def add(self, text: str, kind: str, source_id: str = "", domain: str = "", agent_id: str = "agent0",
            confidentiality: Confidentiality = Confidentiality.INTERNAL) -> MemoryRecord:
        label = self.registry.label_for(kind, source_id, domain, agent_id, confidentiality)
        rid = self._new_id()
        spans = [Span(f"{rid}.s{i}", t, label, rid, i) for i, t in enumerate(split_sentences(text))]
        return self._commit(MemoryRecord(rid, spans, (), "write", self.clock()))

    def add_derived(self, items: Sequence[tuple[str, Label]], parent_ids: Sequence[str], op: str,
                    edges: dict | None = None) -> MemoryRecord:
        rid = self._new_id()
        spans = [Span(f"{rid}.s{i}", t, l, rid, i) for i, (t, l) in enumerate(items)]
        return self._commit(MemoryRecord(rid, spans, tuple(parent_ids), op, self.clock(), edges=edges or {}))

    def _commit(self, rec: MemoryRecord) -> MemoryRecord:
        if self.untrusted_ttl is not None and rec.label.integrity == Integrity.UNTRUSTED:
            rec.expires_at = rec.created_at + self.untrusted_ttl
        self.records[rec.id] = rec
        self._seal(rec)
        return rec

    def _seal(self, rec: MemoryRecord) -> None:
        rec.mac = rec.digest(self.secret)
        self._persist(rec)

    def replace_span(self, span_id: str, new_label: Label) -> Span:
        rec = self.records[span_id.split(".")[0]]
        for i, s in enumerate(rec.spans):
            if s.id == span_id:
                rec.spans[i] = replace(s, label=new_label)
                self._seal(rec)
                return rec.spans[i]
        raise KeyError(span_id)

    def verify(self, rid: str) -> bool:
        rec = self.records[rid]
        return rec.mac == rec.digest(self.secret)

    # ---- reads ----------------------------------------------------------------------
    def spans(self, include_quarantined: bool = False) -> list[Span]:
        return [s for r in self.records.values() if include_quarantined or not r.quarantined for s in r.spans]

    def span(self, span_id: str) -> Span:
        return next(s for s in self.records[span_id.split(".")[0]].spans if s.id == span_id)

    def _embed(self, spans: Sequence[Span]) -> np.ndarray:
        need = [s for s in spans if s.id not in self._vec]
        if need:
            for s, v in zip(need, self.embedder.embed([s.text for s in need])):
                self._vec[s.id] = v
        return np.stack([self._vec[s.id] for s in spans]) if spans else np.zeros((0, 1))

    def _bm25(self, spans: Sequence[Span], query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        for sp in spans:
            if sp.id not in self._toks:
                self._toks[sp.id] = [w for w in _TOK.findall(sp.text.lower()) if w not in _STOP]
        docs = [self._toks[sp.id] for sp in spans]
        n, avg = len(docs), (sum(map(len, docs)) / max(len(docs), 1)) or 1.0
        df: dict[str, int] = {}
        for d in docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        q = [w for w in dict.fromkeys(_TOK.findall(query.lower())) if w not in _STOP]
        out = np.zeros(n)
        for i, d in enumerate(docs):
            tf: dict[str, int] = {}
            for w in d:
                tf[w] = tf.get(w, 0) + 1
            for w in q:
                if w in tf:
                    idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                    out[i] += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        return out

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        spans = self.spans()
        if not spans:
            return []
        dense = self._embed(spans) @ self.embedder.embed([query])[0] if self.retriever != "bm25" else None
        sparse = self._bm25(spans, query) if self.retriever != "dense" else None
        if self.retriever == "dense":
            scores = dense
        elif self.retriever == "bm25":
            scores = sparse
        else:                                                  # reciprocal rank fusion
            scores = np.zeros(len(spans))
            for sc in (dense, sparse):
                for rank, i in enumerate(np.argsort(-sc)):
                    scores[i] += 1.0 / (60 + rank)
        order = np.argsort(-scores)[:k]
        return [Retrieved(spans[i], self.records[spans[i].record_id], float(scores[i])) for i in order]

    @staticmethod
    def render(items: Sequence[Retrieved]) -> str:
        """Context text with VISIBLE provenance tags."""
        return "\n".join(f"[{r.span.label.tag()}] {r.span.text}" for r in items)

    # ---- quarantine / delete --------------------------------------------------------
    def quarantine(self, rid: str, on: bool = True) -> None:
        self.records[rid].quarantined = on
        self._seal(self.records[rid])

    def delete(self, rid: str) -> None:
        self.records.pop(rid, None)
        for sid in [k for k in self._vec if k.startswith(rid + ".")]:
            del self._vec[sid]
        for sid in [k for k in self._toks if k.startswith(rid + ".")]:
            del self._toks[sid]
        if self.path:
            with self._db() as c:
                c.execute("DELETE FROM records WHERE id=?", (rid,))

    # ---- persistence ----------------------------------------------------------------
    @contextlib.contextmanager
    def _db(self):
        """Commit on success, roll back on error, and always close (sqlite3's own context manager never closes,
        which leaks the file handle and breaks temp-dir cleanup on Windows)."""
        c = sqlite3.connect(self.path)
        try:
            with c:
                yield c
        finally:
            c.close()

    def _persist(self, rec: MemoryRecord) -> None:
        if not self.path:
            return
        with self._db() as c:
            c.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, body TEXT, mac TEXT)")
            c.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)",
                      (rec.id, json.dumps(rec.to_dict(), sort_keys=True), rec.mac))

    def _load(self) -> None:
        with self._db() as c:
            c.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, body TEXT, mac TEXT)")
            rows = c.execute("SELECT body, mac FROM records").fetchall()
        for body, mac in rows:
            rec = MemoryRecord.from_dict(json.loads(body), mac)
            if rec.mac != rec.digest(self.secret):
                raise TamperError(f"record {rec.id} failed integrity check")
            self.records[rec.id] = rec
            self._n = max(self._n, int(rec.id[1:]))
