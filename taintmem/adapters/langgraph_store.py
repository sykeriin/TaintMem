"""LangGraph BaseStore-shaped adapter (duck-typed: put/get/search/delete + async twins).

The caller supplies the CHANNEL of each write (`channel=(kind, id, domain)`), never the value: a dict that
came from a tool or web page must not be able to claim its own provenance. No channel -> `default_channel`,
which defaults to untrusted tool output (fail closed). Values must carry text under "text"."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..store import TaintMemStore


@dataclass
class Item:
    namespace: tuple
    key: str
    value: dict
    score: Optional[float] = None


class TaintMemLangGraphStore:
    def __init__(self, store: TaintMemStore | None = None, default_channel=("tool", "unknown", "")):
        self.store, self.default_channel = store or TaintMemStore(), default_channel
        self._index: dict[tuple, str] = {}          # (namespace, key) -> record id

    def put(self, namespace: tuple, key: str, value: dict, index=None, *, channel=None, ttl=None) -> None:
        if value is None:                            # LangGraph convention: put(None) deletes
            return self.delete(namespace, key)
        if "text" not in value:
            raise ValueError("value must contain 'text'")
        if (namespace, key) in self._index:
            self.delete(namespace, key)
        kind, sid, dom = channel or self.default_channel
        rec = self.store.add(str(value["text"]), kind, sid, dom)
        self._index[(tuple(namespace), key)] = rec.id

    def _item(self, namespace, key, rid, score=None) -> Optional[Item]:
        rec = self.store.records.get(rid)
        if rec is None:
            return None
        lab = rec.label
        return Item(tuple(namespace), key, {"text": rec.text, "taint": {"integrity": lab.integrity.name,
                                                                          "sources": sorted(s.group for s in lab.sources)}}, score)

    def get(self, namespace: tuple, key: str) -> Optional[Item]:
        rid = self._index.get((tuple(namespace), key))
        return self._item(namespace, key, rid) if rid else None

    def delete(self, namespace: tuple, key: str) -> None:
        rid = self._index.pop((tuple(namespace), key), None)
        if rid:
            self.store.delete(rid)

    def search(self, namespace_prefix: tuple, /, *, query: str | None = None, limit: int = 10, offset: int = 0) -> list[Item]:
        rev = {rid: k for k, rid in self._index.items()}
        pfx = tuple(namespace_prefix)
        if query:
            hits = [(h.record.id, h.score) for h in self.store.search(query, limit + offset + len(self._index))]
            seen, ordered = set(), []
            for rid, sc in hits:
                if rid not in seen:
                    seen.add(rid); ordered.append((rid, sc))
        else:
            ordered = [(rid, None) for rid in rev]
        items = []
        for rid, sc in ordered:
            ns, key = rev.get(rid, (None, None))
            if ns is not None and ns[:len(pfx)] == pfx:
                it = self._item(ns, key, rid, sc)
                if it:
                    items.append(it)
        return items[offset:offset + limit]

    def list_namespaces(self) -> list[tuple]:
        return sorted({ns for ns, _ in self._index})

    async def aput(self, *a, **k): return self.put(*a, **k)
    async def aget(self, *a, **k): return self.get(*a, **k)
    async def asearch(self, *a, **k): return self.search(*a, **k)
    async def adelete(self, *a, **k): return self.delete(*a, **k)
