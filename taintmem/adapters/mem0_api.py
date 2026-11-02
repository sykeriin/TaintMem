"""Mem0-shaped facade: add / search / get / get_all / update / delete.

Differences from Mem0 (deliberate): no LLM fact-extraction on add (we store sentences verbatim, so labels stay
exact); roles map to channels (user -> user principal, assistant -> agent, tool -> untrusted tool); results carry
a `taint` field. `principals` lists the user_ids whose 'user' turns are trusted."""
from __future__ import annotations

from typing import Optional, Sequence, Union

from ..channels import ChannelRegistry
from ..store import TaintMemStore

_ROLE = {"user": "user", "assistant": "agent", "system": "system", "tool": "tool"}


class Memory:
    def __init__(self, store: TaintMemStore | None = None, principals: Optional[Sequence[str]] = None):
        self.store = store or TaintMemStore(ChannelRegistry(trusted_users=set(principals) if principals is not None else None))
        self._owner: dict[str, str] = {}

    def add(self, messages: Union[str, Sequence[dict]], user_id: str, metadata: dict | None = None,
            domain: str = "") -> dict:
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)
        out = []
        for m in msgs:
            kind = _ROLE.get(m.get("role", "tool"), "tool")
            sid = user_id if kind == "user" else m.get("name", kind)
            rec = self.store.add(m["content"], kind, sid, domain)
            self._owner[rec.id] = user_id
            out.append({"id": rec.id, "memory": rec.text, "event": "ADD"})
        return {"results": out}

    def _fmt(self, rec, score=None) -> dict:
        lab = rec.label
        d = {"id": rec.id, "memory": rec.text, "user_id": self._owner.get(rec.id),
             "taint": {"integrity": lab.integrity.name, "sources": sorted(s.group for s in lab.sources)}}
        if score is not None:
            d["score"] = score
        return d

    def search(self, query: str, user_id: str, limit: int = 5) -> dict:
        seen, out = set(), []
        for h in self.store.search(query, limit * 4):
            if h.record.id not in seen and self._owner.get(h.record.id) == user_id:
                seen.add(h.record.id); out.append(self._fmt(h.record, h.score))
        return {"results": out[:limit]}

    def get(self, memory_id: str) -> Optional[dict]:
        r = self.store.records.get(memory_id)
        return self._fmt(r) if r else None

    def get_all(self, user_id: str) -> dict:
        return {"results": [self._fmt(r) for r in self.store.records.values() if self._owner.get(r.id) == user_id]}

    def delete(self, memory_id: str) -> None:
        self._owner.pop(memory_id, None)
        self.store.delete(memory_id)
