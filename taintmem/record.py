from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Optional

from .labels import Confidentiality, Integrity, Label, Source, meet


@dataclass(frozen=True)
class Span:
    id: str
    text: str
    label: Label
    record_id: str
    index: int = 0


@dataclass
class MemoryRecord:
    id: str
    spans: list
    parent_ids: tuple = ()
    op: str = "write"                      # write | summarise | merge | reflect | dedupe
    created_at: float = 0.0
    expires_at: Optional[float] = None
    quarantined: bool = False
    edges: dict = field(default_factory=dict)   # span id -> input span ids it was attributed to
    mac: str = ""

    @property
    def label(self) -> Label:
        return meet(s.label for s in self.spans)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.spans)

    # --- serialisation / integrity -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id, "parent_ids": list(self.parent_ids), "op": self.op,
            "created_at": self.created_at, "expires_at": self.expires_at,
            "quarantined": self.quarantined, "edges": self.edges,
            "spans": [{"id": s.id, "text": s.text, "index": s.index, "label": _label_to_dict(s.label)}
                      for s in self.spans],
        }

    @staticmethod
    def from_dict(d: dict, mac: str = "") -> "MemoryRecord":
        spans = [Span(s["id"], s["text"], _label_from_dict(s["label"]), d["id"], s["index"]) for s in d["spans"]]
        return MemoryRecord(d["id"], spans, tuple(d["parent_ids"]), d["op"], d["created_at"],
                            d["expires_at"], d["quarantined"], d["edges"], mac)

    def digest(self, key: bytes) -> str:
        body = self.to_dict()
        return hmac.new(key, json.dumps(body, sort_keys=True).encode(), hashlib.sha256).hexdigest()


def _label_to_dict(l: Label) -> dict:
    return {"integrity": int(l.integrity), "confidentiality": int(l.confidentiality),
            "sources": sorted([s.kind, s.id, s.domain] for s in l.sources),
            "agent_id": l.agent_id, "endorsed_by": l.endorsed_by}


def _label_from_dict(d: dict) -> Label:
    return Label(Integrity(d["integrity"]), Confidentiality(d["confidentiality"]),
                 frozenset(Source(*s) for s in d["sources"]), d["agent_id"], d["endorsed_by"])
