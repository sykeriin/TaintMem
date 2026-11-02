"""MPBench loader. UNVERIFIED SCHEMA: I could not read the released repo (the URL I tried served a different
project), so field names are guessed from the paper's description (write phase + retrieval phase, six attack
classes, benign cases). `FIELD_ALIASES` is the only place to adapt once the real files are inspected;
unknown schemas raise instead of silently mis-parsing."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

FIELD_ALIASES = {
    "id": ("id", "case_id", "uid"),
    "attack_class": ("attack_class", "attack_type", "class", "category"),
    "channel": ("channel", "write_channel", "source"),
    "write_input": ("write_input", "memory_input", "untrusted_input", "content", "payload"),
    "query": ("query", "retrieval_query", "task", "user_query"),
    "target": ("target", "target_behavior", "target_behaviour", "expected_attack"),
    "is_attack": ("is_attack", "adversarial", "malicious"),
}


@dataclass
class MPCase:
    id: str
    attack_class: str
    channel: str
    write_input: str
    query: str
    target: str
    is_attack: bool


def _pick(d: dict, name: str, default=None):
    for a in FIELD_ALIASES[name]:
        if a in d:
            return d[a]
    return default


def load_mpbench(path: str, limit: Optional[int] = None) -> list[MPCase]:
    txt = open(path).read().strip()
    rows = [json.loads(l) for l in txt.splitlines() if l.strip()] if path.endswith(".jsonl") else json.loads(txt)
    if isinstance(rows, dict):
        rows = rows.get("cases") or rows.get("data") or []
    out = []
    for i, r in enumerate(rows[:limit]):
        miss = [f for f in ("write_input", "query") if _pick(r, f) is None]
        if miss:
            raise ValueError(f"row {i}: cannot find {miss}; adapt FIELD_ALIASES after inspecting the real schema")
        out.append(MPCase(str(_pick(r, "id", i)), str(_pick(r, "attack_class", "unknown")), str(_pick(r, "channel", "tool")),
                          str(_pick(r, "write_input")), str(_pick(r, "query")), str(_pick(r, "target", "")),
                          bool(_pick(r, "is_attack", True))))
    return out


def channel_kind(channel: str) -> str:
    """Map a case's write channel to a TaintMem kind. Unknown -> 'tool' (untrusted: fail closed)."""
    c = channel.lower()
    for k in ("web", "email", "file", "user", "agent"):
        if k in c:
            return k
    return "tool"
