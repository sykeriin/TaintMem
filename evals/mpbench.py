"""MPBench case loader.

Schema is taken from the paper's Appendix D.2 (arXiv 2606.04329): each case is a JSON object with
attack_type, attack_signal (strong|moderate|weak), domain (1 of 7), adversarial_goal (1 of 6), user_query (makes
the agent process the payload), context (external content holding the payload), expected_memory (target entry,
used for the ASR judgment) and retrieval_query (follow-up, used for the RSR judgment).

CAVEAT: as of 2026-11-03 I found NO released dataset: the paper gives no code/data link (the repo
'Jacobhhy/Agent-Memory-Poisoning' that a search summary suggested is MemoryGraft, a different paper). This
loader therefore has only been tested on a synthetic fixture in that schema. File names and the benign-case
encoding are unknown, so `is_attack` is inferred. ASR/RSR in the paper are LLM-judged; no judge is implemented here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

ATTACK_CLASSES = ("explicit", "conditional", "salience-driven", "policy-conformant", "false-precedent", "skill-procedure")
REQUIRED = ("user_query", "context", "retrieval_query")


@dataclass
class MPCase:
    id: str
    attack_type: Optional[str]
    attack_signal: Optional[str]
    domain: str
    adversarial_goal: Optional[str]
    user_query: str
    context: str
    expected_memory: str
    retrieval_query: str
    is_attack: bool


def load_mpbench(path: str, limit: Optional[int] = None) -> list[MPCase]:
    with open(path) as f:
        txt = f.read().strip()
    rows = [json.loads(l) for l in txt.splitlines() if l.strip()] if path.endswith(".jsonl") else json.loads(txt)
    if isinstance(rows, dict):
        rows = rows.get("cases") or rows.get("data") or []
    out = []
    for i, r in enumerate(rows[:limit]):
        miss = [f for f in REQUIRED if f not in r]
        if miss:
            raise ValueError(f"row {i}: missing {miss}; expected MPBench Appendix D.2 fields")
        at = r.get("attack_type")
        attack = bool(at) and str(at).lower() not in ("benign", "none") and bool(r.get("expected_memory"))
        out.append(MPCase(str(r.get("id", i)), at, r.get("attack_signal"), str(r.get("domain", "unknown")),
                          r.get("adversarial_goal"), r["user_query"], r["context"], str(r.get("expected_memory") or ""),
                          r["retrieval_query"], attack))
    return out


def write_kind(channel: str = "tool") -> str:
    """The case does not say which channel `context` arrived on; default to untrusted tool output (fail closed)."""
    return channel if channel in ("web", "email", "file", "tool", "agent") else "tool"
