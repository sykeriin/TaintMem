"""LLM-free utility check on real LongMemEval data: does the store retrieve the evidence turn(s)?

For each question we ingest every haystack turn as its own record (user -> trusted, assistant -> SEMI) and
ask whether ANY top-k retrieved span comes from a turn with has_answer=True. This is an upper bound on
answer quality (retrieval can only hurt), and it shows what dropping non-trusted memory costs.
Note: the *oracle* file contains only evidence sessions, so absolute recall is optimistic; the cleaned 's'
file (full haystack) is much harder and is not used here.

python -m evals.retrieval_recall PATH/longmemeval_oracle.json [--k 8] [--limit 500]
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

from taintmem import ChannelRegistry, Integrity, TaintMemStore

from .longmemeval import load_longmemeval


def recall_at_k(item: dict, k: int, keep: set[str], retriever: str = "dense") -> tuple[bool, int]:
    """keep = roles ingested ('user', 'assistant'). Returns (hit, n_spans). hit=None-like False if no evidence kept."""
    st = TaintMemStore(ChannelRegistry(), retriever=retriever)
    evidence: set[str] = set()
    for sess in item["haystack_sessions"]:
        for turn in sess:
            role = turn.get("role")
            if role not in keep or not turn.get("content", "").strip():
                continue
            rec = st.add(turn["content"], "user" if role == "user" else "agent", "owner" if role == "user" else "assistant")
            if turn.get("has_answer"):
                evidence.add(rec.id)
    if not evidence:
        return False, len(st.spans())
    hits = st.search(item["question"], k)
    return any(h.record.id in evidence for h in hits), len(st.spans())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--k", type=int, default=8); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    items = load_longmemeval(a.path, a.limit)
    t0 = time.time()
    modes = (("all turns", {"user", "assistant"}), ("user only", {"user"}))
    res = {}
    for ret in ("dense", "bm25", "hybrid"):
        for name, keep in modes:
            d = defaultdict(list)
            for it in items:
                hit, _ = recall_at_k(it, a.k, keep, ret)
                d[it["question_type"]].append(int(hit)); d["ALL"].append(int(hit))
            res[f"{ret:<7}{name}"] = d
    print(f"{len(items)} items, k={a.k}, {time.time() - t0:.0f}s  (recall of an evidence turn; oracle haystack = optimistic)")
    types = sorted(next(iter(res.values())))
    print(f"{'retriever / memory':<22}" + "".join(f"{t[:13]:>14}" for t in types))
    for name, d in res.items():
        print(f"{name:<22}" + "".join(f"{sum(d[t]) / len(d[t]):>14.3f}" for t in types))
    print(f"{'n':<22}" + "".join(f"{len(next(iter(res.values()))[t]):>14}" for t in types))


if __name__ == "__main__":
    main()
