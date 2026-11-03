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


def recall_at_k(item: dict, k: int, keep: set[str]) -> tuple[bool, int]:
    """keep = roles ingested ('user', 'assistant'). Returns (hit, n_spans). hit=None-like False if no evidence kept."""
    st = TaintMemStore(ChannelRegistry())
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
    res = {name: defaultdict(list) for name in ("all turns", "user turns only (trusted-only memory)")}
    for it in items:
        for name, keep in (("all turns", {"user", "assistant"}), ("user turns only (trusted-only memory)", {"user"})):
            hit, _ = recall_at_k(it, a.k, keep)
            res[name][it["question_type"]].append(int(hit))
            res[name]["ALL"].append(int(hit))
    print(f"{len(items)} items, k={a.k}, {time.time() - t0:.0f}s  (retrieval recall of an evidence turn; oracle haystack)")
    types = sorted(res["all turns"])
    print(f"{'memory':<40}" + "".join(f"{t[:14]:>16}" for t in types))
    for name, d in res.items():
        print(f"{name:<40}" + "".join(f"{sum(d[t]) / len(d[t]):>16.3f}" for t in types))
    print(f"{'n':<40}" + "".join(f"{len(res['all turns'][t]):>16}" for t in types))


if __name__ == "__main__":
    main()
