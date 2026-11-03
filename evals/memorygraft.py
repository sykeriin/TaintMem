"""Real attack data from MemoryGraft (arXiv 2512.16962): 100 benign + 10 poisoned 'experiences' and 12 queries.

LLM-free check of two things on REAL poisoned records:
  1. poisoned-retrieval proportion (PRP) of OUR retriever (their paper: 23/48 = 47.9% with GPT embeddings + BM25);
  2. provenance coverage: what share of retrieved records / tasks carry an UNTRUSTED label if the poisoned ones
     entered through an untrusted channel and the benign ones through the agent's own (SEMI) channel.
This does NOT measure whether a model would follow a poisoned precedent. Queries are read with `ast` (the repo's
experiment.py is third-party code and is never imported or executed).

python -m evals.memorygraft REPO_DIR [--k 3]
"""
from __future__ import annotations

import argparse
import ast
import json
import os

from taintmem import ChannelRegistry, Integrity, TaintMemStore


def load(repo: str):
    seeds = json.load(open(os.path.join(repo, "memorygraft/payloads/experience_seeds.json")))
    src = open(os.path.join(repo, "memorygraft/experiment.py")).read()
    queries = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "EVALUATION_QUERIES" for t in node.targets):
            queries = ast.literal_eval(node.value)
    if not queries:
        raise ValueError("EVALUATION_QUERIES not found")
    return seeds["benign_experiences"], seeds["poisoned_experiences"], queries


def run(benign, poisoned, queries, k: int = 3) -> dict:
    st = TaintMemStore(ChannelRegistry())
    pois_ids: set[str] = set()
    for e in benign:
        st.add(f'{e["req"]} {e["resp"]}', "agent", "own-runs")                  # agent's own history: SEMI
    for e in poisoned:
        pois_ids.add(st.add(f'{e["req"]} {e["resp"]}', "tool", "attacker-readme", "fastclean.example").id)
    hit = total = untrusted = tasks_touched = 0
    for q in queries:
        got = st.search(q, k)
        total += len(got)
        hit += sum(h.record.id in pois_ids for h in got)
        u = sum(h.span.label.integrity == Integrity.UNTRUSTED for h in got)
        untrusted += u
        tasks_touched += u > 0
    return {"retrieved": total, "poisoned_hits": hit, "prp": hit / max(total, 1),
            "store_poison_share": len(pois_ids) / (len(benign) + len(poisoned)),
            "flagged_share_of_retrieved": untrusted / max(total, 1),
            "poison_hits_flagged": (untrusted == hit), "tasks_touched": tasks_touched, "tasks": len(queries)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("repo"); ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    b, p, q = load(a.repo)
    r = run(b, p, q, a.k)
    print(f"store: {len(b)} benign + {len(p)} poisoned ({r['store_poison_share']:.1%} poisoned); {r['tasks']} queries, top-{a.k}")
    print(f"our retriever PRP = {r['poisoned_hits']}/{r['retrieved']} = {r['prp']:.1%}   (paper: 23/48 = 47.9%, different retriever)")
    print(f"retrieved records carrying an UNTRUSTED label: {r['flagged_share_of_retrieved']:.1%}; "
          f"every poisoned hit flagged: {r['poison_hits_flagged']}; tasks touched by flagged memory: {r['tasks_touched']}/{r['tasks']}")


if __name__ == "__main__":
    main()
