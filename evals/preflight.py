"""Sanity-check a backbone BEFORE a batch: python -m evals.preflight --backbone ollama:qwen2.5:3b
Runs one benign + one attacked scenario unprotected and prints the raw output, so format failures
(model not returning JSON) are visible instead of being silently scored as errors."""
from __future__ import annotations

import argparse

from taintmem.agent.loop import build_prompt, parse_json
from taintmem import ChannelRegistry, TaintMemStore

from .attacks import generate_agent
from .grid import make_backbone


def preflight(llm, n_each: int = 1) -> dict:
    out = {"ok": 0, "total": 0, "samples": []}
    for sc in [s for s in generate_agent(10, 0) if s.attack in ("benign", "direct_email")][:2 * n_each]:
        st = TaintMemStore(ChannelRegistry())
        for k, sid, dom, t in sc.records:
            st.add(t, k, sid, dom)
        raw = llm.complete(*build_prompt([r.span for r in st.search(sc.task, 6)], sc.task, True))
        parsed = parse_json(raw)
        out["total"] += 1; out["ok"] += bool(parsed and ("tool" in parsed or "answer" in parsed))
        out["samples"].append({"attack": sc.attack, "raw": raw[:300], "parsed": parsed})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", required=True)
    a = ap.parse_args()
    llm, _ = make_backbone(a.backbone)
    r = preflight(llm)
    for s in r["samples"]:
        print(f"[{s['attack']}] parsed={s['parsed']}\n  raw: {s['raw']!r}")
    print(f"\nformat OK: {r['ok']}/{r['total']}", "" if r["ok"] == r["total"] else
          "  <- model is not following the JSON format; grid results would mostly measure format errors")


if __name__ == "__main__":
    main()
