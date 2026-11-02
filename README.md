# TaintMem

Provenance-tracked memory for AI agents (memory-poisoning defence). Research prototype.
Plan: [docs/PLAN.md](docs/PLAN.md) · Findings and caveats: [docs/FINDINGS.md](docs/FINDINGS.md) ·
Threat model: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) · Paper skeleton (results PENDING): [docs/paper/main.md](docs/paper/main.md)

```bash
pip install -e .                                   # only dependency: numpy
python -m unittest discover -s tests               # 62 tests
python examples/demo_laundering.py                 # planted payload -> 2 rewrites -> BLOCK -> auditor
python -m evals.sweep                              # dev sweep + held-out toy benchmark -> docs/RESULTS_TOY.md
python -m evals.grid --backbone toy --n 80         # config x attack grid (toy backbone)
python -m evals.grid --backbone anthropic:<model> --n 60 --yes   # real run, needs ANTHROPIC_API_KEY (untested)
python -m taintmem.cli --db mem.sqlite list|why|purge|endorse|viewer
```

| module | role |
|---|---|
| `labels`, `channels` | integrity lattice; labels come from the channel (and authenticated principal), never the text |
| `store`, `record` | labelled spans, search, SQLite + HMAC tamper check |
| `attribution`, `propagate`, `rewrite` | sentence-level label propagation through LLM rewrites: conservative / lineage / partitioned / **span** |
| `policy`, `agent/loop` | strict / balanced gating; retrieve→decide→gate→confirm→execute→audit loop |
| `auditor` | counterfactual re-runs: group ablation, bisection, random-ablation surrogate |
| `lifecycle`, `provenance`, `viewer`, `cli` | TTL, purge, user-bound endorsement, "why do you remember this", HTML view |
| `llm/backends` | Anthropic / OpenAI-compatible / cached / function LLMs (stdlib HTTP) |
| `adapters/` | LangGraph-store-shaped and Mem0-shaped facades (duck-typed; those libraries are not installed here) |
| `evals/` | laundering benchmark (+oracle floor), attack scenarios, grid, LongMemEval/MPBench loaders |

**Limits:** no real LLM has been run; results so far use rule-based stand-ins (see FINDINGS).
