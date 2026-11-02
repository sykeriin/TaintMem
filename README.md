# TaintMem

Provenance-tracked memory for AI agents (memory-poisoning defence). Research prototype, week 1.
Full plan: [docs/PLAN.md](docs/PLAN.md). Current results and caveats: [docs/FINDINGS.md](docs/FINDINGS.md).

```bash
pip install -e .                      # only dependency: numpy
python -m unittest discover -s tests  # 31 tests
python examples/demo_laundering.py    # planted payload -> 2 rewrites -> policy BLOCK -> auditor
python -m evals.run --n 100 --passes 1 2 3 [--drift 0.6 --mix]
```

| module | role |
|---|---|
| `labels`, `channels` | integrity/confidentiality lattice; labels come from the channel, never the text |
| `store`, `record` | labelled spans, similarity search, SQLite + HMAC tamper check |
| `propagate`, `rewrite` | label propagation through LLM rewrites: conservative / lineage / partitioned / **span** |
| `policy` | strict (context taint) and balanced (argument taint) gating of consequential tools |
| `auditor` | counterfactual re-runs: group ablation, bisection, random-ablation surrogate |
| `lifecycle` | TTL expiry, domain purge, user-event-bound endorsement |

Limits: the bundled "LLM" is a rule-based stand-in; real backends are not implemented yet.
