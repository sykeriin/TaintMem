# Findings (updated 2 Nov 2026)

**Status: the full pipeline exists and is tested, but no research claim is supported yet.** No real LLM has been
run (no API key / GPU). The bundled "LLMs" are deterministic rule-based stand-ins and the benchmarks are
synthetic and written by the authors. Reproduce: `python -m evals.sweep`, `python -m evals.grid --backbone toy`.

## 1. Correction to my week-1 note
Week 1 said sentence-level ("span") propagation was "no better than conservative" under heavy paraphrase +
mixing (creep 1.00). Adding an **oracle floor** (the best any sentence-level labeller can do: label = min true
integrity of tokens in the sentence) shows part of that creep is **unavoidable**: if the rewriter fuses a trusted
and an untrusted fact into one sentence, no sentence-level scheme can keep the trusted fact trusted.

## 2. Held-out toy results (seed 1000, n=160, k=3 passes; full tables in docs/RESULTS_TOY.md)
| regime | conservative | lineage | partitioned | span (plain) | span (IDF) | oracle floor |
|---|---|---|---|---|---|---|
| easy: creep | 1.000 | 1.000 | 0.000 | 0.025 | 0.075 | 0.025 |
| drift 0.6 + mixing: creep | 1.000 | 1.000 | 0.000 | 0.999 | **0.750** | 0.750 |
| drift 0.9 + mixing: creep | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.750 |
| synthesis kept (cross-class facts in one sentence) | 1.00 | 1.00 | **0.00** | 1.00 | 1.00 | 1.00 |
| laundering (all regimes, all families) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Reading it (what the data supports, no more):
* Conservative/lineage creep is 1.0 by construction (every scenario holds one untrusted email), so that column
  only shows they cannot keep any trusted fact trusted once one untrusted input is in context.
* Span ≈ oracle in the easy regime and (with IDF) at moderate drift. At drift 0.9 attribution collapses and span
  falls back to the conservative label: safe, but 0.25 above the floor.
* Partitioned has zero creep and zero laundering but **destroys all cross-class synthesis**; its apparent win is
  a utility cost the creep metric alone hides.
* **Zero laundering is not a result**: the rule-based summariser always keeps identifiers and obeys exactly one
  injection template. Real LLMs and adaptive attackers are untested.

## 3. A confirmed trade-off: identifier anchoring
`anchor_dominates=True` (trust an exact-identifier match over weaker lexical hits) reduces boilerplate creep but
launders **6.7%** in the easy regime on family g (injection says "the account on file"; the rewriter inserts the
*trusted* IBAN, so the sentence anchors only to a trusted span). It is therefore off by default. This family was
designed by me to probe exactly this, so the number shows the mechanism exists, not how common it is.

## 4. Implementation findings
* Per-call IDF is unstable on tiny batches (a correct attribution scored 0.285 < tau_taint 0.30 with 2 inputs).
  Fixed by computing document frequency over the whole store and falling back to plain cosine under 4 documents.
* MINJA-style attackers (a different *user* writing to shared memory) are labelled TRUSTED unless the host
  authenticates principals (`trusted_users`). Tested in both directions.

## 5. Agent-level toy grid (`evals.grid --backbone toy`)
Undefended ASR 1.00 on all four attack types; every gated configuration 0.00; auditor named the poisoned span in
all audited cases. **This only validates plumbing**: the backbone is a perfectly gullible heuristic and benign
tasks never need untrusted facts. `strict` mode gated 100% of tasks (util auto 0.00): the burden/utility trade-off
is real, its size on real tasks is unknown.

## 6. Unverified / blocked
* Real-model numbers, MPBench, LongMemEval: blocked on API key / GPU. Backends tested only against a local fake server.
* MPBench loader uses a **guessed schema** (the repo URL I tried served a different project); LongMemEval schema
  is from memory. Both fail loudly on mismatch.
* Statistical power: toy confidence intervals are over synthetic scenarios, not over data distributions.

## 7. Remaining work that only you / a model can unblock
| item | blocked on |
|---|---|
| Any real-LLM number (RQ2-RQ5) | API key, hosted open-model endpoint, or a local model via `ollama:` (small models only: see README) |
| MPBench run | correct data location; loader schema is a guess |
| LongMemEval run | the dataset file (schema in loader is from memory) |
| Hand-annotating ~200 sentences to validate fact-token ground truth | a human |
| Paper numbers | all of the above; skeleton marks them PENDING |
| Optional encoders (`SentenceTransformerEmbedder`, `NLIVerifier`) | written and tested with fakes only; not run with the real libraries |
