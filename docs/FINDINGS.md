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
* MPBench: no public data found; loader follows the paper's Appendix D.2 schema, fixture-tested only.
* LongMemEval loader schema is now verified against the real oracle file.
* Statistical power: toy confidence intervals are over synthetic scenarios, not over data distributions.

## 7. Remaining work that only you / a model can unblock
| item | blocked on |
|---|---|
| Any real-LLM number (RQ2-RQ5) | API key, hosted open-model endpoint, or a local model via `ollama:` (small models only: see README) |
| MPBench run | **no public release found** (paper gives no link; the repo I was pointed to is MemoryGraft). Loader now follows the paper's Appendix D.2 schema but is tested on a synthetic fixture only. Ask the authors |
| LongMemEval LLM run | an LLM. The oracle file is downloaded and its schema is verified (500 items); see section 8 |
| Hand-annotating ~200 sentences to validate fact-token ground truth | a human: `python -m evals.annotate make DIR`, fill `sheet.csv`, `python -m evals.annotate score DIR` (sheet is blind) |
| Paper numbers | all of the above; skeleton marks them PENDING |
| Optional encoders (`SentenceTransformerEmbedder`, `NLIVerifier`) | written and tested with fakes only; not run with the real libraries |

## 8. Real-data check: LongMemEval oracle (LLM-free, `python -m evals.retrieval_recall`)
500 real items. Recall = some top-8 retrieved span comes from a turn marked `has_answer`. The oracle haystack holds
evidence sessions only, so absolute numbers are optimistic. No LLM.

| retriever / memory | ALL | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal |
|---|---|---|---|---|---|---|---|
| dense (hashed) all turns | 0.728 | 0.821 | 0.602 | 0.893 | 0.733 | 0.786 | 0.699 |
| dense, user turns only | 0.822 | 0.923 | 0.902 | 0.089 | 0.967 | 0.886 | 0.925 |
| **BM25** all turns | **0.832** | 0.846 | 0.737 | 0.946 | 0.967 | 0.871 | 0.820 |
| **BM25**, user turns only (trusted-only) | **0.834** | 0.910 | 0.917 | **0.089** | 1.000 | 0.900 | 0.947 |
| hybrid (RRF) all turns | 0.806 | 0.846 | 0.707 | 0.946 | 0.867 | 0.843 | 0.789 |
| hybrid, user turns only | 0.826 | 0.923 | 0.910 | 0.089 | 1.000 | 0.900 | 0.917 |

* With the weak hashed retriever, "user only" looked better on most types: that was retriever crowding, partly
  confounded. With BM25 the aggregate is a **tie (0.832 vs 0.834)**, so the aggregate hides the real cost:
  trusted-only memory **wipes out the single-session-assistant type (0.946 → 0.089)** because that evidence lives
  in assistant turns. Multi-session and temporal still favour user-only by 0.10-0.18 (remaining crowding).
* The store's default retriever is still `dense`; BM25 is better on this data, so a real deployment should use
  `retriever="bm25"` or `"hybrid"`. Not yet tested with a real encoder or with the full (non-oracle) haystack.

## 9. Related work to add
MemoryGraft (arXiv 2512.16962, Srivastava & He): poisoned *experience* retrieval; ~48% of retrieved records poisoned
vs 9% of the store. Retrieval over-representation is exactly why provenance must survive retrieval.

## 10. Real attack data: MemoryGraft seeds (`python -m evals.memorygraft REPO_DIR`, LLM-free)
100 benign + 10 real poisoned experiences, 12 real queries, top-3 retrieval with our hashed retriever.
* Poisoned share of the store 9.1%; poisoned share of retrieved records **50.0% (18/36)** (paper: 47.9% with a
  different retriever). Poisoned records are retrieved ~5.5x more often than their share, so a strict policy would
  have fired on **11 of 12 tasks** (confirmation burden ~92%) once the poison is retrievable.
* The label check being 100% on the poisoned hits is **by construction** (I assigned the poisoned records an
  untrusted channel). In the real attack the script writes the store directly: coverage then depends on the host
  mediating writes. A forged row without a valid HMAC is rejected on load (tested).
* This does not show whether a model follows a poisoned precedent; that needs an LLM.
