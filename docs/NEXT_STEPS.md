# Next steps (as of 3 Nov 2026)

**Where things stand.** The system, benchmarks, tooling and docs are built and tested (73 tests). **No real LLM has
ever been run**, so no research claim is supported yet. Everything below is ordered by what unblocks the most.

## A. Needs you (I cannot do these from the cloud session)
| # | Action | Why | How |
|---|---|---|---|
| A1 | **Run the local preflight** on your laptop | First contact with a real model; backends are only tested against a fake server | see `docs/LOCAL_RUN.md` |
| A2 | **Get any real-model access** beyond the 3050 | Small local models (<=8B) cannot reproduce MPBench-style claims; GPT-OSS-120B is out of reach locally | hosted open-model endpoint (`openai:<url>\|<model>`) or an API key; estimate cost first (`evals.grid` prints it, needs `--yes`) |
| A3 | **Ask the MPBench authors for the data** | The paper gives no release link; the repo I was first pointed to is MemoryGraft | draft in `docs/outreach/mpbench_data_request.md` (not sent) |
| A4 | **Hand-annotate 200 sentences** | The benchmark's ground truth is fact-token based and unvalidated | `python -m evals.annotate make DIR`, fill `sheet.csv` (blind), `python -m evals.annotate score DIR` |
| A5 | **Decide the venue/deadline** | I do not know current deadlines and will not guess | check the target venue's CFP |

## B. Needs a model (I can do these once A1/A2 give me output)
1. Run `evals.preflight` output through me; fix any API/format quirks (expect some).
2. `evals.grid` on the real backbone, small `--n` first; report ASR / utility / confirmation burden with CIs.
3. Replace the rule-based summariser with the real LLM in the laundering benchmark (`Rewriter` already takes any `LLM`);
   this is the first test of the central claim (sentence-level labels vs conservative vs oracle floor).
4. Real encoder for attribution (`SentenceTransformerEmbedder`, optional NLI) and re-run the sweep on dev seeds only.
5. LongMemEval with a real answerer (oracle file is already verified; full `_s` haystack is ~250 MB).

## C. Can be done now without a model (candidates, in value order)
1. Add adaptive attacks that target *attribution itself* (paraphrase until cosine < threshold while keeping the
   payload meaning) — the benchmark cannot currently launder anything, which is a weakness, not a result.
2. Utility-side metric for gated *actions* (confirmation burden on realistic benign tasks, not just the payment toy).
3. Implicit-flow test (untrusted memory changes *which* trusted memory is retrieved) to quantify the strict-mode gap.
4. Packaging: PyPI metadata, type hints pass, API docs.

## D. Go / no-go rules (written before seeing real numbers)
* Span-level laundering <= conservative + 2 points on non-adaptive families, **with real-LLM rewriters**.
* Label creep at 3 passes >= 30% below conservative **and within a stated margin of the oracle floor**.
* Utility retention >= 90% of undefended on a real long-memory task.
* If adaptive attribution attacks push laundering above 5%, narrow the claim and report it as a limitation.
* If span does not beat conservative + partitioned on utility at equal safety, the paper becomes an analysis of
  creep/laundering/oracle-floor trade-offs (still reportable), not a "new defence" paper.

## E. Honest timeline
The 8-week plan started 2 Nov with a 9 Nov compute deadline. As of today there is no compute, so the schedule is
slipping day-for-day. Earliest realistic point for first real numbers: the day A1/A2 produce output.

## F. Known weak spots to keep in mind
* Toy "LLM"s and toy attackers: zero laundering is a property of the harness.
* Identifier anchoring flatters attribution on this benchmark.
* Hashed/IDF lexical attribution collapses under heavy paraphrase (falls back safe, but costly).
* `anchor_dominates` launders on reference-resolution (6.7% on family g), so it stays off.
* MINJA-style attackers need authenticated principals; social-engineered endorsement is not defended.
* MPBench: no verifiable release; loader follows the paper's Appendix D.2 schema, fixture-tested only.
