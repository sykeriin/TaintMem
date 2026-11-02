# TaintMem: sentence-level provenance for agent memory under LLM rewriting  (DRAFT SKELETON)

> Every number below is PENDING a real-model run. Do not fill from the toy benchmark.

## 1. Introduction
Memory poisoning turns one prompt injection into a persistent compromise. Record-level lineage (MemLineage,
2605.14421) and conservative inheritance (TMA-NM, 2606.24322) exist; both label whole records. We study
**sentence-level label propagation through summarisation/merge/reflection** and measure label creep against
laundering and utility.

## 2. Threat model  (see docs/THREAT_MODEL.md)
## 3. System
Channel labels, Biba low-water-mark, propagation modes (conservative / lineage / partitioned / span),
policy (strict/balanced), counterfactual auditor, lifecycle.
## 4. Evaluation
### 4.1 Suites: MPBench subset (PENDING schema check) | laundering set (this repo) | LongMemEval subset (PENDING)
### 4.2 Metrics: ASR, utility retention, confirmation burden, creep (+ oracle floor), laundering, synthesis kept, auditor precision
### 4.3 Results
| Table | Status |
|---|---|
| MPBench ASR by config x backbone | PENDING (needs API/GPU) |
| LongMemEval utility retention | PENDING |
| Laundering set, real-LLM rewriters | PENDING |
| Auditor precision, split payloads | PENDING |
| Toy-benchmark machinery check | done: docs/RESULTS_TOY.md (NOT evidence about LLMs) |
## 5. Limitations
Lexical attribution; identifier anchoring is benchmark-friendly; oracle floor shows part of "creep" is unavoidable
when a rewriter fuses trusted and untrusted facts into one sentence.
## 6. Related work
CaMeL, FIDES, RTBAS, A-MemGuard, MemLineage, TMA-NM, NeuroTaint, SENTINEL/FARMA, MINJA, AgentPoison, eTAMP, MPBench.
