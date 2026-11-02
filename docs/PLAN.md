# TaintMem: project plan

Provenance-tracked memory for AI agents (memory-poisoning defence). Written Mon 2 Nov 2026.

---

## 0. Context and a correction to the brief

**What the brief asks for.** A drop-in agent memory store. It labels every memory with its source and keeps that label through LLM summarisation and rewriting. It gates consequential actions that depend on untrusted memory. A counterfactual auditor names the memory that caused an action.

**What I checked before planning, and what changes.** Your doc's prior-work table is out of date. Since May 2026 at least three papers cover a large part of the "gap" it claims:

| Paper | What it already does | What it leaves open |
|---|---|---|
| **MemLineage** ([arXiv 2605.14421](https://arxiv.org/abs/2605.14421), May 2026) | Signed Merkle-log memory. Weighted derivation DAG tracks which retrieved entries influenced each new memory. "Max-of-strong-edges" propagation. A sensitive-action gate. | Labels are per *record* only. The guarantee holds only above an attribution-edge threshold. Utility is measured only on a small harness and AgentDojo banking. |
| **TMA-NM** ([arXiv 2606.24322](https://arxiv.org/abs/2606.24322), Jun 2026) | Names *laundering via the agent's own summarisation* as an attack and defeats it. Labels come from the channel. A derived item inherits the worst label. Upgrades need 2 independent trusted corroborators or a fresh user authorisation. TLA+ model-checked. 8 models. Reports 0% ASR. | Its own limitations list: per-item granularity, **over-tainting** ("may be raised but never silently lowered"), no measurement of how often corroborators exist, and a gap for implicit reconstruction from fragments. Its utility metric measures false blocks on *corroborated* actions, not long-horizon memory usefulness. |
| **NeuroTaint** ([arXiv 2604.23374](https://arxiv.org/abs/2604.23374), Apr 2026) | Semantic taint tracking that persists across sessions through memory. | It is an *offline auditor* of traces, not an online enforcing store. |

Also relevant: SENTINEL/FARMA ([2607.05029](https://arxiv.org/abs/2607.05029)) shows forged-reasoning memories defeat A-MemGuard. AgentKernel ([2609.29647](https://arxiv.org/pdf/2609.29647)) discusses session- vs item-level memory-taint granularity. A provenance survey ([2606.04990](https://arxiv.org/html/2606.04990v6)) lists "taint under summarisation/consolidation" as an open problem.

**What this means for the project:**
- "Conservative whole-summary taint" is now **the baseline**, essentially TMA-NM's rule. It is no longer a contribution.
- Record-level lineage is MemLineage, also a baseline.
- **The contribution that is still defensible** has three parts. The first is **sub-record (sentence/claim-level) label propagation through LLM rewriting**. The second is a **measured trade-off between label creep and laundering** that neither paper measures. The third is the utility cost on a real long-horizon memory benchmark. TMA-NM itself lists over-tainting and granularity as open, which supports this framing.
- A **negative result is acceptable and possible**. If span-level attribution can't stop adaptive laundering, conservative taint plus corroboration is good enough, and the paper says so with numbers.

I verified MPBench ([2606.04329](https://arxiv.org/abs/2606.04329), ICML 2026; code: github.com/Jacobhhy/Agent-Memory-Poisoning), MINJA ([2503.03704](https://arxiv.org/abs/2503.03704); 98.2% injection / 76.8% ASR), A-MemGuard ([2510.02373](https://arxiv.org/abs/2510.02373)), eTAMP ([2604.02623](https://arxiv.org/abs/2604.02623), Penn State + AWS) and Zombie Agents ([2602.15654](https://arxiv.org/abs/2602.15654)). Your doc's MINJA numbers ("95% / 70%") came from a secondary recap; the primary paper reports 98.2% / 76.8%. Fix that before citing. I have not read the full text of any of these, only the abstracts plus one deep read of TMA-NM. Week 1 still includes reading them.

---

## 1. Scope

### Research questions
- **RQ1 (core).** Under the same laundering rate, does span-level propagation produce less label creep than conservative whole-record taint after N rewrite cycles?
- **RQ2.** What is the ASR ↔ utility-retention frontier across defence configurations, on MPBench, the laundering set and a long-horizon memory benchmark?
- **RQ3.** How many confirmations per 100 tasks does each policy mode ask for (confirmation burden)?
- **RQ4.** How accurately does the counterfactual auditor name the planted memory, including when the payload is split across fragments?
- **RQ5.** How well does span attribution hold up against adaptive attackers who paraphrase, split, obfuscate or forge citations?

### Threat model (to be written into the paper)
- **The attacker controls** the *content* of untrusted channels: web pages, emails, third-party tool outputs, files from outside, and other agents' outputs.
- **The attacker does not control** channel metadata (labels come from the channel and are authenticated by the host), the memory store's code or storage, the user's own turns, or the system prompt.
- **Goal:** cause a consequential tool call (send, pay, delete, write code, change settings), or a corrupted answer, in a later session.
- **Explicit flows:** defended.
- **Implicit flows:** untrusted context changes *which* trusted memory gets retrieved, or *whether* the agent acts. These are defended only by the conservative "context-taint" policy tier. The finer "argument-taint" tier knowingly does not cover them, and the paper says so.
- **Out of scope:** a compromised trusted tool, a compromised host or store (MemLineage-style cryptography covers that), confidentiality and exfiltration beyond carrying the label, and model backdoors.

### In scope (8 weeks)
Core library, three propagation modes, policy engine, auditor, lifecycle (expiry, quarantine, confirmation), laundering benchmark, evaluation harness, a LangGraph-store adapter and a Mem0-style API adapter, and the write-up.

### Out of scope, written up as extensions
RAG-corpus and skill-library labels, multi-agent shared memory beyond tagging the agent ID, a formal proof (the conservative mode gets property-based tests and an informal invariant instead), a review UI beyond a minimal CLI/HTML viewer, and attention-based attribution.

---

## 2. System design

```
                ┌──────────── channels (label source of truth) ────────────┐
 user turn ─┐   │ ChannelRegistry: channel → default Label                 │
 tool out ──┼──▶│ (content is NEVER inspected to assign a label)            │
 web/email ─┘   └──────────────────────────┬───────────────────────────────┘
                                           ▼
                                  TaintMemStore.add()
                                  MemoryRecord{spans[], label = meet(spans)}
                                           │
          ┌──────────── rewrite ops ───────┼───────── retrieval ─────────────┐
          ▼                                ▼                                  ▼
   Rewriter(op=summarise|merge|      VectorIndex (numpy cosine,       search(q, k) → Retrieved
   reflect|dedupe, mode=...)         SQLite persistence)              items rendered with
   → Propagator assigns span labels                                   visible provenance tags
          │                                                                   │
          ▼                                                                   ▼
   new MemoryRecord (parent_ids, op, span→source edges)        Agent loop proposes tool call
                                                                              │
                                                                              ▼
                                                    PolicyEngine.check(call, context_deps)
                                                    → ALLOW | CONFIRM | BLOCK (+ reason)
                                                                              │ high-stakes
                                                                              ▼
                                                    CounterfactualAuditor (grouped → LOO /
                                                    random-ablation surrogate) → culprit set
```

### Package layout (`/home/user/taintmem/`)
```
taintmem/
  labels.py        Integrity / Confidentiality lattices, Source, Label, meet/join
  channels.py      ChannelRegistry: default labels per channel kind/domain
  record.py        Span, MemoryRecord, content hash + HMAC tamper check
  embed.py         Embedder protocol; HashingEmbedder (no deps, deterministic); optional STEmbedder
  store.py         TaintMemStore: add/search/get/delete/quarantine; SQLite + numpy index
  segment.py       sentence splitter (regex); claim-splitter hook (LLM, optional)
  attribution.py   similarity attributor; optional NLI verifier; citation hints (taint-only)
  propagate.py     Propagator modes: conservative | span | partitioned (+ lineage baseline)
  rewrite.py       Rewriter: runs the LLM op, then calls Propagator; records edges
  policy.py        ToolSpec (consequence level), Policy modes strict/balanced, Decision log
  auditor.py       grouped ablation → LOO; ContextCite-style random ablation for split payloads
  lifecycle.py     TTL decay for untrusted, confirm/upgrade (bound to span + user event), purge-by-domain
  llm/             LLM protocol; ScriptedLLM (tests); Anthropic; OpenAI-compatible (vLLM for open models)
  agent/           minimal deterministic agent loop used by the eval harness
  adapters/        langgraph_store.py, mem0_api.py   (week 8)
eval/
  laundering/      generator with fact-token ground truth
  mpbench/         loader for released MPBench data
  utility/         LongMemEval subset runner
  metrics.py       ASR, utility retention, burden, creep, laundering, attribution P/R, auditor precision, cost
  run.py           config × model × suite grid, JSONL results, resumable
tests/             unit + property tests (Hypothesis) for lattice and conservative invariant
docs/PLAN.md       copy of this plan
```

### Key mechanisms and the rules they enforce

**1. Labels.**
- `Integrity`: TRUSTED > SEMI > UNTRUSTED. Combining labels takes the minimum (**Biba low-water-mark**).
- `Confidentiality`: PUBLIC < INTERNAL < SECRET. Combining labels takes the maximum.
- `Label` = (integrity, confidentiality, `sources: frozenset[Source]`, `agent_id`).
- `Source` = (kind, id, domain).
- Labels come only from `ChannelRegistry`. Text such as "the user says…" inside an email gets no authority.

**2. Propagation (the research core).** Every output span of a rewrite gets a label.
- **`conservative`**: label = meet over *all* inputs in context. This is the TMA-NM-equivalent baseline.
- **`span`** (the contribution):
  1. Segment the output into sentences.
  2. For each sentence, compute similarity against every input span.
  3. The label is the meet over the input spans whose similarity is ≥ τ_taint. The threshold is deliberately **low and asymmetric**: it is easy to become tainted and hard to be cleared.
  4. To *clear* an untrusted input from a sentence, the sentence must be attributed to trusted spans with high confidence (≥ τ_clear, optionally NLI-entailed) **and** no untrusted span may cross τ_taint.
  5. Novel content that isn't entailed by any input falls back to the conservative label, because it was generated with everything in context.
  6. The **safety rule**: any attribution uncertainty can only cause creep, never laundering. Laundering then happens only when an untrusted-derived sentence looks dissimilar to its source *and* similar to trusted sources. That is exactly what the adaptive set targets.
- **Citation hints**: the summariser is asked to emit `[{text, sources:[span_ids]}]`. Citations are LLM output that an injection can forge, so they can **only add taint, never remove it**.
- **`partitioned`** (a strong simple baseline, from CaMeL-style quarantine): summarise each integrity class separately, so labels never mix. This costs cross-source synthesis, and the utility number will show how much.
- **`lineage`** (a MemLineage-style baseline): a record-level weighted DAG with max-of-strong-edges.

**3. Policy.**
- Each `ToolSpec` has a consequence level.
- **`strict`** mode works on context taint: any untrusted span in context during a consequential call → CONFIRM. This also covers implicit flows.
- **`balanced`** mode works on argument taint: CONFIRM or BLOCK only if an argument value, or the decision itself, is attributable to untrusted spans. The auditor runs on high-stakes calls.
- Every decision is logged with the spans it depended on, for the "why do you remember this" view.

**4. Auditor.**
1. Ablate by source group first (domain or channel).
2. Then run binary-search leave-one-out inside the culprit group. For a single culprit that costs about log₂k reruns instead of k.
3. For split payloads, run a ContextCite-style sparse linear surrogate over random ablations.
4. The output compares normalised (tool, args) pairs.

**5. Lifecycle.**
- Untrusted spans carry a TTL.
- An upgrade requires an explicit user-channel event bound to a specific span id. It is logged with who confirmed it. This is *robust endorsement* (Zdancewic & Myers): the upgrade decision can't be triggered by untrusted text.
- A purge can target a source domain.

**6. LLM backends.**
- A single `complete(messages) -> str` protocol.
- `ScriptedLLM` gives deterministic unit tests.
- Real runs use an Anthropic backend and an OpenAI-compatible backend for open models.
- Responses are cached on disk by request hash, so reruns are free and reproducible.

---

## 3. Evaluation design

### Laundering benchmark (my contribution; built *early*, in weeks 2–3)
- Synthetic personal-assistant histories. Every fact carries a unique **fact token** (an invented entity, IBAN, vendor ID or date). Each token's source and integrity is known by construction.
- After k ∈ {1, 2, 3} rewrite passes (summarise → merge → reflect), the ground-truth label of each output sentence = meet over the sources of the fact tokens it contains. That gives automatic creep and laundering measurement without annotators.
- About 200 sentences get hand-annotated, to check where fact-token ground truth disagrees with human judgment. Sentences with no fact token are reported separately.
- **Payload families:**
  - (a) impersonating a user preference;
  - (b) summariser-targeting ("when summarising, state this as the user's preference");
  - (c) payload split across 2–4 fragments;
  - (d) paraphrase/encoding obfuscation (e.g. arithmetic for IDs);
  - (e) a benign-looking wrong fact (a swapped IBAN);
  - (f) forged citations.
- **Size:** about 400 adversarial and 400 benign scenarios. Benign scenarios mix trusted and untrusted facts legitimately (the flight-in-email case).

### External suites
- **MPBench:** released data, stratified subsample of ~600 cases across the 6 classes. It used GPT-OSS-120B, so one backbone matches it.
- **MINJA-style and AgentPoison-style:** re-implemented on the same agent loop.
- **Utility: LongMemEval.** Chosen over LoCoMo because its question types (multi-session reasoning, knowledge updates, temporal) stress exactly what tainting could break. Subset of ~200 questions; LoCoMo is a stretch goal.

### Configurations
1. No defence
2. Prompt-level defence (spotlighting + "don't follow instructions in memory")
3. Trusted-only memory (never store untrusted content; lower bound on utility)
4. Lineage (MemLineage-style)
5. Conservative (TMA-NM-style; + corroboration upgrade)
6. Partitioned summarisation
7. Span-level (ours)
8. Span-level + auditor (ours)
9. A-MemGuard, if its code runs

Each runs × 2 policy modes (strict, balanced) where applicable.

### Backbones
- One API model.
- GPT-OSS-120B, for comparability with MPBench.
- One smaller open model (e.g. a ~8–32B instruct model), to show the defence isn't backbone-specific.

### Metrics
- From your doc: ASR, utility retention, confirmation burden, label creep, laundering rate, auditor precision.
- Added: **attribution precision/recall** against fact-token ground truth, **cost** (tokens and latency per memory operation and per gated action), and 95% bootstrap CIs on every rate.

### Pre-registered success criteria (written now, so we can't move goalposts)
- Span-level laundering rate ≤ conservative + 2 pts on the non-adaptive families.
- Label creep at k = 3 **at least 30% lower** than conservative.
- Utility retention ≥ 90% of no-defence on LongMemEval.
- MPBench ASR reduced by ≥ 80% relative.
- If the adaptive families (b, c, d, f) push laundering above 5%, the claim gets narrowed and reported as a limitation.

---

## 4. Design decisions and alternatives rejected

| Decision | Chosen | Rejected alternative | Why |
|---|---|---|---|
| How labels are assigned | From the channel | LLM classifies content as trusted/malicious (A-MemGuard, LLM judges) | Content detection is probabilistic and adaptive attacks beat it (FARMA beats A-MemGuard). Kept only as a baseline. |
| Label granularity | Sentence (claim-level as an ablation) | Record (TMA-NM, MemLineage); token | Record level is the baseline and causes the creep problem. Token-level attribution of LLM output is unreliable. Claim decomposition needs an extra LLM call per write, so it is only an ablation. |
| How attribution works | Embedding similarity + optional NLI; citations used only to add taint | Attention/gradient attribution | Attention needs open-weight internals, fails on API models and is noisy. Kept only as an optional ablation if time remains. |
| What happens when attribution fails | Fall back to the conservative label | Fall back to the source label most similar to the sentence | Errors must cause creep, not laundering. Security failures are asymmetric. |
| Store integrity | Content hash + HMAC | Signed Merkle log (MemLineage) | Already done and orthogonal; not this project's question. |
| Formal guarantee | Property tests (Hypothesis) + a written invariant for conservative mode | TLA+ model checking (TMA-NM) | Already done for the conservative rule. Not the bottleneck for an 8-week project. |
| Auditor method | Grouped → binary LOO → random-ablation surrogate | Attention scores; plain LOO only | Plain LOO misses split payloads. Attention isn't causal. |
| Storage | Own minimal store + adapters | Fork Mem0/Letta | Mem0's `add()` runs its own LLM extract/update internally, so I'd have to patch internals to intercept rewrites. Own store first, adapters afterwards. |
| Agent runtime for the eval | Own deterministic loop | LangGraph for the eval | Reproducibility and caching. A LangGraph adapter is still shipped for real use. |
| Embedder default | Hashing embedder (no deps) | sentence-transformers only | Tests run anywhere. ST is used for real eval runs. |
| Upgrade path | Explicit user event bound to a span id | Upgrade when the user "restates" a fact (text match) | Text-match upgrades can be socially engineered; binding to an event follows robust endorsement. |
| Benchmark ground truth | Fact tokens + ~200 human checks | Full human annotation | Annotating everything doesn't scale. Fact tokens are cheap, and the human check measures where the shortcut is wrong. |

---

## 5. Timeline (real dates)

| Week | Dates | Deliverable |
|---|---|---|
| 1 | 2–8 Nov | Read MemLineage, TMA-NM, MPBench, CaMeL, FIDES, A-MemGuard, ContextCite **in full**. Run MPBench's released code as-is. **This session:** repo skeleton, labels, records, store, conservative/span/partitioned propagation, policy, auditor, ScriptedLLM, tests, toy laundering demo. |
| 2 | 9–15 Nov | Real LLM backends + disk cache. Laundering generator v1 (families a, b, e). Pilot of 50 scenarios to measure tokens per episode and size the budget. |
| 3 | 16–22 Nov | Span propagation tuned (τ_taint, τ_clear, NLI on/off). Creep and laundering curves vs k. Families c, d, f. |
| 4 | 23–29 Nov | Lineage and partitioned baselines. Hand-annotate 200 sentences. **Go/no-go:** does span beat conservative on creep at equal laundering? If not, pivot the paper to "measuring creep/laundering in memory IFC" (still publishable as an analysis). |
| 5 | 30 Nov–6 Dec | Policy strict/balanced, confirmation burden, minimal provenance viewer. |
| 6 | 7–13 Dec | Auditor (grouped, LOO, surrogate). Split-payload tests. |
| 7 | 14–20 Dec | Full grid: MPBench subset, laundering set, LongMemEval subset × 9 configs × 3 backbones. |
| 8 | 21–27 Dec | pip package + LangGraph/Mem0 adapters. Write-up. |
| Buffer | 28 Dec–10 Jan | Holidays will eat part of weeks 7–8; plan on this buffer being used. |

**Venues:** check the actual deadlines yourself. I won't guess dates. Likely fits are a security venue's next cycle (USENIX Security / CCS / IEEE S&P), or the ML-security workshops at ICML 2027.

---

## 6. Risks (honest)

1. **Novelty overlap.** MemLineage and TMA-NM are close. Mitigation: frame against them explicitly, use them as baselines, and contact the authors early.
2. **Attribution can be gamed.** Paraphrase and split attacks may beat the similarity check, and that may simply be the finding.
3. **Budget.** A full grid can reach the billion-token range. The week 2 pilot fixes the real number; the plan is subsampling and self-hosting the open models.
4. **MPBench reproducibility.** Released code may not fit our agent loop, which may need adapter work. The paper also notes some domains use labeled context blocks, not real tool retrieval.
5. **Utility metric validity.** LongMemEval questions are mostly answers, not actions. Gated *actions* are measured on the laundering set's benign scenarios.

---

## 7. Related topics and reading list (beyond the brief)

- **IFC foundations:** Denning's lattice model (1976); Biba integrity (low-water-mark = our conservative rule; label creep is its known weakness); Myers & Liskov decentralized label model; robust declassification/endorsement (Zdancewic & Myers); TaintDroid.
- **Agent prompt-injection defences:** CaMeL, FIDES, Progent, RTBAS, spotlighting (Hines et al. 2024), instruction hierarchy (Wallace et al. 2024), StruQ/SecAlign; benchmarks AgentDojo, InjecAgent.
- **Memory/RAG attacks:** MINJA, AgentPoison, PoisonedRAG, eTAMP, Zombie Agents, FARMA, MPBench, EHR memory poisoning (2601.05504).
- **Memory defences:** A-MemGuard, MemLineage, TMA-NM, NeuroTaint, SENTINEL, AgentKernel.
- **Attribution:** ContextCite (ablation-based context attribution, the basis for the auditor), AIS/attribution evaluation, ALCE (citation generation), NLI-based factuality (SummaC), FActScore claim decomposition.
- **Data provenance:** where/why provenance (Buneman et al.), W3C PROV-DM. Useful vocabulary for the provenance UI.
- **Memory systems:** MemGPT/Letta, Mem0, Zep, Generative Agents' reflection (the original "reflection launders memory" mechanism), LongMemEval, LoCoMo.

---

## 8. What I build immediately after approval (this session)

1. `git init /home/user/taintmem`; `pyproject.toml`, which needs only numpy + pydantic (already installed); pytest if pip can reach the index, otherwise the stdlib `unittest`.
2. `labels.py`, `channels.py`, `record.py`, `embed.py` (HashingEmbedder), `store.py` (in-memory + SQLite).
3. `segment.py`, `attribution.py`, `propagate.py` (conservative, span, partitioned), `rewrite.py` with `ScriptedLLM`.
4. `policy.py` (strict/balanced), `auditor.py` (grouped + binary LOO), `lifecycle.py`.
5. `eval/laundering/generator.py` (families a, b, e with fact tokens) + `eval/metrics.py` (creep, laundering).
6. Tests:
   - lattice laws;
   - **conservative invariant** (no output span is ever more trusted than the least-trusted input in context);
   - **span safety** (attribution failure never upgrades a label);
   - policy decisions;
   - the auditor finds a single planted culprit.
7. `examples/demo_laundering.py`: plant an email payload, run 2 summarise passes, show conservative vs span labels and a policy BLOCK, run the auditor.
8. Copy this plan to `docs/PLAN.md`; commit locally (no remote push unless you give me one).

## Decisions from your answers (2 Nov)
- **Compute: none yet.** This session builds and tests only against `ScriptedLLM` plus a rule-based extractive summariser, so the propagation code runs on realistic text without an API.
  - **This is a hard blocker for week 2.** Without an API key or GPU by around 9 Nov, nothing in RQ2–RQ5 can be measured, and the 8-week timeline slips one-for-one.
  - Cheapest unblock: a hosted GPT-OSS-120B endpoint (OpenAI-compatible). It doubles as the MPBench-comparable backbone.
- **Goal: both paper and library.** This is the tightest schedule. If something has to give, cut the library polish (adapters, viewer) before the eval.
- **Repo: push to GitHub.** You didn't give an owner/repo name. I'll build in `/home/user/taintmem`, commit locally, and ask you for `owner/repo` before pushing. Branch: `taintmem-core`; no PR unless you ask.

## Verification
- `python -m pytest -q` (or `python -m unittest`) passes: lattice, invariant, propagation, policy and auditor tests.
- `python examples/demo_laundering.py` prints, per output sentence, its label under conservative vs span. It also prints creep/laundering numbers on the toy set, a BLOCK decision for the poisoned payment, and the auditor naming the planted memory id.
- No real LLM is called this session. The real-model eval starts in week 2, once you've confirmed which API keys and GPU access you have.
