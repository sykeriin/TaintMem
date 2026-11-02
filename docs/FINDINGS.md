# Findings so far (2 Nov 2026, week 1)

**Status: machinery works; no research claim is supported yet.** Everything below uses a rule-based
stand-in for an LLM (`RuleSummarizer`) and a benchmark I wrote myself (`evals/laundering.py`). No real model
has been run (no API key / GPU yet).

## What the toy benchmark shows (100 scenarios, `python -m evals.run`)

| regime | conservative | lineage | partitioned | span |
|---|---|---|---|---|
| easy (no drift, no mixing): creep @ k=1..3 | 1.00 | 1.00 | 0.00* | 0.00 |
| easy: laundering | 0.00 | 0.00 | 0.00 | 0.00 |
| hard (`--drift 0.6 --mix`): creep @ k=1 | 1.00 | 1.00 | 0.00* | 0.00 |
| hard: creep @ k=3 | 1.00 | 1.00 | 0.00* | **1.00** |
| hard: laundering @ k=1,3 | 0.00 | 0.00 | 0.00 | 0.00 |

\* Not comparable. `partitioned` changes the summariser's *inputs* (one call per integrity class), so its
sentences, and hence the denominator, differ. It trivially avoids creep by never synthesising across
sources, a utility cost this benchmark does not measure.

## Interpretation (what I do and do not conclude)
1. **Easy regime is uninformative.** Every scenario contains one untrusted email, so conservative taints
   100% of trusted facts, while span keeps them only because each fact has an exact identifier that anchors
   attribution. That identifier shortcut is a property of my benchmark, not evidence about real text.
2. **Hard regime exposes the real failure mode.** At k=3 span creep is 1.00: a trusted sentence anchored
   to its trusted source by an exact identifier still shares boilerplate ("The user's ...") with untrusted
   spans, crosses `tau_taint=0.30`, and the meet pulls it down. So today span is *no better than
   conservative* once lexical drift + mixing are on. The hypothesis behind RQ1 is **untested, not confirmed**.
   I did not tune `tau_taint` against my own benchmark; that needs a real encoder (IDF-weighted or
   sentence-transformer) and a held-out set first.
3. **Zero laundering is not a result.** The toy summariser only obeys one injection template and always
   leaves the identifier in the output. Adaptive families (c split payload, f forged citations, real
   paraphrase) are not built yet.
4. **Open design question surfaced:** should an exact identifier anchor to a trusted span *dominate*
   weaker lexical hits? That would fix this creep but lets an attacker who copies a trusted identifier
   launder an adjacent instruction. Needs a test in the adaptive set before deciding.

## Observed qualitative benefit
In `examples/demo_laundering.py`, conservative labels the poisoned sentence with sources
`airline.example, evil.example, user` (provenance smeared), while span reports only `evil.example`, and
keeps the legitimate vendor IBAN trusted (so the legitimate payment would not be gated).

## Next
Real LLM backend + disk cache; IDF/real-encoder attribution; adaptive families c/f; measure utility cost of
`partitioned`; fixed denominators (score all methods on the same ground-truth fact tokens, not on their own
output sentences).
