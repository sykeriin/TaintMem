"""Label propagation through LLM rewriting.

conservative : output = meet over ALL inputs in context                (TMA-NM-style baseline)
lineage      : one label per output record, from parent records
               whose attribution weight >= tau_edge                    (MemLineage-style baseline)
partitioned  : never mix integrity classes in one rewrite              (CaMeL-style quarantine)
span         : per-sentence label from attributed inputs (contribution)

Safety rule for `span`: uncertainty may only raise taint. A sentence is cleared of an untrusted input
only if it is confidently attributed (>= tau_clear) to trusted inputs and nothing untrusted reaches
tau_taint. Citations emitted by the LLM can add taint but never remove it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .attribution import Attributor
from .labels import Integrity, Label, meet
from .record import Span


@dataclass
class Assignment:
    text: str
    label: Label
    attributed_to: tuple = ()
    reason: str = ""


class ConservativePropagator:
    name = "conservative"
    partition = False

    def assign(self, sentences: Sequence[str], inputs: Sequence[Span],
               citations: Optional[Sequence[Sequence[str]]] = None) -> list[Assignment]:
        g = meet(s.label for s in inputs)
        return [Assignment(t, g, (), "meet-of-context") for t in sentences]


class PartitionedPropagator(ConservativePropagator):
    name = "partitioned"
    partition = True    # Rewriter runs one LLM call per integrity class


class SpanPropagator:
    name = "span"
    partition = False

    def __init__(self, attributor: Attributor | None = None, tau_taint: float = 0.30, tau_clear: float = 0.50,
                 anchor_dominates: bool = False):
        """anchor_dominates: if a sentence shares an exact identifier with some inputs, attribute it ONLY to
        those. Cuts boilerplate creep, but an attacker who copies a trusted identifier next to an instruction
        can then launder it (evals family g). Off by default for that reason."""
        assert tau_taint <= tau_clear
        self.attr = attributor or Attributor()
        self.tau_taint, self.tau_clear = tau_taint, tau_clear
        self.anchor_dominates = anchor_dominates

    def assign(self, sentences, inputs, citations=None):
        g = meet(s.label for s in inputs)
        by_id = {s.id: s for s in inputs}
        out = []
        for i, sent in enumerate(sentences):
            sims = self.attr.scores(sent, inputs)
            hit = [j for j, v in enumerate(sims) if v >= self.tau_taint]
            if self.anchor_dominates:
                anc = self.attr.anchored(sent, inputs)
                if anc.any():
                    hit = [j for j in hit if anc[j]]
            cited = [by_id[c] for c in (citations[i] if citations else ()) if c in by_id]
            if not hit and not cited:
                out.append(Assignment(sent, g, (), "novel->conservative"))
                continue
            cand = {inputs[j].id: inputs[j] for j in hit}
            cand.update({c.id: c for c in cited})
            label = meet(c.label for c in cand.values())
            reason = "attributed"
            if label.integrity > g.integrity:                      # would clear some untrusted input
                best = max((sims[j] for j in hit), default=0.0)
                if best < self.tau_clear:
                    label, reason = g, "weak-attribution->conservative"
                else:
                    reason = "cleared"
            out.append(Assignment(sent, label, tuple(cand), reason))
        return out


class LineagePropagator:
    name = "lineage"
    partition = False

    def __init__(self, attributor: Attributor | None = None, tau_edge: float = 0.25):
        self.attr = attributor or Attributor()
        self.tau_edge = tau_edge

    def assign(self, sentences, inputs, citations=None):
        g = meet(s.label for s in inputs)
        by_rec: dict[str, list[Span]] = {}
        for s in inputs:
            by_rec.setdefault(s.record_id, []).append(s)
        kept = []
        for rid, spans in by_rec.items():
            w = max((float(self.attr.scores(t, spans).max()) for t in sentences), default=0.0)
            if w >= self.tau_edge:
                kept.append(meet(s.label for s in spans))
        label = meet(kept) if kept else g
        return [Assignment(t, label, (), "record-lineage") for t in sentences]
