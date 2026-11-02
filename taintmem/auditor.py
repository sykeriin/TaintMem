"""Counterfactual auditor: which memories CAUSED this action? Re-run the decision with memories removed."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .policy import ToolCall
from .record import Span

DecideFn = Callable[[Sequence[Span]], Optional[ToolCall]]


@dataclass
class AuditReport:
    culprits: tuple          # span ids
    method: str              # group+bisect | group-joint | surrogate | none
    n_calls: int


class CounterfactualAuditor:
    def __init__(self, decide: DecideFn, seed: int = 0, n_samples: int = 64):
        self.decide, self.rng, self.n_samples = decide, random.Random(seed), n_samples
        self.n_calls = 0

    def _same(self, spans: Sequence[Span], target: tuple) -> bool:
        self.n_calls += 1
        c = self.decide(spans)
        return c is not None and c.key() == target

    def audit(self, call: ToolCall, context: Sequence[Span]) -> AuditReport:
        self.n_calls = 0
        target, ctx = call.key(), list(context)
        # 1. ablate whole source groups (domain / channel)
        groups: dict[str, list[Span]] = {}
        for s in ctx:
            groups.setdefault(min((x.group for x in s.label.sources), default="?"), []).append(s)
        causal = [g for g in groups.values() if not self._same([s for s in ctx if s not in g], target)]
        if causal:
            g = causal[0]
            # 2. bisect inside the group: O(log k) calls for a single culprit
            while len(g) > 1:
                a, b = g[:len(g) // 2], g[len(g) // 2:]
                if not self._same([s for s in ctx if s not in a], target):
                    g = a
                elif not self._same([s for s in ctx if s not in b], target):
                    g = b
                else:
                    return AuditReport(tuple(s.id for s in g), "group-joint", self.n_calls)
            return AuditReport((g[0].id,), "group+bisect", self.n_calls)
        # 3. payload spread across groups: random-ablation linear surrogate (ContextCite-style)
        n = len(ctx)
        if n == 0:
            return AuditReport((), "none", self.n_calls)
        X = np.array([[self.rng.random() < 0.5 for _ in range(n)] for _ in range(self.n_samples)], dtype=float)
        y = np.array([float(self._same([s for s, m in zip(ctx, row) if m], target)) for row in X])
        A = np.hstack([X, np.ones((len(X), 1))])
        w = np.linalg.lstsq(A, y, rcond=None)[0][:n]
        if w.max() <= 1e-6:
            return AuditReport((), "none", self.n_calls)
        return AuditReport(tuple(ctx[i].id for i in np.argsort(-w) if w[i] >= 0.5 * w.max()), "surrogate", self.n_calls)
