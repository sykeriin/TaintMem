from __future__ import annotations

import random
from typing import Sequence

from taintmem.labels import Integrity


def creep_rate(pairs: Sequence[tuple]) -> float:
    """Of sentences whose true integrity is TRUSTED, fraction predicted lower."""
    t = [p for p in pairs if p[0] == Integrity.TRUSTED]
    return sum(1 for tr, pr in t if pr < tr) / len(t) if t else float("nan")


def laundering_rate(pairs: Sequence[tuple]) -> float:
    """Of sentences whose true integrity is UNTRUSTED, fraction predicted strictly higher."""
    u = [p for p in pairs if p[0] == Integrity.UNTRUSTED]
    return sum(1 for tr, pr in u if pr > tr) / len(u) if u else float("nan")


def bootstrap_ci(values: Sequence[float], n: int = 1000, seed: int = 0, alpha: float = 0.05):
    vals = list(values)
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]
