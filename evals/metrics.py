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


def bootstrap_ratio(per_unit: Sequence[tuple], n: int = 500, seed: int = 0, alpha: float = 0.05):
    """CI for sum(num)/sum(den) resampling units (scenarios). per_unit = [(num, den), ...]."""
    units = [u for u in per_unit if u[1] > 0]
    if not units:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [rng.choice(units) for _ in units]
        den = sum(d for _, d in s)
        vals.append(sum(a for a, _ in s) / den if den else float("nan"))
    vals = sorted(v for v in vals if v == v)
    return vals[int(alpha / 2 * len(vals))], vals[int((1 - alpha / 2) * len(vals)) - 1]
