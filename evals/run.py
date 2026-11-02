"""python -m evals.run --n 200 --passes 1 2 3"""
from __future__ import annotations

import argparse

from .laundering import FAMILIES, generate, make_propagators, run_scenario
from .metrics import bootstrap_ci, creep_rate, laundering_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--mix", action="store_true")
    a = ap.parse_args()
    scs = generate(a.n, a.seed)
    props = make_propagators()
    print(f"drift={a.drift} mix={a.mix}")
    print(f"{a.n} scenarios, families={list(FAMILIES)}  (rule-based summariser: machinery smoke test only)\n")
    print(f"{'passes':>6} {'propagator':<13} {'creep':>7} {'creep 95% CI':>16} {'laundering':>11} {'n_trusted':>9} {'n_untrusted':>11}")
    for k in a.passes:
        for name, p in props.items():
            pairs, per_sc = [], []
            for sc in scs:
                r = run_scenario(sc, p, k, a.drift, a.mix)
                pairs += r
                per_sc.append(creep_rate(r))
            lo, hi = bootstrap_ci([x for x in per_sc if x == x])
            nt = sum(1 for t, _ in pairs if t == 2)
            nu = sum(1 for t, _ in pairs if t == 0)
            print(f"{k:>6} {name:<13} {creep_rate(pairs):>7.3f} {f'[{lo:.3f},{hi:.3f}]':>16} {laundering_rate(pairs):>11.3f} {nt:>9} {nu:>11}")
        print()


if __name__ == "__main__":
    main()
