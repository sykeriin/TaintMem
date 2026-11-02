"""python -m evals.run --n 200 --seed 1000 --passes 3 [--drift 0.6 --mix] [--by-family]"""
from __future__ import annotations

import argparse

from .laundering import evaluate, make_propagators, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000, help="dev seeds are < 1000; report on >= 1000")
    ap.add_argument("--passes", type=int, nargs="+", default=[3])
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--mix", action="store_true")
    ap.add_argument("--by-family", action="store_true")
    a = ap.parse_args()
    from .laundering import generate
    scs = generate(a.n, a.seed)
    print(f"n={a.n} seed={a.seed} drift={a.drift} mix={a.mix}  (rule-based summariser: machinery test, not LLM evidence)")
    print(f"{'k':>2} {'propagator':<13}{'family':<24}{'creep':>7} {'usable':>7} {'launder':>8} {'synth':>6}")
    for k in a.passes:
        res = evaluate(scs, make_propagators(), k, a.drift, a.mix, a.by_family)
        for key, c in res.items():
            name, fam = key if a.by_family else (key, "ALL")
            m = summarize(c)
            print(f"{k:>2} {name:<13}{fam:<24}{m['creep']:>7.3f} {m['usable']:>7.3f} {m['laundering']:>8.3f} {m['synthesis']:>6.2f}")
        print()


if __name__ == "__main__":
    main()
