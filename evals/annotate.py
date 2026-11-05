"""Blind annotation sheet to validate the fact-token ground truth against a human.

python -m evals.annotate make OUT_DIR [--n 200]    -> OUT_DIR/sheet.csv (give to annotator), OUT_DIR/key.json (keep)
python -m evals.annotate score OUT_DIR             -> agreement of filled sheet.csv vs key.json

Annotator question per sentence: "Does this sentence contain information that came from an UNTRUSTED record
(channel email/web/tool)?" Answer yes / no / unsure in the last column. The CSV carries no ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

from taintmem import ConservativePropagator, Integrity

from .laundering import FAMILIES, _flat, final_spans, generate

QUESTION = "contains_untrusted_info (yes/no/unsure)"


def make_sheet(out_dir: str, n: int = 200, seed: int = 2000, passes: int = 2, drift: float = 0.6, mix: bool = True) -> int:
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    rows, key = [], {}
    for sc in generate(max(n, 40), seed):
        inputs = [f"[{k}{'|' + d if d else ''}] {t}" for k, _, d, t in sc.records] + \
                 [f"[user] {t}" for t in sc.later_trusted[:max(passes - 1, 0)]]
        for sp in final_spans(sc, ConservativePropagator(), passes, drift, mix):
            f = _flat(sp.text)
            hit = [(t, lv) for t, lv in sc.tokens.items() if t.upper() in f]
            if hit:
                sid = f"s{len(rows):04d}"
                truth = min(lv for _, lv in hit)
                rows.append({"id": sid, "sentence": sp.text, "source_records": " || ".join(inputs), QUESTION: ""})
                key[sid] = {"truth": truth.name, "family": sc.family, "tokens": [t for t, _ in hit]}
    rng.shuffle(rows)
    rows = rows[:n]
    with open(os.path.join(out_dir, "sheet.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "sentence", "source_records", QUESTION]); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "key.json"), "w", encoding="utf-8") as f:
        json.dump({r["id"]: key[r["id"]] for r in rows}, f, indent=1)
    return len(rows)


def score(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "key.json"), encoding="utf-8") as kf:
        key = json.load(kf)
    with open(os.path.join(out_dir, "sheet.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    conf = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    bad, unsure, blank = [], 0, 0
    for r in rows:
        a = (r.get(QUESTION) or "").strip().lower()
        if not a:
            blank += 1; continue
        if a == "unsure":
            unsure += 1; continue
        if a not in ("yes", "no"):
            raise ValueError(f"{r['id']}: answer must be yes/no/unsure, got {a!r}")
        truth_untrusted = key[r["id"]]["truth"] == Integrity.UNTRUSTED.name
        said = a == "yes"
        conf[("t" if said == truth_untrusted else "f") + ("p" if said else "n")] += 1
        if said != truth_untrusted:
            bad.append({"id": r["id"], "sentence": r["sentence"], "truth": key[r["id"]]["truth"], "human": a,
                        "family": key[r["id"]]["family"]})
    n = sum(conf.values())
    return {"n_scored": n, "agreement": (conf["tp"] + conf["tn"]) / n if n else float("nan"), "confusion": conf,
            "unsure": unsure, "blank": blank, "disagreements": bad}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("make", "score")); ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "make":
        print(f"wrote {make_sheet(a.out_dir, a.n)} sentences to {a.out_dir}/sheet.csv (key.json: do NOT share with the annotator)")
    else:
        r = score(a.out_dir)
        print(f"scored {r['n_scored']}  agreement {r['agreement']:.3f}  confusion {r['confusion']}  unsure {r['unsure']}  blank {r['blank']}")
        for d in r["disagreements"][:15]:
            print(f"  {d['id']} [{d['family']}] truth={d['truth']} human={d['human']}: {d['sentence'][:100]}")


if __name__ == "__main__":
    main()
