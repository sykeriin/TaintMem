from __future__ import annotations

import re

_ABBR = ("dr.", "mr.", "mrs.", "ms.", "e.g.", "i.e.", "vs.", "no.", "st.", "inc.", "ltd.")
_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\[(])|\n+')


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT.split(text) if p and p.strip()]
    out: list[str] = []
    for p in parts:
        if out and out[-1].lower().endswith(_ABBR):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out
