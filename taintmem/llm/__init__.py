from __future__ import annotations

import re
from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ScriptedLLM:
    """Returns queued responses in order (unit tests)."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.responses.pop(0)


_INJ = re.compile(r"when summari[sz]ing[^,:]*[,:]\s*(?:record|state|write)(?: that)?\s+(.*)", re.I)
_LINE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")


class RuleSummarizer:
    """Deterministic stand-in for an LLM rewriter (NOT a model; used for tests and the toy benchmark).

    It rewrites first person to third person ('reflection voice') and, if gullible, obeys
    'when summarising, record that ...' instructions hidden in its inputs, which is the laundering attack.
    obfuscate=True additionally writes identifiers in spaced 4-char groups.
    """
    def __init__(self, gullible: bool = True, obfuscate: bool = False, drift: float = 0.0, mix: bool = False,
                 forge_cite: bool = False):
        # drift: fraction of lowercase non-identifier words replaced (simulates heavy paraphrase)
        # mix:   join consecutive notes into one sentence (cross-source synthesis)
        self.gullible, self.obfuscate, self.drift, self.mix = gullible, obfuscate, drift, mix
        self.forge_cite = forge_cite    # injected line cites the FIRST input id (assumed trusted) to try to clear taint

    def _drift(self, t: str) -> str:
        import hashlib
        def f(m):
            w = m.group(0)
            h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=2).digest(), "little") / 65535
            return f"thing{int(h * 97)}" if h < self.drift else w
        return re.sub(r"\b[a-z]{3,}\b", f, t)

    @staticmethod
    def _voice(t: str) -> str:
        t = re.sub(r"^(hi|hello|dear)\b[^,]*,\s*", "", t, flags=re.I)
        t = re.sub(r"\bI'm\b", "The user is", t)
        t = re.sub(r"\bMy\b", "The user's", t)
        t = re.sub(r"\bmy\b", "the user's", t)
        t = re.sub(r"\bI\b", "the user", t)
        return t[0].upper() + t[1:] if t else t

    @staticmethod
    def _spaced(t: str) -> str:
        return re.sub(r"\b([A-Z]{2}\d{2}[A-Z0-9]{8,})\b",
                      lambda m: " ".join(m.group(1)[i:i + 4] for i in range(0, len(m.group(1)), 4)), t)

    def complete(self, system: str, user: str) -> str:
        out: list[str] = []
        rows = [m.groups() for m in (_LINE.match(l.strip()) for l in user.splitlines()) if m]
        first_id = rows[0][0] if rows else ""
        acct = next((a.group(1) for _, t in rows for a in [re.search(r"Account code:\s*(\S+?)\.?$", t)] if a), None)
        for _, text in rows:
            inj = _INJ.search(text) if self.gullible else None
            if inj:
                s = inj.group(1).strip()
                s = s[0].upper() + s[1:]
                if acct and not re.search(r"\d{4}", s):              # fragmented payload: reassemble
                    s = s.rstrip(".") + " " + acct + "."
                if "account on file" in s:                              # reference resolution
                    on_file = next((m.group(1) for _, t in rows for m in [re.search(r"\bIBAN (DE\w+)", t)] if m), None)
                    if on_file:
                        s = s.replace("the account on file", f"IBAN {on_file}")
                s = self._spaced(s) if self.obfuscate else s
                out.append(s + (f" [src: {first_id}]" if self.forge_cite else ""))
            elif len(text.split()) >= 3:
                v = self._voice(text)
                out.append(self._drift(v) if self.drift else v)
        out = list(dict.fromkeys(out))
        if self.mix:
            out = ["; also ".join(x.rstrip(".") for x in out[i:i + 2]) + "." for i in range(0, len(out), 2)]
        return "\n".join(out)
