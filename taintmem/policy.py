"""Policy-gated actions. strict = context taint (also covers implicit flows); balanced = argument taint."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Sequence

from .attribution import identifiers
from .labels import Integrity
from .record import Span


class Level(IntEnum):
    READ = 0
    WRITE = 1      # consequential: send, delete, change settings
    CRITICAL = 2   # irreversible: pay, deploy, execute code


@dataclass(frozen=True)
class ToolSpec:
    name: str
    level: Level
    sensitive_args: tuple | None = None    # None = all args


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict = field(default_factory=dict, hash=False, compare=False)

    def key(self) -> tuple:
        return (self.name, tuple(sorted((k, _norm(v)) for k, v in self.args.items())))


def _norm(v: Any) -> str:
    return re.sub(r"[\s-]", "", str(v)).lower()


@dataclass
class Decision:
    action: str                 # ALLOW | CONFIRM | BLOCK
    reason: str
    depends_on: tuple = ()      # span ids
    tainted_args: tuple = ()


class Policy:
    def __init__(self, tools: Sequence[ToolSpec], mode: str = "balanced", min_integrity: Integrity = Integrity.TRUSTED):
        assert mode in ("strict", "balanced")
        self.tools = {t.name: t for t in tools}
        self.mode, self.min = mode, min_integrity
        self.log: list[tuple[ToolCall, Decision]] = []

    def _risky(self, level: Level, why: str, deps=(), args=()) -> Decision:
        return Decision("BLOCK" if level == Level.CRITICAL else "CONFIRM", why, tuple(deps), tuple(args))

    def _deps(self, value: Any, context: Sequence[Span]) -> list[Span]:
        v = _norm(value)
        ids = identifiers(str(value))
        hits = []
        for s in context:
            if (len(v) >= 3 and v in _norm(s.text)) or (ids and ids & identifiers(s.text)):
                hits.append(s)
        return hits

    def check(self, call: ToolCall, context: Sequence[Span]) -> Decision:
        spec = self.tools.get(call.name)
        if spec is None:
            d = Decision("BLOCK", "unknown tool (fail closed)")
        elif spec.level == Level.READ:
            d = Decision("ALLOW", "read-only")
        else:
            low = [s for s in context if s.label.integrity < self.min]
            if self.mode == "strict":
                d = (self._risky(spec.level, "untrusted memory in context", [s.id for s in low])
                     if low else Decision("ALLOW", "context fully trusted"))
            else:
                d = self._balanced(spec, call, context, low)
        self.log.append((call, d))
        return d

    def _balanced(self, spec: ToolSpec, call: ToolCall, context, low) -> Decision:
        names = spec.sensitive_args if spec.sensitive_args is not None else tuple(call.args)
        tainted, deps = [], []
        for a in names:
            if a not in call.args:
                continue
            hits = self._deps(call.args[a], context)
            if any(h.label.integrity >= self.min for h in hits):
                continue                                   # value is backed by a trusted span
            if hits:
                tainted.append(a); deps += [h.id for h in hits]
            elif low:                                      # unattributable value + untrusted context: fail closed
                tainted.append(a); deps += [s.id for s in low]
        if tainted:
            return self._risky(spec.level, "argument depends on untrusted memory", dict.fromkeys(deps), tainted)
        return Decision("ALLOW", "arguments backed by trusted memory")

    def confirmations_per_100(self, n_tasks: int) -> float:
        return 100.0 * sum(1 for _, d in self.log if d.action != "ALLOW") / max(n_tasks, 1)
