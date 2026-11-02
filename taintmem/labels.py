"""Label lattices. Integrity combines by MIN (Biba low-water-mark), confidentiality by MAX."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


class Integrity(IntEnum):
    UNTRUSTED = 0
    SEMI = 1
    TRUSTED = 2


class Confidentiality(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    SECRET = 2


@dataclass(frozen=True)
class Source:
    kind: str          # user | system | tool | web | email | file | agent
    id: str = ""
    domain: str = ""

    @property
    def group(self) -> str:
        """Key used by the auditor to ablate 'everything from one origin' at once."""
        return self.domain or f"{self.kind}:{self.id}"


@dataclass(frozen=True)
class Label:
    integrity: Integrity
    confidentiality: Confidentiality = Confidentiality.PUBLIC
    sources: frozenset = frozenset()
    agent_id: str = "agent0"
    endorsed_by: str | None = None   # set only by lifecycle.endorse (user-channel event)

    def tag(self) -> str:
        srcs = ",".join(sorted(s.group for s in self.sources)) or "-"
        mark = "*" if self.endorsed_by else ""
        return f"{self.integrity.name.lower()}{mark}|{srcs}"


TOP = Label(Integrity.TRUSTED)   # identity element of meet


def meet(labels: Iterable[Label]) -> Label:
    """Least-trusted combination. meet([]) is the identity (TRUSTED, PUBLIC)."""
    labels = list(labels)
    if not labels:
        return TOP
    agents = sorted({l.agent_id for l in labels})
    return Label(
        integrity=min(l.integrity for l in labels),
        confidentiality=max(l.confidentiality for l in labels),
        sources=frozenset().union(*(l.sources for l in labels)),
        agent_id="+".join(agents),
        endorsed_by=None,
    )
