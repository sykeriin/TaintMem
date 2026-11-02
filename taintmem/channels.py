"""Channel -> label. The label is decided by WHERE content arrived, never by what it says."""
from __future__ import annotations

from .labels import Confidentiality, Integrity, Label, Source

_DEFAULT = {
    "user": Integrity.TRUSTED,
    "system": Integrity.TRUSTED,
    "file": Integrity.SEMI,
    "agent": Integrity.SEMI,
    "web": Integrity.UNTRUSTED,
    "email": Integrity.UNTRUSTED,
    "tool": Integrity.UNTRUSTED,   # third-party tool output; allow-list to promote
}


class ChannelRegistry:
    def __init__(self, trusted_tools=(), semi_domains=(), defaults=None):
        self.defaults = dict(_DEFAULT if defaults is None else defaults)
        self.trusted_tools = set(trusted_tools)
        self.semi_domains = set(semi_domains)

    def label_for(self, kind: str, id: str = "", domain: str = "", agent_id: str = "agent0",
                  confidentiality: Confidentiality = Confidentiality.INTERNAL) -> Label:
        integ = self.defaults.get(kind, Integrity.UNTRUSTED)        # unknown channel: fail closed
        if kind == "tool" and id in self.trusted_tools:
            integ = Integrity.TRUSTED
        if kind in ("web", "email") and domain in self.semi_domains:
            integ = Integrity.SEMI                                  # never TRUSTED from the open internet
        return Label(integ, confidentiality, frozenset({Source(kind, id, domain)}), agent_id)
