"""A deterministic, deliberately naive payment agent (recency-biased). A stand-in for an LLM decision."""
from __future__ import annotations

from typing import Optional, Sequence

from ..attribution import identifiers
from ..policy import ToolCall
from ..record import Span


def toy_payment_agent(invoice: str):
    def decide(context: Sequence[Span]) -> Optional[ToolCall]:
        chosen = None
        for s in context:                       # later memories win
            for i in sorted(identifiers(s.text)):
                if i.startswith(("GB", "DE")) and len(i) >= 15:
                    chosen = i
        return ToolCall("pay_invoice", {"invoice": invoice, "iban": chosen}) if chosen else None
    return decide
