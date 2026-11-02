"""Expiry, quarantine, purge and explicit endorsement (the only way an untrusted span becomes trusted)."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .labels import Integrity
from .store import TaintMemStore


@dataclass(frozen=True)
class UserEvent:
    """Constructed ONLY by the host UI layer after authenticating a human action. Never from model text."""
    user_id: str
    authenticated: bool
    span_id: str
    action: str = "confirm"


def expire(store: TaintMemStore) -> list[str]:
    now = store.clock()
    gone = [r.id for r in store.records.values()
            if r.expires_at is not None and r.expires_at <= now
            and not any(s.label.endorsed_by for s in r.spans)]
    for rid in gone:
        store.delete(rid)
    return gone


def purge_domain(store: TaintMemStore, domain: str) -> list[str]:
    gone = [r.id for r in store.records.values()
            if any(src.domain == domain for s in r.spans for src in s.label.sources)]
    for rid in gone:
        store.delete(rid)
    return gone


def endorse(store: TaintMemStore, event: UserEvent) -> bool:
    """Upgrade one span to TRUSTED. Bound to a specific span id and an authenticated user event."""
    if not (isinstance(event, UserEvent) and event.authenticated and event.action == "confirm"):
        return False
    span = store.span(event.span_id)
    store.replace_span(span.id, replace(span.label, integrity=Integrity.TRUSTED, endorsed_by=event.user_id))
    store.audit_log.append({"t": store.clock(), "event": "endorse", "span": span.id, "user": event.user_id,
                            "was": span.label.integrity.name})
    return True
