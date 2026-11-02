"""Static HTML provenance viewer (no JS, no external assets; all text HTML-escaped)."""
from __future__ import annotations

import html

from .labels import Integrity
from .store import TaintMemStore

_COL = {Integrity.TRUSTED: "#1a7f37", Integrity.SEMI: "#9a6700", Integrity.UNTRUSTED: "#cf222e"}


def render_html(store: TaintMemStore) -> str:
    rows = []
    for r in sorted(store.records.values(), key=lambda r: r.created_at):
        for s in r.spans:
            srcs = ", ".join(sorted(x.group for x in s.label.sources)) or "-"
            rows.append(f"<tr><td>{html.escape(s.id)}</td>"
                        f"<td style='color:{_COL[s.label.integrity]};font-weight:600'>{s.label.integrity.name}"
                        f"{' (endorsed by ' + html.escape(s.label.endorsed_by) + ')' if s.label.endorsed_by else ''}</td>"
                        f"<td>{html.escape(srcs)}</td><td>{html.escape(r.op)}</td><td>{html.escape(s.text)}</td></tr>")
    return ("<!doctype html><meta charset=utf-8><title>TaintMem provenance</title>"
            "<style>body{font:14px system-ui;margin:2rem}td,th{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left}</style>"
            "<h1>Memory provenance</h1><table><tr><th>span</th><th>integrity</th><th>source</th><th>op</th><th>text</th></tr>"
            + "".join(rows) + "</table>")
