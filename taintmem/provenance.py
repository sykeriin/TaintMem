"""'Why do you remember this, and from where?'"""
from __future__ import annotations

from .store import TaintMemStore


def lineage(store: TaintMemStore, span_id: str, depth: int = 8) -> dict:
    sp = store.span(span_id)
    node = {"span": sp.id, "text": sp.text, "label": sp.label.tag(), "op": store.records[sp.record_id].op, "from": []}
    rec = store.records[sp.record_id]
    srcs = rec.edges.get(sp.id, [])
    if depth > 0:
        for sid in srcs:                                        # sentence-level attribution edges
            try:
                node["from"].append(lineage(store, sid, depth - 1))
            except (KeyError, StopIteration):
                node["from"].append({"span": sid, "deleted": True})
        if not srcs:                                            # no edges: fall back to parent records
            for pid in rec.parent_ids:
                if pid in store.records:
                    node["from"].append({"record": pid, "label": store.records[pid].label.tag()})
                else:
                    node["from"].append({"record": pid, "deleted": True})
    return node


def explain(store: TaintMemStore, span_id: str) -> str:
    def walk(n, ind=0):
        pad = "  " * ind
        if n.get("deleted"):
            return f"{pad}- {n.get('span') or n.get('record')}  (original deleted)"
        if "record" in n:
            return f"{pad}- record {n['record']}  [{n['label']}]"
        s = f"{pad}- {n['span']} [{n['label']}] ({n['op']}) {n['text'][:70]}"
        return "\n".join([s] + [walk(c, ind + 1) for c in n["from"]])
    return walk(lineage(store, span_id))
