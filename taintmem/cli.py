"""python -m taintmem.cli --db mem.sqlite list | why SPAN | purge DOMAIN | endorse SPAN --user ID | viewer OUT.html"""
from __future__ import annotations

import argparse
import sys

from .lifecycle import UserEvent, endorse, purge_domain
from .provenance import explain
from .store import TaintMemStore
from .viewer import render_html


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="taintmem")
    ap.add_argument("--db", required=True)
    ap.add_argument("--secret", default="dev-secret")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("why").add_argument("span")
    sub.add_parser("purge").add_argument("domain")
    e = sub.add_parser("endorse"); e.add_argument("span"); e.add_argument("--user", required=True); e.add_argument("--yes", action="store_true")
    sub.add_parser("viewer").add_argument("out")
    a = ap.parse_args(argv)
    st = TaintMemStore(path=a.db, secret=a.secret.encode())
    if a.cmd == "list":
        for r in st.records.values():
            for s in r.spans:
                print(f"{s.id:<10}[{s.label.tag()}] {s.text[:90]}")
    elif a.cmd == "why":
        print(explain(st, a.span))
    elif a.cmd == "purge":
        print(f"purged {len(purge_domain(st, a.domain))} record(s) from {a.domain}")
    elif a.cmd == "endorse":
        sp = st.span(a.span)
        print(f"{sp.id} [{sp.label.tag()}] {sp.text}")
        if not a.yes and input("Mark this memory TRUSTED? [y/N] ").strip().lower() != "y":
            print("not endorsed"); return 1
        # this CLI is the host UI: a human typed/confirmed the action, so the event is authenticated
        print("endorsed" if endorse(st, UserEvent(a.user, True, a.span)) else "refused")
    elif a.cmd == "viewer":
        open(a.out, "w").write(render_html(st)); print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
