"""kagent ask "question" [--role hr] [--trace out.json] ; kagent ingest"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import AgentConfig
from .errors import KagentError
from .factory import build
from .settings import Settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kagent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--role", default="employee")
    ask.add_argument("--trace", help="write the span trace to this JSON file")
    ask.add_argument("--json", action="store_true")
    sub.add_parser("ingest", help="(re)index the corpus into the SQLite store")
    a = ap.parse_args(argv)
    try:
        agent, store, report = build(Settings.from_env({**__import__("os").environ, "KB_HOST": "127.0.0.1"}), AgentConfig())
    except KagentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if a.cmd == "ingest":
        print(json.dumps(report.__dict__, indent=2))
        return 0
    res = agent.ask(a.question, a.role)
    if a.trace:
        with open(a.trace, "w") as f:
            json.dump(res.trace, f, indent=2)
    if a.json:
        print(json.dumps(res.as_dict(), indent=2))
        return 0
    print(f"[{res.status}] {res.answer}")
    for c in res.citations:
        print(f"  [{c.n}] {c.doc_id} > {c.section}")
    m = res.metrics
    print(f"Metrics: {m['llm_calls']} LLM calls, {m['tokens_in']}+{m['tokens_out']} tokens, ${m['cost_usd']:.5f}, verified={res.verified}")
    return 0 if res.status in {"answered", "refused", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
