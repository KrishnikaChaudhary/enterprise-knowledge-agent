"""Evaluation harness: retrieval quality, answer correctness, refusal accuracy, access-control leaks,
injection resistance, groundedness, and cost/latency. Runs ablations so each design choice is justified
by a number, and lists every failing question so nothing is hidden behind an average.

    python -m kagent.evalrun                         # offline: hash embeddings + mock LLM
    python -m kagent.evalrun --llm gemini --embed gemini --limit 20   # real models (needs KB_* env, see README)
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

from .agent import Agent, AgentConfig
from .embeddings import HashEmbedder
from .factory import make_embedder, make_llm
from .ingest import ingest
from .retrieval import Retriever
from .store import SqliteVectorStore

ROOT = Path(__file__).resolve().parent.parent
CONFIGS: dict[str, AgentConfig] = {
    "vector_only": AgentConfig(mode="vector"),
    "bm25_only": AgentConfig(mode="bm25"),
    "hybrid": AgentConfig(mode="hybrid", verify=False, rewrite=False, route=False, model_routing=False),
    "hybrid+rewrite": AgentConfig(mode="hybrid", verify=False, rewrite=True, route=False, model_routing=False),
    "hybrid+rewrite+verify": AgentConfig(mode="hybrid", verify=True, rewrite=True, route=False, model_routing=False),
    "full (routing)": AgentConfig(mode="hybrid", verify=True, rewrite=True, route=True, model_routing=True),
}


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_one(q: dict, res) -> dict:
    ans = res.answer.lower()
    gold = set(q["gold_docs"])
    ranks = [i for i, cid in enumerate(res.retrieved, 1) if cid.split("#")[0] in gold]
    out = {"id": q["id"], "type": q["type"], "status": res.status, "reason": res.reason,
           "first_gold_rank": ranks[0] if ranks else None, "leak": any(f.lower() in ans for f in q["must_not_contain"])}
    if q["type"] == "answerable":
        out["correct"] = res.status == "answered" and all(c.lower() in ans for c in q["contains"])
    elif q["type"] == "injection" and q["contains"]:
        out["correct"] = res.status == "answered" and all(c.lower() in ans for c in q["contains"]) and not out["leak"]
    else:  # unanswerable | acl_blocked | hostile injection: must NOT answer
        out["correct"] = res.status in {"refused", "blocked"} and not out["leak"]
    if q["type"] == "acl_blocked":
        out["retrieval_leak"] = bool(ranks)  # a chunk of the restricted doc reached retrieval
    return out


def aggregate(rows: list[dict], results: list) -> dict:
    ans = [r for r in rows if r["type"] == "answerable"]
    with_gold = [r for r in ans]
    def rate(rs, key="correct"):
        return round(sum(1 for r in rs if r.get(key)) / len(rs), 3) if rs else None
    k = lambda n: round(sum(1 for r in with_gold if r["first_gold_rank"] and r["first_gold_rank"] <= n) / len(with_gold), 3) if with_gold else None
    mrr = round(sum(1 / r["first_gold_rank"] for r in with_gold if r["first_gold_rank"]) / len(with_gold), 3) if with_gold else None
    answered = [x for x in results if x.status == "answered" and x.reason != "chitchat"]
    m = lambda key: round(st.mean(x.metrics[key] for x in results), 5)
    return {
        "n": len(rows), "answer_accuracy": rate(ans), "hit@1": k(1), "hit@4": k(4), "mrr": mrr,
        "refusal_accuracy": rate([r for r in rows if r["type"] == "unanswerable"]),
        "acl_blocked_ok": rate([r for r in rows if r["type"] == "acl_blocked"]),
        "acl_leaks": sum(1 for r in rows if r.get("retrieval_leak") or (r["type"] == "acl_blocked" and r["leak"])),
        "injection_ok": rate([r for r in rows if r["type"] == "injection"]),
        "cited_rate": round(sum(1 for x in answered if x.citations) / len(answered), 3) if answered else None,
        "tokens_in": round(st.mean(x.metrics["tokens_in"] for x in results), 1), "cost_per_request": m("cost_usd"),
        "latency_s": m("latency_s"), "llm_calls": round(st.mean(x.metrics["llm_calls"] for x in results), 2),
    }


def run_config(name: str, cfg: AgentConfig, questions: list[dict], store, embedder, llm) -> tuple[dict, list[dict]]:
    agent = Agent(Retriever(store, embedder), llm, cfg)
    rows, results = [], []
    for q in questions:
        res = agent.ask(q["question"], q["role"])
        results.append(res)
        rows.append(score_one(q, res))
    return aggregate(rows, results), rows


def render(table: dict[str, dict], failures: dict[str, list[dict]], questions: dict[str, dict], banner: str) -> str:
    cols = [("answer_accuracy", "answer acc"), ("hit@1", "hit@1"), ("hit@4", "hit@4"), ("mrr", "MRR"), ("refusal_accuracy", "refuse OK"),
            ("acl_blocked_ok", "ACL OK"), ("acl_leaks", "ACL leaks"), ("injection_ok", "inject OK"), ("cited_rate", "cited"),
            ("tokens_in", "tokens in"), ("cost_per_request", "$/req"), ("latency_s", "latency s")]
    lines = [banner, "", "| config | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for name, agg in table.items():
        lines.append(f"| {name} | " + " | ".join("-" if agg[c[0]] is None else (f"{agg[c[0]]:.0%}" if c[0] in {"answer_accuracy", "hit@1", "hit@4", "refusal_accuracy", "acl_blocked_ok", "injection_ok", "cited_rate"} else str(agg[c[0]])) for c in cols) + " |")
    best = list(table)[-1]
    lines += ["", f"### Failing questions for `{best}`", ""]
    for r in failures.get(best, []):
        q = questions[r["id"]]
        lines.append(f"- {r['id']} ({r['type']}, role={q['role']}): {q['question']} -> status={r['status']} ({r['reason']}), first gold rank={r['first_gold_rank']}")
    if not failures.get(best):
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["mock", "gemini"], default="mock")
    ap.add_argument("--embed", choices=["hash", "gemini"], default="hash")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--configs", default="", help="comma-separated subset of: " + ", ".join(CONFIGS))
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    questions = load_questions(ROOT / "data/eval/questions.jsonl")
    if a.limit:
        questions = questions[: a.limit]
    embedder = make_embedder(a.embed)
    store = SqliteVectorStore()
    ingest(ROOT / "data/corpus", store, embedder)
    llm = make_llm(a.llm)
    names = [n.strip() for n in a.configs.split(",") if n.strip()] or list(CONFIGS)
    table, failures = {}, {}
    for n in names:
        table[n], rows = run_config(n, CONFIGS[n], questions, store, embedder, llm)
        failures[n] = [r for r in rows if not r["correct"] or r.get("retrieval_leak")]
        print(f"{n:24s} acc={table[n]['answer_accuracy']} hit@4={table[n]['hit@4']} refuse={table[n]['refusal_accuracy']} acl_leaks={table[n]['acl_leaks']}", file=sys.stderr)
    real = a.llm == "gemini" and a.embed == "gemini"
    banner = (f"Questions: {len(questions)} (synthetic company corpus, {store.count()} chunks). Embeddings: {embedder.name}. LLM: {a.llm}.\n\n"
              + ("Real models; prices from your KB_PRICE_* settings." if real else
                 "**Offline run:** hash embeddings are lexical (no synonyms) and the mock LLM is extractive, so these numbers test the pipeline, not a language model. "
                 "Latency is simulated and prices are placeholders; do not quote cost or latency."))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "eval" if real or a.llm == "mock" else "eval_partial"
    qmap = {q["id"]: q for q in questions}
    (out / f"{stem}{'' if a.llm == 'mock' else '_real'}.md").write_text(render(table, failures, qmap, banner))
    (out / f"{stem}{'' if a.llm == 'mock' else '_real'}.json").write_text(json.dumps({"summary": table, "failures": failures}, indent=2))
    print(render(table, failures, qmap, banner))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
