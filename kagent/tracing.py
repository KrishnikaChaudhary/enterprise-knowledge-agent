"""LLM-native observability: one trace per request, one span per agent step / LLM call / tool call.

Tracks tokens in/out, cost-per-request, tokens/sec, time-to-first-token and latency.
Spans carry trace_id / span_id / parent_id, so the JSON export maps 1:1 onto OpenTelemetry
(swap Tracer.export for an OTLP / Cloud Trace exporter when the SDK is available).
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

# USD per 1M tokens (input, output). ILLUSTRATIVE PLACEHOLDERS: replace with your provider's real prices.
PRICING = {"small": (0.15, 0.60), "large": (3.00, 15.00), "embed": (0.10, 0.0)}


def count_tokens(text: str) -> int:
    """Cheap approximation (~1.3 tokens/word). Replace with the provider's tokenizer / usage field."""
    return max(1, round(len(text.split()) * 1.3))


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICING[model]
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    kind: str  # request | agent | llm | tool | embed | retrieve
    attrs: dict = field(default_factory=dict)
    start: float = 0.0
    wall_s: float = 0.0
    latency_s: float = 0.0  # llm: model latency (simulated by MockLLM, measured otherwise); tool: tool latency
    tokens_in: int = 0
    tokens_out: int = 0
    ttft_s: float = 0.0
    cost_usd: float = 0.0
    status: str = "ok"  # ok | error
    error: str | None = None


class Tracer:
    def __init__(self, name: str):
        self.name = name
        self.trace_id = uuid.uuid4().hex
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, kind: str = "agent", **attrs):
        parent = self._stack[-1].span_id if self._stack else None
        sp = Span(uuid.uuid4().hex[:8], parent, name, kind, dict(attrs), start=time.perf_counter())
        self.spans.append(sp)
        self._stack.append(sp)
        try:
            yield sp
        except BaseException as exc:
            sp.status, sp.error = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.wall_s = time.perf_counter() - sp.start
            self._stack.pop()

    def summary(self) -> dict:
        llm = [s for s in self.spans if s.kind == "llm"]
        tools = [s for s in self.spans if s.kind == "tool"]
        tin = sum(s.tokens_in for s in llm)
        tout = sum(s.tokens_out for s in llm)
        llm_time = sum(s.latency_s for s in llm)
        per_agent: dict[str, dict] = {}
        for s in llm:
            a = per_agent.setdefault(s.attrs.get("agent", "?"), {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
            a["calls"] += 1
            a["tokens_in"] += s.tokens_in
            a["tokens_out"] += s.tokens_out
            a["cost_usd"] += s.cost_usd
        return {
            "trace_id": self.trace_id,
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "tool_cache_hits": sum(1 for s in tools if s.attrs.get("cache_hit")),
            "span_errors": sum(1 for s in self.spans if s.status == "error"),
            "injection_lines_removed": sum(s.attrs.get("injection_lines_removed", 0) for s in tools),
            "tokens_in": tin,
            "tokens_out": tout,
            "total_tokens": tin + tout,
            "cost_usd": sum(s.cost_usd for s in llm),
            "tokens_per_sec": (tout / llm_time) if llm_time else 0.0,
            "avg_ttft_s": (sum(s.ttft_s for s in llm) / len(llm)) if llm else 0.0,
            "latency_s": llm_time + sum(s.latency_s for s in tools),  # sequential execution
            "per_agent": per_agent,
        }

    def payload(self) -> dict:
        return {"trace_id": self.trace_id, "name": self.name, "summary": self.summary(), "spans": [asdict(s) for s in self.spans]}

    def export(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload(), indent=2))
