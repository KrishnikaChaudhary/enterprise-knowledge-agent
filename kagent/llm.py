"""LLM interface plus a deterministic offline MockLLM.

MockLLM is extractive: it answers by selecting sentences from the supplied context, and its verifier
checks lexical support. Token counts come from real prompt sizes (approximate tokenizer); latency,
TTFT and tokens/sec are SIMULATED and prices are placeholders, so cost/latency figures from the mock
are not quotable. Quality metrics measure the retrieval + grounding pipeline, not a language model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .textutil import STOPWORDS, sentences, tokens
from .tracing import cost_usd, count_tokens


@dataclass
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int
    model: str  # tier: small | large
    ttft_s: float
    latency_s: float
    cost_usd: float


class LLMClient(Protocol):
    def complete(self, role: str, prompt: str, model: str) -> LLMResult: ...


NOT_FOUND = "NOT_FOUND"
_SPEED = {"small": (0.15, 160.0), "large": (0.40, 55.0)}  # (ttft s, tokens/s): simulated
_CITE = re.compile(r"\[(\d+)\]")
GREETING = frozenset("hi hello hey thanks thank bye goodbye".split())


def parse_context(prompt: str) -> dict[int, str]:
    """Context blocks look like '[n] (title > section) text...' separated by blank lines, inside <context> tags."""
    found = re.findall(r"<context>\n(.*?)\n</context>", prompt, re.S)  # last block wins: instructions may mention the tag
    out: dict[int, str] = {}
    for block in re.split(r"\n\s*\n", found[-1].strip() if found else ""):
        mm = re.match(r"\[(\d+)\]\s*(.*)", block.strip(), re.S)
        if mm:
            out[int(mm.group(1))] = mm.group(2)
    return out


def _question(prompt: str) -> str:
    found = re.findall(r"<question>(.*?)</question>", prompt, re.S)
    return found[-1].strip() if found else ""


class MockLLM:
    def __init__(self, min_overlap: float = 0.34):
        self.min_overlap = min_overlap

    def complete(self, role: str, prompt: str, model: str) -> LLMResult:
        text = getattr(self, f"_{role}", self._default)(prompt)
        tin, tout = count_tokens(prompt), count_tokens(text)
        ttft, tps = _SPEED[model]
        return LLMResult(text, tin, tout, model, ttft, ttft + tout / tps, cost_usd(model, tin, tout))

    def _default(self, prompt: str) -> str:
        return ""

    def _router(self, prompt: str) -> str:
        q = _question(prompt).lower()
        toks = set(re.findall(r"[a-z]+", q))
        if toks and toks <= GREETING | STOPWORDS:
            return "ROUTE: chitchat"
        if re.search(r"\b(weather|joke|poem|lottery|stock price|capital of)\b", q):
            return "ROUTE: out_of_scope"
        return "ROUTE: kb"

    def _rewrite(self, prompt: str) -> str:
        return " ".join(t for t in re.findall(r"[A-Za-z0-9\-]+", _question(prompt)) if t.lower() not in STOPWORDS)

    def _answer(self, prompt: str) -> str:
        qt = set(tokens(_question(prompt)))
        if not qt:
            return NOT_FOUND
        scored = []
        for n, block in parse_context(prompt).items():
            body = re.sub(r"^\([^)]*\)\s*", "", block)
            for sent in sentences(body):
                st = set(tokens(sent))
                ov = len(qt & st) / len(qt)
                if ov:
                    scored.append((ov, n, sent))
        scored.sort(key=lambda t: -t[0])
        if not scored or scored[0][0] < self.min_overlap:
            return NOT_FOUND
        best = scored[0][0]
        picked = [s for s in scored if s[0] >= max(self.min_overlap, best * 0.8)][:2]
        return " ".join(f"{sent.rstrip('.')} [{n}]." for _ov, n, sent in picked)

    def _verify(self, prompt: str) -> str:
        found = re.findall(r"<answer>(.*?)</answer>", prompt, re.S)
        ctx = parse_context(prompt)
        bad = []
        for sent in sentences(found[-1] if found else ""):
            cited = [int(x) for x in _CITE.findall(sent)]
            body = set(tokens(_CITE.sub("", sent)))
            if not cited or not body:
                bad.append(sent)
                continue
            have = set().union(*(set(tokens(ctx.get(n, ""))) for n in cited))
            if len(body & have) / len(body) < 0.7:
                bad.append(sent)
        return "VERDICT: PASS" if not bad else f"VERDICT: FAIL - unsupported: {bad[0][:120]}"
