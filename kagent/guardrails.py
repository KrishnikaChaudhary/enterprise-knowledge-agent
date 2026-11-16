"""Safety rails: per-request Budget, prompt-injection scrubbing of retrieved text, secret/PII redaction.

Retrieved documents are UNTRUSTED input (anyone who can edit a wiki page can plant instructions in it).
Structural defences matter most: the model has no tools that act, retrieved text is fenced and labelled
as data, access control is enforced in the retriever (before any text reaches the model), and answers
must cite retrieved chunks. The regex scrubbing is a heuristic layer on top. See docs/SECURITY.md.
"""
from __future__ import annotations

import re

from .errors import BudgetExceeded


class Budget:
    def __init__(self, max_cost_usd: float, max_llm_calls: int, max_total_tokens: int):
        self.max_cost_usd, self.max_llm_calls, self.max_total_tokens = max_cost_usd, max_llm_calls, max_total_tokens
        self.calls = 0
        self.cost_usd = 0.0
        self.tokens = 0

    def before_call(self) -> None:
        if self.calls >= self.max_llm_calls:
            raise BudgetExceeded("llm_calls", self.max_llm_calls, self.calls)

    def after_call(self, result) -> None:
        self.calls += 1
        self.cost_usd += result.cost_usd
        self.tokens += result.tokens_in + result.tokens_out
        if self.cost_usd > self.max_cost_usd:
            raise BudgetExceeded("cost_usd", self.max_cost_usd, round(self.cost_usd, 6))
        if self.tokens > self.max_total_tokens:
            raise BudgetExceeded("total_tokens", self.max_total_tokens, self.tokens)


_INJECTION = [re.compile(p, re.I) for p in (
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)",
    r"\byou\s+are\s+now\b",
    r"\bsystem\s+prompt\b",
    r"\b(reveal|print|show|leak)\s+(your|the)\s+(instructions?|prompt|api[\s_-]?key|secrets?)\b",
    r"\bnew\s+instructions?\s*:",
    r"<\s*/?\s*(system|assistant|context|document)\s*>",
)]
REMOVED = "[line removed: instruction-like content in untrusted data]"


def sanitize(text: str) -> tuple[str, int]:
    """Replace lines that look like instructions aimed at the model. Returns (clean_text, lines_removed)."""
    out, removed = [], 0
    for line in text.splitlines():
        if any(p.search(line) for p in _INJECTION):
            out.append(REMOVED)
            removed += 1
        else:
            out.append(line)
    return "\n".join(out), removed


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION)


_REDACT = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{10,}"), "sk-[REDACTED]"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}", re.I), r"\1[REDACTED]"),
    (re.compile(r"((?:x-api-key|x-goog-api-key|api[_-]?key|token)[\"'\s:=]+)[A-Za-z0-9._\-]{8,}", re.I), r"\1[REDACTED]"),
]
_PII = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_ID]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[REDACTED_NUMBER]"),
]


def redact_secrets(text: str) -> str:
    for pat, repl in _REDACT:
        text = pat.sub(repl, text)
    return text


def redact_output(text: str) -> str:
    """Applied to every answer: credentials and obvious ID/card numbers never leave the service."""
    text = redact_secrets(text)
    for pat, repl in _PII:
        text = pat.sub(repl, text)
    return text
