"""Tokenisation shared by the embedder, BM25, and the support/groundedness checks."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9][a-z0-9_.\-]*[a-z0-9]|[a-z0-9]")
STOPWORDS = frozenset(
    "a an and are as at be but by can could do does did for from had has have how i if in into is it its of on or "
    "our should so than that the their them then there these they this to was we were what when where which who whom "
    "why will with would you your me my us about any per".split()
)
_SUFFIXES = ("ations", "ation", "ingly", "ings", "ing", "edly", "ed", "ies", "es", "s")


def stem(tok: str) -> str:
    """Tiny suffix stripper (not Porter): good enough to match 'reimbursed'/'reimbursement'-style variants."""
    if len(tok) <= 3 or tok.isdigit():
        return tok
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            base = tok[: -len(suf)]
            return base + "y" if suf == "ies" else base
    return tok


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    out = []
    for m in _WORD.findall(text.lower().replace("_", " ")):
        for part in re.split(r"[.\-]", m) if not m.replace(".", "").replace("-", "").isdigit() else [m]:
            if part and (keep_stop or part not in STOPWORDS) and (len(part) > 1 or part.isdigit()):  # drops possessive 's' etc.
                out.append(stem(part))
    return out


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\-*])|\n+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]
