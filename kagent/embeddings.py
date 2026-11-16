"""Embedding backends.

* HashEmbedder  : deterministic, offline, dependency-free feature hashing over stems and bigrams.
                  It captures LEXICAL similarity only (no synonyms), so it is a test double that exercises
                  the whole pipeline, not a stand-in for a real embedding model.
* (kagent.gemini) GeminiEmbedder: Gemini embedding models over the Gemini API or Vertex AI.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

from .textutil import tokens


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]: ...


def l2_normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are stored L2-normalised


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> list[float]:
        toks = tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 40) & 1 else -1.0
        return l2_normalize(v)

    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        return [self._vec(t) for t in texts]
