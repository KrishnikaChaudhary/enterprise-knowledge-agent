"""Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion (RRF).

Why hybrid: dense vectors catch paraphrases, BM25 catches exact identifiers, codes and numbers that
embeddings blur. RRF fuses ranks, so the two score scales never need calibrating.
Access control is applied before scoring (see store.can_see). `support()` gives a lexical grounding
score in [0,1] used to refuse questions the corpus cannot answer.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .chunking import Chunk
from .embeddings import Embedder
from .store import SqliteVectorStore
from .textutil import tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    support: float = 0.0


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokens(c.embed_text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = tokens(query)
        out = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                out.append((self.chunks[i], s))
        out.sort(key=lambda t: -t[1])
        return out[:k]


def support(question: str, chunk: Chunk, idf: dict[str, float] | None = None) -> float:
    """IDF-weighted share of the question's content terms found in the chunk."""
    q = set(tokens(question))
    if not q:
        return 0.0
    c = set(tokens(chunk.embed_text))
    w = (lambda t: (idf or {}).get(t, 1.0))
    return sum(w(t) for t in q if t in c) / sum(w(t) for t in q)


class Retriever:
    def __init__(self, store: SqliteVectorStore, embedder: Embedder, mode: str = "hybrid", rrf_k: int = 60, pool: int = 30):
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError("mode must be vector, bm25 or hybrid")
        self.store, self.embedder, self.mode, self.rrf_k, self.pool = store, embedder, mode, rrf_k, pool
        self._bm25: dict[str, tuple[int, BM25]] = {}

    def _bm25_for(self, role: str) -> BM25:
        n = self.store.count()
        cached = self._bm25.get(role)
        if cached and cached[0] == n:
            return cached[1]
        idx = BM25(self.store.visible_chunks(role))
        self._bm25[role] = (n, idx)
        return idx

    def retrieve(self, query: str, role: str = "employee", k: int = 4) -> list[Hit]:
        role = (role or "employee").lower()
        vec_hits: list[tuple[Chunk, float]] = []
        bm_hits: list[tuple[Chunk, float]] = []
        if self.mode in {"vector", "hybrid"}:
            qv = self.embedder.embed([query], "query")[0]
            vec_hits = self.store.search(qv, self.pool, role)
        if self.mode in {"bm25", "hybrid"}:
            bm_hits = self._bm25_for(role).search(query, self.pool)
        fused: dict[str, Hit] = {}
        for rank, (c, _s) in enumerate(vec_hits, 1):
            h = fused.setdefault(c.id, Hit(c, 0.0))
            h.vector_rank = rank
            h.score += 1.0 / (self.rrf_k + rank)
        for rank, (c, _s) in enumerate(bm_hits, 1):
            h = fused.setdefault(c.id, Hit(c, 0.0))
            h.bm25_rank = rank
            h.score += 1.0 / (self.rrf_k + rank)
        hits = sorted(fused.values(), key=lambda h: -h.score)[:k]
        idf = self._bm25_for(role).idf if hits else None
        for h in hits:
            h.support = support(query, h.chunk, idf)
        return hits
