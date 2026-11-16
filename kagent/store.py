"""Vector store on SQLite (standard library only).

Exact cosine search over L2-normalised float32 vectors, with the access-control filter applied BEFORE
scoring, so a chunk a caller may not see is never scored, ranked, or sent to the model.
Exact search is right up to ~10^5 chunks; beyond that use the `VectorStore` protocol to plug in
Vertex AI Vector Search / AlloyDB pgvector (see docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import array
import json
import sqlite3
import threading
from typing import Iterable, Protocol

from .chunking import Chunk
from .embeddings import cosine


def can_see(acl: Iterable[str], role: str) -> bool:
    acl = set(acl)
    return "all" in acl or role == "admin" or role in acl


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], embed_model: str) -> None: ...
    def search(self, qvec: list[float], k: int, role: str) -> list[tuple[Chunk, float]]: ...
    def visible_chunks(self, role: str) -> list[Chunk]: ...
    def count(self) -> int: ...


class SqliteVectorStore:
    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, title TEXT, section TEXT, "
                "text TEXT, acl TEXT, kind TEXT, meta TEXT, content_hash TEXT, embed_model TEXT, vec BLOB)"
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_doc ON chunks(doc_id)")
            self._db.commit()

    # ---- writes ----
    def existing_hashes(self) -> dict[str, tuple[str, str]]:
        with self._lock:
            return {r[0]: (r[1], r[2]) for r in self._db.execute("SELECT id, content_hash, embed_model FROM chunks")}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], embed_model: str) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        rows = [(c.id, c.doc_id, c.title, c.section, c.text, ",".join(c.acl), c.kind, json.dumps(c.meta), c.content_hash,
                 embed_model, array.array("f", v).tobytes()) for c, v in zip(chunks, vectors)]
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            self._db.commit()

    def delete_missing(self, doc_id: str, keep_ids: set[str]) -> int:
        with self._lock:
            ids = [r[0] for r in self._db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
            gone = [i for i in ids if i not in keep_ids]
            self._db.executemany("DELETE FROM chunks WHERE id=?", [(i,) for i in gone])
            self._db.commit()
            return len(gone)

    def doc_ids(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT DISTINCT doc_id FROM chunks")]

    # ---- reads ----
    @staticmethod
    def _chunk(r) -> Chunk:
        return Chunk(r[0], r[1], r[2], r[3], r[4], tuple(r[5].split(",")), r[6], json.loads(r[7] or "{}"))

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def visible_chunks(self, role: str) -> list[Chunk]:
        with self._lock:
            rows = self._db.execute("SELECT id, doc_id, title, section, text, acl, kind, meta FROM chunks").fetchall()
        return [c for c in (self._chunk(r) for r in rows) if can_see(c.acl, role)]

    def search(self, qvec: list[float], k: int, role: str) -> list[tuple[Chunk, float]]:
        with self._lock:
            rows = self._db.execute("SELECT id, doc_id, title, section, text, acl, kind, meta, vec FROM chunks").fetchall()
        scored = []
        for r in rows:
            if not can_see(r[5].split(","), role):
                continue
            vec = array.array("f")
            vec.frombytes(r[8])
            if len(vec) != len(qvec):
                raise ValueError(f"embedding dimension mismatch: store has {len(vec)}, query has {len(qvec)}; re-ingest with the same embedder")
            scored.append((self._chunk(r[:8]), cosine(list(vec), qvec)))
        scored.sort(key=lambda t: -t[1])
        return scored[:k]
