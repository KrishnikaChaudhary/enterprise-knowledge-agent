"""Ingestion pipeline: files -> chunks -> embeddings -> store. Incremental and idempotent.

Unchanged chunks (same content hash and embedding model) are not re-embedded, so re-running over an
edited corpus only pays for what changed; chunks of a document that no longer exist are deleted.
Structured data: a CSV is ingested as one chunk per row; its ACL comes from a `<name>.acl` sidecar file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_csv, chunk_markdown, parse_acl
from .embeddings import Embedder
from .errors import IngestError
from .store import SqliteVectorStore
from .tracing import Tracer, count_tokens


@dataclass
class IngestReport:
    docs: int = 0
    chunks: int = 0
    embedded: int = 0
    skipped_unchanged: int = 0
    deleted: int = 0
    embed_tokens: int = 0


def load_chunks(root: str | Path, max_words: int = 120) -> list[Chunk]:
    root = Path(root)
    if not root.is_dir():
        raise IngestError(f"corpus directory not found: {root}")
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".csv"}:
            continue
        doc_id = path.relative_to(root).as_posix()
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise IngestError(f"cannot read {doc_id}: {exc}") from exc
        if path.suffix.lower() == ".csv":
            side = path.with_suffix(".acl")
            acl = parse_acl(side.read_text().strip()) if side.exists() else ("all",)
            first = raw.splitlines()[0].split(",")[0].strip() if raw.strip() else None
            chunks += chunk_csv(doc_id, raw, title=path.stem.replace("_", " ").title(), acl=acl, key_column=first)
        else:
            chunks += chunk_markdown(doc_id, raw, max_words=max_words)
    return chunks


def ingest(root: str | Path, store: SqliteVectorStore, embedder: Embedder, tracer: Tracer | None = None,
           max_words: int = 120, batch: int = 64) -> IngestReport:
    chunks = load_chunks(root, max_words)
    known = store.existing_hashes()
    rep = IngestReport(docs=len({c.doc_id for c in chunks}), chunks=len(chunks))
    todo = [c for c in chunks if known.get(c.id) != (c.content_hash, embedder.name)]
    rep.skipped_unchanged = len(chunks) - len(todo)
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        texts = [c.embed_text for c in part]
        if tracer:
            with tracer.span("embed_chunks", "embed", n=len(part), model=embedder.name) as sp:
                vecs = embedder.embed(texts, "document")
                sp.tokens_in = sum(count_tokens(t) for t in texts)
        else:
            vecs = embedder.embed(texts, "document")
        rep.embed_tokens += sum(count_tokens(t) for t in texts)
        store.upsert(part, vecs, embedder.name)
        rep.embedded += len(part)
    by_doc: dict[str, set[str]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, set()).add(c.id)
    for doc in store.doc_ids():
        rep.deleted += store.delete_missing(doc, by_doc.get(doc, set()))
    return rep
