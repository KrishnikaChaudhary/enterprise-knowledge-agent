"""Structure-aware chunking.

Markdown is split on headings, then packed paragraph by paragraph up to `max_words`, carrying a
one-sentence overlap between chunks of the same section. Every chunk keeps its title and section so the
text that gets embedded starts with a contextual header ("Doc title > Section").
CSV rows become one chunk each ("column: value; ...") so structured facts are retrievable too.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field

from .textutil import sentences


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    text: str
    acl: tuple[str, ...] = ("all",)
    kind: str = "doc"  # doc | row
    meta: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        return f"{self.title} > {self.section}\n{self.text}" if self.section else f"{self.title}\n{self.text}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256((self.embed_text + "|" + ",".join(self.acl)).encode()).hexdigest()[:16]


def parse_front_matter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.find("\n---", 4)
    if end < 0:
        return {}, raw
    meta = {}
    for line in raw[4:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    return meta, raw[end + 4:].lstrip("\n")


def parse_acl(value: str | None) -> tuple[str, ...]:
    roles = tuple(sorted({r.strip().lower() for r in (value or "all").split(",") if r.strip()}))
    return roles or ("all",)


def chunk_markdown(doc_id: str, raw: str, max_words: int = 120, overlap_sentences: int = 1) -> list[Chunk]:
    meta, body = parse_front_matter(raw)
    title = meta.get("title") or doc_id
    acl = parse_acl(meta.get("acl"))
    sections: list[tuple[str, list[str]]] = [("", [])]
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"^#{1,6}\s+(.*)$", block.split("\n", 1)[0])
        if m:
            sections.append((m.group(1).strip(), []))
            block = block.split("\n", 1)[1].strip() if "\n" in block else ""
            if not block:
                continue
        sections[-1][1].append(block)
    chunks: list[Chunk] = []
    for section, paras in sections:
        if not paras:
            continue
        cur: list[str] = []
        words = 0
        carry: list[str] = []

        def flush() -> None:
            nonlocal cur, words, carry
            if not cur:
                return
            text = "\n".join(cur).strip()
            idx = len(chunks)
            chunks.append(Chunk(f"{doc_id}#{idx}", doc_id, title, section, text, acl, "doc"))
            sents = sentences(text)
            carry = sents[-overlap_sentences:] if overlap_sentences else []
            cur, words = list(carry), sum(len(s.split()) for s in carry)

        for para in paras:
            for sent in sentences(para) or [para]:
                n = len(sent.split())
                if words + n > max_words and cur and cur != carry:
                    flush()
                cur.append(sent)
                words += n
        if cur and cur != carry:
            flush()
    return chunks


def chunk_csv(doc_id: str, raw: str, title: str | None = None, acl: tuple[str, ...] = ("all",), key_column: str | None = None) -> list[Chunk]:
    reader = csv.DictReader(io.StringIO(raw))
    chunks = []
    for i, row in enumerate(reader):
        row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
        if not any(row.values()):
            continue
        key = row.get(key_column or "", "") or str(i)
        text = "; ".join(f"{k}: {v}" for k, v in row.items() if v)
        chunks.append(Chunk(f"{doc_id}#{key}", doc_id, title or doc_id, key, text, acl, "row", {"row": row}))
    return chunks
