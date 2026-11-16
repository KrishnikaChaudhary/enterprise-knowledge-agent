import pytest

from kagent.chunking import chunk_markdown
from kagent.embeddings import HashEmbedder, cosine
from kagent.ingest import ingest, load_chunks
from kagent.errors import IngestError
from kagent.retrieval import BM25, Retriever, support
from kagent.store import SqliteVectorStore, can_see

from conftest import CORPUS


def test_hash_embedder_is_deterministic_normalised_and_lexically_sensible(embedder):
    a, b, c = embedder.embed(["annual leave days", "annual leave days", "server uptime"])
    assert a == b and abs(cosine(a, a) - 1) < 1e-9
    assert cosine(a, embedder.embed(["days of annual leave"])[0]) > cosine(a, c)


def test_can_see_rules():
    assert can_see(("all",), "employee") and can_see(("hr",), "hr") and can_see(("hr",), "admin")
    assert not can_see(("hr", "legal"), "employee") and not can_see(("security",), "hr")


def test_corpus_loads_docs_and_rows():
    chunks = load_chunks(CORPUS)
    assert {c.kind for c in chunks} == {"doc", "row"}
    assert any(c.id == "product/plans.csv#Pro" for c in chunks)
    assert next(c for c in chunks if c.doc_id == "hr/compensation_bands.md").acl == ("hr",)


def test_missing_corpus_dir_is_a_handled_error(tmp_path):
    with pytest.raises(IngestError):
        load_chunks(tmp_path / "nope")


def test_acl_filter_applies_before_scoring(store, embedder):
    r = Retriever(store, embedder)
    for mode in ("vector", "bm25", "hybrid"):
        r.mode = mode
        docs = {h.chunk.doc_id for h in r.retrieve("salary band level L4 bonus", "employee", 10)}
        assert "hr/compensation_bands.md" not in docs, mode
    r.mode = "hybrid"
    assert r.retrieve("salary band level L4", "hr", 1)[0].chunk.doc_id == "hr/compensation_bands.md"
    assert r.retrieve("salary band level L4", "admin", 1)[0].chunk.doc_id == "hr/compensation_bands.md"
    assert all(can_see(c.acl, "employee") for c in store.visible_chunks("employee"))


def test_bm25_nails_exact_identifiers():
    chunks = chunk_markdown("a.md", "# A\n\nError code E-4071 means the label printer is offline.\n\n# B\n\nPrinters are cleaned monthly by facilities.")
    top = BM25(chunks).search("E-4071", 2)[0][0]
    assert "E-4071" in top.text


def test_hybrid_fuses_both_rankings(store, embedder):
    r = Retriever(store, embedder, mode="hybrid")
    hits = r.retrieve("How many days of annual leave do full-time employees get?", "employee", 4)
    assert hits[0].chunk.doc_id == "hr/leave_policy.md"
    assert hits[0].vector_rank is not None and hits[0].bm25_rank is not None
    assert hits[0].score > 1 / 61  # contributions from both rankers add up


def test_support_score_separates_answerable_from_not(store, embedder):
    r = Retriever(store, embedder)
    good = r.retrieve("How long is parental leave for the primary caregiver?", "employee", 2)[0].support
    bad = r.retrieve("What is the CEO's favourite colour?", "employee", 2)[0].support
    assert good > 0.7 > 0.45 > bad


def test_invalid_mode_rejected(store, embedder):
    with pytest.raises(ValueError):
        Retriever(store, embedder, mode="magic")


def test_incremental_ingest_and_deletion(tmp_path, embedder):
    (tmp_path / "a.md").write_text("# One\n\nAlpha facts live here.\n")
    (tmp_path / "b.md").write_text("# Two\n\nBeta facts live here.\n")
    s = SqliteVectorStore(str(tmp_path / "kb.db"))
    r1 = ingest(tmp_path, s, embedder)
    assert (r1.chunks, r1.embedded, r1.skipped_unchanged) == (2, 2, 0)
    r2 = ingest(tmp_path, s, embedder)
    assert (r2.embedded, r2.skipped_unchanged) == (0, 2)  # nothing re-embedded
    (tmp_path / "a.md").write_text("# One\n\nAlpha facts changed.\n")
    (tmp_path / "b.md").unlink()
    r3 = ingest(tmp_path, s, embedder)
    assert (r3.embedded, r3.deleted, r3.skipped_unchanged) == (1, 1, 0)
    assert s.count() == 1
    # persistence: reopen
    assert SqliteVectorStore(str(tmp_path / "kb.db")).count() == 1


def test_changing_embedder_reembeds_everything(tmp_path):
    (tmp_path / "a.md").write_text("# One\n\nAlpha facts live here.\n")
    s = SqliteVectorStore()
    ingest(tmp_path, s, HashEmbedder(64))
    assert ingest(tmp_path, s, HashEmbedder(128)).embedded == 1


def test_dimension_mismatch_is_a_clear_error(tmp_path):
    (tmp_path / "a.md").write_text("# One\n\nAlpha facts live here.\n")
    s = SqliteVectorStore()
    ingest(tmp_path, s, HashEmbedder(64))
    with pytest.raises(ValueError, match="dimension mismatch"):
        s.search(HashEmbedder(128).embed(["alpha"])[0], 3, "employee")
