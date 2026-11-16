from kagent.chunking import chunk_csv, chunk_markdown, parse_acl, parse_front_matter
from kagent.textutil import sentences, stem, tokens

DOC = """---
title: Sample Policy
acl: HR, Legal
---
# Leave

Employees get 24 days. Unused days expire.

# Sick

Ten days per year. A note is needed after 3 days.
"""


def test_tokens_stem_and_stopwords():
    assert tokens("Reimbursed expenses over $50 need receipts") == ["reimburs", "expens", "over", "50", "need", "receipt"]
    assert stem("days") == "day" and stem("ies") == "ies" and stem("policies") == "policy"
    assert tokens("VPN-required on Wi-Fi") == ["vpn", "requir", "wi", "fi"]


def test_front_matter_and_acl():
    meta, body = parse_front_matter(DOC)
    assert meta == {"title": "Sample Policy", "acl": "HR, Legal"} and body.startswith("# Leave")
    assert parse_acl("HR, Legal") == ("hr", "legal") and parse_acl(None) == ("all",) and parse_acl("") == ("all",)


def test_markdown_chunks_keep_title_section_acl_and_header():
    chunks = chunk_markdown("p.md", DOC)
    assert [c.section for c in chunks] == ["Leave", "Sick"]
    assert all(c.title == "Sample Policy" and c.acl == ("hr", "legal") for c in chunks)
    assert chunks[0].embed_text.startswith("Sample Policy > Leave\n")
    assert chunks[0].id == "p.md#0" and chunks[1].id == "p.md#1"


def test_long_section_splits_with_overlap():
    body = "# S\n\n" + " ".join(f"Sentence number {i} has several words in it." for i in range(40))
    chunks = chunk_markdown("long.md", body, max_words=40, overlap_sentences=1)
    assert len(chunks) > 3
    first_last = sentences(chunks[0].text)[-1]
    assert sentences(chunks[1].text)[0] == first_last  # one-sentence overlap carried over
    assert all(len(c.text.split()) <= 40 + 10 for c in chunks)


def test_content_hash_changes_with_text_and_acl():
    a = chunk_markdown("p.md", DOC)[0]
    b = chunk_markdown("p.md", DOC.replace("24 days", "25 days"))[0]
    c = chunk_markdown("p.md", DOC.replace("HR, Legal", "HR"))[0]
    assert len({a.content_hash, b.content_hash, c.content_hash}) == 3


def test_csv_rows_become_chunks_with_key_and_metadata():
    raw = "plan,price,seats\nPro,299,25\nStarter,49,5\n,,\n"
    chunks = chunk_csv("plans.csv", raw, title="Plans", key_column="plan")
    assert [c.id for c in chunks] == ["plans.csv#Pro", "plans.csv#Starter"]
    assert chunks[0].text == "plan: Pro; price: 299; seats: 25" and chunks[0].kind == "row"
    assert chunks[0].meta["row"]["price"] == "299"


def test_empty_and_heading_only_documents():
    assert chunk_markdown("e.md", "") == []
    assert chunk_markdown("h.md", "# Only a heading") == []
