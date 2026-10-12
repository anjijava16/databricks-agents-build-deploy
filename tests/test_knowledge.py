from retail_insights.knowledge import chunking
from retail_insights.knowledge.chunking import (
    Section,
    chunk_document,
    filing_sections,
    merge_small_sections,
    pack,
)
from retail_insights.knowledge.corpus import policy_chunks, policy_documents
from retail_insights.knowledge.documents import (
    FILING,
    POLICY,
    html_to_text,
    parse_file,
    parse_filing,
    parse_front_matter,
    parse_policy,
)

FILING_HTML = """<html><head><title>x</title><style>.a{color:red}</style></head><body>
<ix:header><ix:hidden>dei:EntityCentralIndexKey 0000000001</ix:hidden></ix:header>
<div style="display:none">hidden boilerplate</div>
<script>var tracking = 1;</script>
<p>EXAMPLE RETAIL CORP</p><p>Annual Report on Form 10-K</p>
<table><tr><td>Item 1.</td><td>Business</td><td>4</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>9</td></tr></table>
<p>Item 1. Business</p>
<p>{business}</p>
<p>Item 1A. Risk Factors</p>
<p>{risks}</p>
<p>Item 2. Properties</p>
<p>Not applicable.</p>
</body></html>"""


def sample_filing_html() -> str:
    business = " ".join(["Example Retail Corp operates general merchandise stores."] * 40)
    risks = " ".join(["Disruption in our supply chain could reduce product availability."] * 40)
    return FILING_HTML.replace("{business}", business).replace("{risks}", risks)


def test_front_matter_and_policy_parse():
    meta, body = parse_front_matter("---\ndoc_id: POL-X\ntitle: 'Thing'\n---\n# Thing\nbody")
    assert meta == {"doc_id": "POL-X", "title": "Thing"} and body.startswith("# Thing")
    doc = parse_policy("---\ndoc_id: POL-X\ntitle: Thing\nversion: 2\n---\n# Thing\n\nHello there.")
    assert (doc.doc_id, doc.source, doc.title, doc.metadata["version"]) == ("POL-X", POLICY, "Thing", "2")
    assert len(doc.content_hash) == 64


def test_html_to_text_drops_noise():
    text = html_to_text(sample_filing_html())
    assert "EXAMPLE RETAIL CORP" in text
    for junk in ("EntityCentralIndexKey", "hidden boilerplate", "tracking", "color:red"):
        assert junk not in text


def test_filing_sections_skip_table_of_contents():
    doc = parse_filing(
        sample_filing_html(),
        {
            "ticker": "exr",
            "company": "Example Retail Corp",
            "filing_date": "2026-03-01",
            "accession_number": "0000000001-26-000001",
        },
    )
    assert doc.source == FILING and doc.doc_id == "EXR-0000000001-26-000001"
    names = [s.path.split(" > ")[-1] for s in filing_sections(doc.text, doc.title)]
    assert names == ["Item 1. Business", "Item 1A. Risk Factors"]  # TOC rows and "Not applicable" skipped


def test_filing_chunks_carry_metadata():
    doc = parse_file(
        "/Volumes/x/filings/EXR/0001.html",
        sample_filing_html(),
        '{"ticker": "EXR", "company": "Example Retail Corp", "filing_date": "2026-03-01", '
        '"accession_number": "0001", "url": "https://example.com/f"}',
    )
    chunks = chunk_document(doc)
    assert chunks and all(c.ticker == "EXR" and c.url == "https://example.com/f" for c in chunks)
    assert any("Risk Factors" in c.section for c in chunks)
    assert all(c.word_count <= chunking.MAX_WORDS for c in chunks)


def test_pack_respects_target_and_overlap():
    text = "\n\n".join(" ".join(f"w{p}_{i}" for i in range(80)) for p in range(6))
    chunks = pack(text, target=200, overlap=20)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 200 + 20 for c in chunks)
    assert chunks[1].split()[:20] == chunks[0].split()[-20:]  # overlap carried forward


def test_merge_small_sections():
    merged = merge_small_sections(
        [Section("A > x", "short"), Section("A > y", " ".join(["word"] * 80)), Section("A > z", "tiny tail")]
    )
    assert [s.path for s in merged] == ["A > x + y + z"]


def test_policy_corpus_chunks_are_stable_and_unique():
    docs = policy_documents()
    assert len(docs) == 10 and all(d.source == POLICY for d in docs)
    chunks = policy_chunks()
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == [c.chunk_id for c in policy_chunks()]  # deterministic
    for c in chunks:
        assert c.embed_text.startswith(f"{c.title} | {c.section}")
        assert chunking.MIN_SECTION_WORDS <= c.word_count <= chunking.MAX_WORDS
    assert {c.doc_id for c in chunks} == {d.doc_id for d in docs}


def test_policies_agree_with_the_sales_data():
    """The cross-source demo depends on these facts lining up with datagen.py."""
    text = {d.doc_id: d.text for d in policy_documents()}
    assert "Electronics: maximum 15%" in text["POL-PRC-003"]
    assert "10%, 15% and 20%" in text["POL-PRC-003"]
    for tier in ("Bronze", "Silver", "Gold"):
        assert tier in text["POL-LOY-002"]
    for seg in ("Champions", "Loyal", "New or Promising", "At Risk", "Needs Attention"):
        assert seg in text["POL-SEG-005"]
