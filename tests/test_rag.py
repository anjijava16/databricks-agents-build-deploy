import json
from types import SimpleNamespace

from retail_insights.agent.fakes import RuleLLM
from retail_insights.knowledge.chunking import Chunk
from retail_insights.knowledge.corpus import policy_chunks
from retail_insights.rag.chain import REFUSAL, RagChain
from retail_insights.rag.citations import check_citations
from retail_insights.rag.retrieval import BM25Retriever, Passage, VectorSearchRetriever, diversify, tokenize


def chunk(cid, doc, text, source="policy", ticker=""):
    return Chunk(cid, doc, source, doc, "S", 0, text, f"{doc} | S\n{text}", len(text.split()), ticker)


# ------------------------------------------------------------------------- retrieval
def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The returns of Electronics") == ["return", "electronic"]


def test_bm25_ranks_and_filters():
    chunks = [
        chunk("a", "A", "electronics return window is 15 days"),
        chunk("b", "B", "grocery items are not returnable"),
        chunk("c", "C", "Best Buy describes returns of electronics", source="filing", ticker="BBY"),
        chunk("d", "D", "Walmart describes returns of electronics", source="filing", ticker="WMT"),
    ]
    r = BM25Retriever(chunks)
    assert r.search("electronics return window")[0].chunk_id == "a"
    assert [p.chunk_id for p in r.search("electronics returns", source="policy")] == ["a"]
    # PolicyPilot regression: a resolved ticker must scope retrieval to that company
    assert [p.ticker for p in r.search("returns electronics", ticker="bby")] == ["BBY"]
    assert r.search("quantum chromodynamics") == []


def test_diversify_caps_per_document():
    ps = [Passage(f"{d}{i}", d, "policy", d, "", "") for d, i in [("A", 1), ("A", 2), ("A", 3), ("B", 1)]]
    assert [p.chunk_id for p in diversify(ps, k=3, per_doc=2)] == ["A1", "A2", "B1"]


def test_bm25_on_real_corpus():
    r = BM25Retriever(policy_chunks())
    assert r.search("maximum promotional discount Grocery", k=1)[0].doc_id == "POL-PRC-003"
    assert r.search("OTIF on-time in-full target", k=1)[0].doc_id == "POL-SUP-007"


class FakeIndexes:
    def __init__(self):
        self.kwargs = None

    def query_index(self, **kwargs):
        self.kwargs = kwargs
        cols = [
            "chunk_id",
            "doc_id",
            "source",
            "title",
            "section",
            "content",
            "ticker",
            "company",
            "filing_date",
            "url",
            "score",
        ]
        row = [
            "BBY-1::0003",
            "BBY-1",
            "filing",
            "Best Buy 10-K",
            "Item 1A",
            "text",
            "BBY",
            "Best Buy",
            "2026-03-20",
            "https://sec.gov/x",
            0.81,
        ]
        return SimpleNamespace(
            manifest=SimpleNamespace(columns=[SimpleNamespace(name=c) for c in cols]),
            result=SimpleNamespace(data_array=[row]),
        )


def test_vector_search_retriever_sends_hybrid_query_with_filters():
    idx = FakeIndexes()
    r = VectorSearchRetriever(
        "workspace.retail_dev.gold_doc_chunks_index", _client=SimpleNamespace(vector_search_indexes=idx)
    )
    hits = r.search("returns", k=2, source="filing", ticker="bby")
    assert idx.kwargs["query_type"] == "HYBRID" and idx.kwargs["num_results"] == 6
    assert json.loads(idx.kwargs["filters_json"]) == {"source": "filing", "ticker": "BBY"}
    assert hits[0].chunk_id == "BBY-1::0003" and hits[0].score == 0.81 and hits[0].ticker == "BBY"


# ------------------------------------------------------------------------- citations
def test_citation_report():
    r = check_citations("Electronics have 15 days [1]. Gold members are exempt [2].", 2)
    assert r.cited == [1, 2] and not r.invalid and r.coverage == 1.0 and r.grounded()
    r = check_citations("Electronics have 15 days [3].", 2)
    assert r.invalid == [3] and not r.grounded()
    r = check_citations("Electronics have 15 days [1]. Gold members are exempt from the fee.", 2)
    assert r.coverage == 0.5 and r.grounded(0.5) and not r.grounded(0.8)
    assert not check_citations("No sources here at all for this.", 2).grounded()


# ----------------------------------------------------------------------------- chain
def llm_answering(text):
    return RuleLLM([("policy and research assistant", lambda s, m: text)])


def test_rag_chain_grounded_answer():
    llm = llm_answering("Electronics can be returned within 15 days [1].")
    r = RagChain(llm, BM25Retriever(policy_chunks())).run("How long to return electronics?")
    assert r.grounded and r.answer.endswith("[1].")
    assert r.cited_passages[0][1].chunk_id == r.passages[0].chunk_id


def test_rag_chain_refuses_uncited_or_empty():
    r = RagChain(llm_answering("Probably 30 days."), BM25Retriever(policy_chunks())).run(
        "electronics returns"
    )
    assert not r.grounded and r.answer == REFUSAL and r.draft == "Probably 30 days."
    empty = RagChain(llm_answering("x [1]."), BM25Retriever([])).run("anything")
    assert empty.answer == REFUSAL and empty.passages == []
