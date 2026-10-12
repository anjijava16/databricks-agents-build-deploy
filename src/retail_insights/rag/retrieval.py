"""Retrievers.

    Retriever.search(query, k, source=None, ticker=None) -> list[Passage]

Two implementations behind one protocol:

  * VectorSearchRetriever: Databricks Vector Search, HYBRID query (dense
    embeddings from the Databricks-hosted gte-large model + keyword matching),
    with metadata filters. Called through the Databricks SDK, so the same
    unified auth works in a notebook, a job, GitHub Actions and a Databricks App.
  * BM25Retriever: pure-Python keyword ranking over in-memory chunks. Powers
    local development and CI, where there is no workspace. It is a real
    ranking function, so offline retrieval metrics in CI mean something.

PolicyPilot lesson kept here: when the question is about one company, the
ticker filter must reach the search call. Otherwise, as the corpus grows, the
top hits come from the wrong company.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from retail_insights.knowledge.chunking import Chunk


@dataclass
class Passage:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    section: str
    content: str
    score: float = 0.0
    ticker: str = ""
    company: str = ""
    filing_date: str = ""
    url: str = ""
    extra: dict = field(default_factory=dict)

    def label(self) -> str:
        if self.source == "filing":
            return f"{self.title} > {self.section.split(' > ')[-1]}"
        return self.section or self.title

    @classmethod
    def from_chunk(cls, c: Chunk, score: float = 0.0) -> Passage:
        return cls(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            source=c.source,
            title=c.title,
            section=c.section,
            content=c.content,
            score=score,
            ticker=c.ticker,
            company=c.company,
            filing_date=c.filing_date,
            url=c.url,
        )


class Retriever(Protocol):
    def search(
        self, query: str, k: int = 5, source: str | None = None, ticker: str | None = None
    ) -> list[Passage]: ...


def diversify(passages: list[Passage], k: int, per_doc: int = 2) -> list[Passage]:
    """Keep ranking order but cap passages per document, so one long document
    cannot crowd out everything else."""
    out, seen = [], Counter()
    for p in passages:
        if seen[p.doc_id] < per_doc:
            out.append(p)
            seen[p.doc_id] += 1
        if len(out) == k:
            break
    return out


# ------------------------------------------------------------------------------- BM25
_TOKEN = re.compile(r"[a-z0-9]+(?:[.%][0-9]+)?")
_STOP = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for t in _TOKEN.findall(text.lower()):
        if t in _STOP:
            continue
        # light stemming: plural and -ing forms collapse onto the same token
        for suffix in ("ing", "es", "s"):
            if len(t) > 4 and t.endswith(suffix):
                t = t[: -len(suffix)]
                break
        tokens.append(t)
    return tokens


class BM25Retriever:
    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.embed_text)) for c in self.chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, q: list[str], i: int) -> float:
        d, length = self.docs[i], self.lengths[i]
        s = 0.0
        for t in q:
            tf = d.get(t)
            if tf:
                s += (
                    self.idf[t]
                    * tf
                    * (self.k1 + 1)
                    / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
                )
        return s

    def search(
        self, query: str, k: int = 5, source: str | None = None, ticker: str | None = None
    ) -> list[Passage]:
        q = tokenize(query)
        scored = []
        for i, c in enumerate(self.chunks):
            if source and c.source != source:
                continue
            if ticker and c.ticker.upper() != ticker.upper():
                continue
            s = self._score(q, i)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], self.chunks[x[1]].chunk_id))
        ranked = [Passage.from_chunk(self.chunks[i], round(s, 4)) for s, i in scored]
        return diversify(ranked, k)


# ---------------------------------------------------------------------- Vector Search
RESULT_COLUMNS = [
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
]


@dataclass
class VectorSearchRetriever:
    index_name: str
    query_type: str = "HYBRID"
    overfetch: int = 3  # fetch k*overfetch, then diversify down to k
    _client: object = field(default=None, repr=False)

    def _ws(self):
        if self._client is None:
            from databricks.sdk import WorkspaceClient

            self._client = WorkspaceClient()
        return self._client

    def search(
        self, query: str, k: int = 5, source: str | None = None, ticker: str | None = None
    ) -> list[Passage]:
        filters = {}
        if source:
            filters["source"] = source
        if ticker:
            filters["ticker"] = ticker.upper()
        resp = self._ws().vector_search_indexes.query_index(
            index_name=self.index_name,
            columns=RESULT_COLUMNS,
            query_text=query,
            query_type=self.query_type,
            num_results=k * self.overfetch,
            filters_json=json.dumps(filters) if filters else None,
        )
        return diversify(parse_query_response(resp), k)


def parse_query_response(resp) -> list[Passage]:
    """Rows come back as positional arrays; the manifest names the columns and
    the similarity score is appended as the last column."""
    names = [c.name for c in (resp.manifest.columns or [])] if resp.manifest else RESULT_COLUMNS
    rows = (resp.result.data_array or []) if resp.result else []
    out = []
    for row in rows:
        rec = dict(zip(names, row, strict=False))
        score = float(rec.pop("score", row[-1] if len(row) > len(RESULT_COLUMNS) else 0.0) or 0.0)
        out.append(
            Passage(
                **{c: (rec.get(c) or "") for c in RESULT_COLUMNS},
                score=score,
            )
        )
    return out
