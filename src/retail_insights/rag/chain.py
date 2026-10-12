"""Classic RAG: retrieve -> stuff numbered context -> answer -> citation gate.

One retrieval, one generation, a deterministic check. No loops, no tools. This
is the baseline that the agentic version (retail_insights.agentic) is measured
against, and it is the right choice when questions are about documents only
and are reasonably well phrased.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retail_insights.agent.llm import ChatClient
from retail_insights.obs.tracing import span
from retail_insights.rag.citations import CitationReport, check_citations
from retail_insights.rag.retrieval import Passage, Retriever

RAG_SYSTEM_PROMPT = """You are the policy and research assistant for Cedarline Markets, a US \
retail chain. Answer using ONLY the numbered sources below: internal company policies and \
public SEC filings of other retailers.

Rules:
- End every sentence that states a fact with the bracketed number of its source, like [2]. \
Use only numbers that appear below. No other citation style.
- If the sources do not contain the answer, say you could not find it in the knowledge base. \
Do not use general knowledge.
- Policies describe Cedarline. SEC filings describe the company that filed them; never present \
another retailer's practice as Cedarline's.
- Be concise: the direct answer first, then supporting detail.

Sources:
{context}"""

REFUSAL = (
    "I couldn't find a grounded answer to that in the knowledge base. Try rephrasing, or ask "
    "about a specific policy or one of the retailers whose filings are indexed."
)


def format_context(passages: list[Passage], start: int = 1) -> str:
    blocks = []
    for i, p in enumerate(passages, start=start):
        origin = f"SEC filing: {p.title}" if p.source == "filing" else f"Cedarline policy: {p.title}"
        blocks.append(f"[{i}] {origin} | {p.section.split(' > ', 1)[-1]}\n{p.content}")
    return "\n\n".join(blocks)


def complete_text(llm: ChatClient, system: str, user: str, history: list[dict] | None = None) -> str:
    messages = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}]
    return (llm.complete(messages, tools=[]).content or "").strip()


@dataclass
class RagResult:
    answer: str
    passages: list[Passage]
    grounded: bool
    report: CitationReport
    draft: str = ""
    trace: list[dict] = field(default_factory=list)

    @property
    def cited_passages(self) -> list[tuple[int, Passage]]:
        return [(n, self.passages[n - 1]) for n in self.report.cited if n <= len(self.passages)]


class RagChain:
    def __init__(self, llm: ChatClient, retriever: Retriever, k: int = 6, min_coverage: float = 0.5):
        self.llm, self.retriever, self.k, self.min_coverage = llm, retriever, k, min_coverage

    def run(self, question: str, history: list[dict] | None = None) -> RagResult:
        with span("rag.retrieve", "RETRIEVER", {"query": question}) as s:
            passages = self.retriever.search(question, k=self.k)
            s.output(passages)
        if not passages:
            return RagResult(REFUSAL, [], False, CitationReport(), trace=[{"step": "retrieve", "hits": 0}])

        with span("rag.generate", "LLM", {"question": question}) as s:
            draft = complete_text(
                self.llm, RAG_SYSTEM_PROMPT.format(context=format_context(passages)), question, history
            )
            s.output(draft)
        report = check_citations(draft, len(passages))
        grounded = report.grounded(self.min_coverage)
        return RagResult(
            answer=draft if grounded else REFUSAL,
            passages=passages,
            grounded=grounded,
            report=report,
            draft=draft,
            trace=[
                {"step": "retrieve", "hits": len(passages), "top": [p.chunk_id for p in passages]},
                {"step": "generate", "chars": len(draft)},
                {
                    "step": "verify",
                    "cited": report.cited,
                    "invalid": report.invalid,
                    "coverage": round(report.coverage, 2),
                    "grounded": grounded,
                },
            ],
        )
