"""Agentic RAG as a LangGraph state machine.

                 ┌──────────────── rewrite ◄──────────────┐
                 ▼                                         │ (a doc task found nothing
    plan ──► gather ──► grade ──────────────────────────────┤  relevant, retries left)
     │        (analytics sub-agent,                         │
     │         policy/filing retrieval,                     ▼
     │         filing_lookup UC function)              synthesize ◄──┐
     │                                                      │        │ (citation check
     ▼                                                      ▼        │  failed, one retry)
  (no tasks) ──► finalize ◄──────────────────────────── verify ──────┘

What makes it agentic rather than a chain:
  * plan    the model decides which sources a question needs and splits
            compound questions into tasks ("our Electronics discounts" -> SQL;
            "our discount policy" -> policies).
  * gather  runs each task with the right tool. Analytics tasks are delegated
            to the simple tool-calling agent (agent composition), so SQL
            guardrails apply here too.
  * grade   the model judges every retrieved passage; irrelevant ones are dropped
            (corrective RAG).
  * rewrite a task with nothing relevant gets a reformulated query and one retry.
  * verify  a deterministic citation gate; one revision with feedback, then
            refuse instead of returning an unsupported answer.

Every loop is bounded, so worst-case cost is known up front.
"""

from __future__ import annotations

import json
import operator
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from retail_insights.agent.core import RetailAgent
from retail_insights.agent.executor import SqlExecutor
from retail_insights.agent.llm import ChatClient
from retail_insights.agentic import prompts
from retail_insights.knowledge.edgar import RETAIL_PEERS
from retail_insights.obs.tracing import span
from retail_insights.rag.chain import REFUSAL, complete_text, format_context
from retail_insights.rag.citations import CitationReport, check_citations
from retail_insights.rag.retrieval import Passage, Retriever

ROUTES = ("analytics", "policies", "filings", "filing_meta")
MAX_TASKS = 4

COMPANY_TICKERS = {
    "walmart": "WMT",
    "costco": "COST",
    "target": "TGT",
    "home depot": "HD",
    "lowe's": "LOW",
    "lowes": "LOW",
    "amazon": "AMZN",
    "kroger": "KR",
    "best buy": "BBY",
    "dollar general": "DG",
    "tjx": "TJX",
    "tj maxx": "TJX",
}
_ANALYTICS_WORDS = re.compile(
    r"\b(revenue|sales|sold|margin|orders?|units|aov|average order|discount(?:s|ed)? (?:given|applied)|"
    r"returns? amount|return rate|top|best[- ]selling|worst|region|store|channel|online|customers?|"
    r"segment|champions|at risk|how many|how much|trend|week|month|last \d+ days)\b",
    re.IGNORECASE,
)
_POLICY_WORDS = re.compile(
    r"\b(policy|policies|allowed|maximum|max|rule|window|refund|restocking|warranty|loyalty|tier|"
    r"playbook|definition|defined|supplier|shipping|fulfil+ment|procedure|approval|should we)\b",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------- state
@dataclass
class Task:
    route: str
    query: str
    ticker: str | None = None
    attempts: int = 0


@dataclass
class Evidence:
    kind: str  # data | policy | filing | filing_meta
    title: str
    content: str
    passage: Passage | None = None
    sql: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)

    def header(self) -> str:
        return {
            "data": "Sales data",
            "policy": "Cedarline policy",
            "filing": "SEC filing",
            "filing_meta": "Filing registry",
        }[self.kind] + f": {self.title}"


class AgenticState(TypedDict, total=False):
    question: str
    history: list[dict]
    tasks: list[Task]
    pending: list[Task]
    reply: str
    candidates: list[tuple[Task, list[Passage]]]
    evidence: list[Evidence]
    retry: list[Task]
    draft: str
    report: CitationReport
    revisions: int
    final: str
    grounded: bool
    steps: Annotated[list[dict], operator.add]


@dataclass
class AgenticResult:
    answer: str
    grounded: bool
    evidence: list[Evidence]
    tasks: list[Task]
    steps: list[dict]
    report: CitationReport | None = None

    @property
    def routes(self) -> list[str]:
        return [t.route for t in self.tasks]


# ---------------------------------------------------------------------------- helpers
def parse_json(text: str) -> Any:
    """Models wrap JSON in prose or ``` fences; pull out the first object."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON")


def resolve_ticker(value: str | None, question: str = "") -> str | None:
    if value and str(value).upper() in RETAIL_PEERS:
        return str(value).upper()
    low = question.lower()
    for name, ticker in COMPANY_TICKERS.items():
        if re.search(rf"\b{re.escape(name)}\b", low):
            return ticker
    for t in RETAIL_PEERS:
        if re.search(rf"\b{t}\b", question):
            return t
    return None


def heuristic_plan(question: str) -> list[Task]:
    """Keyword router used when the planner's output can't be parsed."""
    tasks: list[Task] = []
    ticker = resolve_ticker(None, question)
    if ticker and re.search(
        r"\b(filed|file|filing date|accession number|when did)\b", question, re.IGNORECASE
    ):
        tasks.append(Task("filing_meta", question, ticker))
    elif ticker:
        tasks.append(Task("filings", question, ticker))
    if _ANALYTICS_WORDS.search(question) and not ticker:
        tasks.append(Task("analytics", question))
    if _POLICY_WORDS.search(question):
        tasks.append(Task("policies", question))
    return tasks or [Task("policies", question)]


def _rows_text(rows: list[dict], limit: int = 25) -> str:
    return json.dumps(rows[:limit], default=str)


# ------------------------------------------------------------------------------ agent
class AgenticRAG:
    def __init__(
        self,
        llm: ChatClient,
        retriever: Retriever,
        analytics: RetailAgent | None = None,
        executor: SqlExecutor | None = None,
        filing_lookup_fn: str | None = None,
        k: int = 5,
        max_rewrites: int = 1,
        max_revisions: int = 1,
        min_coverage: float = 0.5,
    ):
        self.llm, self.retriever, self.analytics = llm, retriever, analytics
        self.executor, self.filing_lookup_fn = executor, filing_lookup_fn
        self.k, self.max_rewrites, self.max_revisions = k, max_rewrites, max_revisions
        self.min_coverage = min_coverage
        self.graph = self._build()

    # --------------------------------------------------------------------- nodes
    def plan(self, s: AgenticState) -> AgenticState:
        q = s["question"]
        with span("agentic.plan", "AGENT", {"question": q}) as sp:
            raw = complete_text(
                self.llm,
                prompts.PLANNER_PROMPT.format(tickers=", ".join(RETAIL_PEERS)),
                q,
                s.get("history"),
            )
            try:
                parsed = parse_json(raw)
                tasks = []
                for t in parsed.get("tasks", [])[:MAX_TASKS]:
                    route = str(t.get("route", "")).strip()
                    if route not in ROUTES:
                        continue
                    query = str(t.get("query") or q).strip()
                    ticker = (
                        resolve_ticker(t.get("ticker"), query)
                        if route in ("filings", "filing_meta")
                        else None
                    )
                    tasks.append(Task(route, query, ticker))
                reply = str(parsed.get("reply") or "") if not tasks else ""
                how = "llm"
                if not tasks and not reply:
                    tasks, how = heuristic_plan(q), "heuristic"
            except (ValueError, json.JSONDecodeError, AttributeError):
                tasks, reply, how = heuristic_plan(q), "", "heuristic"
            sp.output({"tasks": tasks, "reply": reply, "planner": how})
        return {
            "tasks": tasks,
            "pending": list(tasks),
            "reply": reply,
            "evidence": [],
            "candidates": [],
            "revisions": 0,
            "steps": [{"node": "plan", "planner": how, "tasks": [vars(t).copy() for t in tasks]}],
        }

    def gather(self, s: AgenticState) -> AgenticState:
        evidence = list(s.get("evidence", []))
        candidates: list[tuple[Task, list[Passage]]] = []
        log: list[dict] = []
        for task in s.get("pending", []):
            task.attempts += 1
            if task.route == "analytics":
                evidence.append(self._run_analytics(task, log))
            elif task.route == "filing_meta":
                ev = self._run_filing_meta(task, log)
                if ev:
                    evidence.append(ev)
            else:
                source = "policy" if task.route == "policies" else "filing"
                with span("agentic.retrieve", "RETRIEVER", vars(task).copy()) as sp:
                    hits = self.retriever.search(task.query, k=self.k, source=source, ticker=task.ticker)
                    sp.output(hits)
                candidates.append((task, hits))
                log.append(
                    {
                        "tool": "retrieve",
                        "route": task.route,
                        "query": task.query,
                        "ticker": task.ticker,
                        "hits": [h.chunk_id for h in hits],
                    }
                )
        return {
            "evidence": evidence,
            "candidates": candidates,
            "pending": [],
            "steps": [{"node": "gather", "calls": log}],
        }

    def _run_analytics(self, task: Task, log: list[dict]) -> Evidence:
        if self.analytics is None:
            log.append({"tool": "analytics", "error": "analytics agent not configured"})
            return Evidence("data", task.query, "Sales data is not available in this deployment.")
        with span("agentic.analytics", "AGENT", {"query": task.query}) as sp:
            result = self.analytics.run(task.query)
            sp.output({"answer": result.answer, "tools": result.tools_used})
        ok = [st for st in result.steps if st.ok]
        parts = [f"Analyst answer: {result.answer}"]
        rows: list[dict] = []
        for st in ok:
            r = st.result.get("rows") or st.result.get("columns") or []
            rows.extend(r if isinstance(r, list) else [])
            parts.append(f"{st.tool}({json.dumps(st.arguments, default=str)}) -> {_rows_text(r)}")
        log.append(
            {
                "tool": "analytics",
                "query": task.query,
                "sub_tools": result.tools_used,
                "ok": len(ok),
                "stopped": result.stopped_reason,
            }
        )
        return Evidence("data", task.query, "\n".join(parts), sql=[st.sql for st in ok if st.sql], rows=rows)

    def _run_filing_meta(self, task: Task, log: list[dict]) -> Evidence | None:
        if not (self.executor and self.filing_lookup_fn):
            log.append({"tool": "filing_lookup", "error": "not configured"})
            return None
        ticker = (task.ticker or "").replace("'", "")
        sql = f"SELECT * FROM {self.filing_lookup_fn}('{ticker}')"
        try:
            with span("agentic.filing_lookup", "TOOL", {"sql": sql}) as sp:
                rows = self.executor.run(sql).as_records()
                sp.output(rows)
        except Exception as e:
            log.append({"tool": "filing_lookup", "sql": sql, "error": str(e)})
            return None
        log.append({"tool": "filing_lookup", "sql": sql, "rows": len(rows)})
        title = f"filings on record for {ticker or 'all companies'}"
        body = _rows_text(rows) if rows else "No filings on record."
        return Evidence("filing_meta", title, body, sql=[sql], rows=rows)

    def grade(self, s: AgenticState) -> AgenticState:
        evidence = list(s.get("evidence", []))
        retry: list[Task] = []
        log = []
        for task, hits in s.get("candidates", []):
            keep = self._grade_hits(task, hits) if hits else []
            for p in keep:
                if not any(e.passage and e.passage.chunk_id == p.chunk_id for e in evidence):
                    kind = "policy" if p.source == "policy" else "filing"
                    evidence.append(Evidence(kind, p.label(), p.content, passage=p))
            if not keep and task.attempts <= self.max_rewrites:
                retry.append(task)
            log.append({"query": task.query, "retrieved": len(hits), "kept": [p.chunk_id for p in keep]})
        return {
            "evidence": evidence,
            "retry": retry,
            "candidates": [],
            "steps": [{"node": "grade", "results": log, "retry": len(retry)}],
        }

    def _grade_hits(self, task: Task, hits: list[Passage]) -> list[Passage]:
        listing = "\n\n".join(f"[{i}] {p.label()}\n{p.content[:700]}" for i, p in enumerate(hits, 1))
        with span("agentic.grade", "LLM", {"query": task.query}) as sp:
            raw = complete_text(
                self.llm, prompts.GRADER_PROMPT.format(question=task.query, passages=listing), "Grade."
            )
            try:
                ids = {int(i) for i in parse_json(raw).get("relevant", [])}
                keep = [p for i, p in enumerate(hits, 1) if i in ids]
            except (ValueError, json.JSONDecodeError, AttributeError, TypeError):
                keep = hits[:3]  # grader unusable: fall back to the top of the ranking
            sp.output([p.chunk_id for p in keep])
        return keep

    def rewrite(self, s: AgenticState) -> AgenticState:
        pending = []
        log = []
        for task in s.get("retry", []):
            source = "internal policies" if task.route == "policies" else "SEC 10-K filings"
            with span("agentic.rewrite", "LLM", {"query": task.query}) as sp:
                new_q = (
                    complete_text(
                        self.llm, prompts.REWRITE_PROMPT.format(source=source, query=task.query), "Rewrite."
                    )
                    .strip()
                    .strip('"')
                )
                sp.output(new_q)
            log.append({"from": task.query, "to": new_q})
            pending.append(Task(task.route, new_q or task.query, task.ticker, task.attempts))
        return {"pending": pending, "retry": [], "steps": [{"node": "rewrite", "rewrites": log}]}

    def synthesize(self, s: AgenticState) -> AgenticState:
        evidence = s.get("evidence", [])
        if not evidence:
            return {"draft": "", "steps": [{"node": "synthesize", "evidence": 0}]}
        listing = "\n\n".join(f"[{i}] {e.header()}\n{e.content}" for i, e in enumerate(evidence, 1))
        history = list(s.get("history") or [])
        user = s["question"]
        if s.get("revisions", 0) and s.get("draft"):
            r = s["report"]
            problems = []
            if r.invalid:
                problems.append(f"citations {r.invalid} do not exist")
            if not r.cited:
                problems.append("no citations at all")
            if r.coverage < self.min_coverage:
                problems.append(f"only {r.cited_sentences} of {r.claim_sentences} factual sentences cited")
            history += [{"role": "user", "content": user}, {"role": "assistant", "content": s["draft"]}]
            user = prompts.REVISION_NOTE.format(problems="; ".join(problems) or "unsupported claims")
        with span("agentic.synthesize", "LLM", {"question": s["question"]}) as sp:
            draft = complete_text(
                self.llm, prompts.SYNTHESIZER_PROMPT.format(evidence=listing), user, history
            )
            sp.output(draft)
        return {
            "draft": draft,
            "steps": [{"node": "synthesize", "evidence": len(evidence), "revision": s.get("revisions", 0)}],
        }

    def verify(self, s: AgenticState) -> AgenticState:
        report = check_citations(s.get("draft", ""), len(s.get("evidence", [])))
        grounded = report.grounded(self.min_coverage)
        return {
            "report": report,
            "grounded": grounded,
            "revisions": s.get("revisions", 0) + (0 if grounded else 1),
            "steps": [
                {
                    "node": "verify",
                    "grounded": grounded,
                    "cited": report.cited,
                    "invalid": report.invalid,
                    "coverage": round(report.coverage, 2),
                }
            ],
        }

    def finalize(self, s: AgenticState) -> AgenticState:
        if s.get("reply") and not s.get("tasks"):
            final, grounded = s["reply"] or prompts.SCOPE_REPLY, True
        elif s.get("grounded"):
            final, grounded = s["draft"], True
        else:
            final, grounded = REFUSAL, False
        return {"final": final, "grounded": grounded, "steps": [{"node": "finalize", "grounded": grounded}]}

    # -------------------------------------------------------------------- wiring
    def _after_plan(self, s: AgenticState) -> str:
        return "gather" if s.get("tasks") else "finalize"

    def _after_grade(self, s: AgenticState) -> str:
        return "rewrite" if s.get("retry") else "synthesize"

    def _after_verify(self, s: AgenticState) -> str:
        if s.get("grounded") or not s.get("evidence"):
            return "finalize"
        return "synthesize" if s.get("revisions", 0) <= self.max_revisions else "finalize"

    def _build(self):
        from langgraph.graph import END, StateGraph

        g = StateGraph(AgenticState)
        for name in ("plan", "gather", "grade", "rewrite", "synthesize", "verify", "finalize"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("plan")
        g.add_conditional_edges("plan", self._after_plan, {"gather": "gather", "finalize": "finalize"})
        g.add_edge("gather", "grade")
        g.add_conditional_edges(
            "grade", self._after_grade, {"rewrite": "rewrite", "synthesize": "synthesize"}
        )
        g.add_edge("rewrite", "gather")
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges(
            "verify", self._after_verify, {"synthesize": "synthesize", "finalize": "finalize"}
        )
        g.add_edge("finalize", END)
        return g.compile()

    # ------------------------------------------------------------------------ API
    def run(self, question: str, history: list[dict] | None = None) -> AgenticResult:
        with span("agentic_rag", "AGENT", {"question": question}) as sp:
            out = self.graph.invoke(
                {"question": question, "history": history or [], "steps": []},
                {"recursion_limit": 25},
            )
            sp.output(out.get("final", ""))
        return AgenticResult(
            answer=out.get("final", REFUSAL),
            grounded=bool(out.get("grounded")),
            evidence=out.get("evidence", []),
            tasks=out.get("tasks", []),
            steps=out.get("steps", []),
            report=out.get("report"),
        )

    def stream(self, question: str, history: list[dict] | None = None):
        """Yield (node, update) as the graph runs; used by the demo script and the app."""
        for chunk in self.graph.stream(
            {"question": question, "history": history or [], "steps": []}, {"recursion_limit": 25}
        ):
            yield from chunk.items()


__all__ = ["AgenticRAG", "AgenticResult", "Evidence", "Task", "format_context"]
