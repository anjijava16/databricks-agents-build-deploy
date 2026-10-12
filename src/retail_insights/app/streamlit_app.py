"""Retail AI Assistant: one Databricks App, three ways to answer.

    Agentic RAG (default)   plans, routes to SQL / policies / filings, grades, verifies
    Analytics agent         tool-calling agent over the gold tables (SQL only)
    Document RAG            classic retrieve-then-answer over the knowledge base

Local run (needs a Databricks CLI profile for the model and warehouse):
    DATABRICKS_WAREHOUSE_ID=... RETAIL_SCHEMA=retail_dev \\
    uv run streamlit run src/retail_insights/app/streamlit_app.py
Add RETAIL_RETRIEVER=local to use the bundled policy library instead of Vector Search.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Databricks Apps runs this file from the bundle root without installing the
# package, so make `src/` importable.
SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402  (ships with streamlit)
import streamlit as st  # noqa: E402

from retail_insights.agent.factory import Runtime  # noqa: E402

MODES = {
    "Agentic RAG": "Plans the question, pulls sales data, policies and SEC filings as needed, "
    "checks the sources and cites them.",
    "Analytics agent": "Answers from the sales tables with SQL tools only.",
    "Document RAG": "Searches the policy library and filings once and answers with citations.",
}
EXAMPLES = {
    "Agentic RAG": [
        "Did we give Electronics discounts above what our pricing policy allows in the last 30 days?",
        "How many customers are At Risk, and what does our playbook say we should do for them?",
        "What is our Electronics return rate, and what is our return window for Electronics?",
        "What does Best Buy's 10-K say about product returns, and how does our policy compare?",
        "When did Walmart file its latest 10-K?",
    ],
    "Analytics agent": [
        "What was net revenue and average order value over the last 7 days?",
        "Which region had the highest gross margin in the last 30 days?",
        "Top 5 products by net revenue in Electronics",
    ],
    "Document RAG": [
        "Do Gold members pay the electronics restocking fee?",
        "What is the maximum discount allowed on Grocery?",
        "What risks does Target describe about its supply chain?",
    ],
}

st.set_page_config(page_title="Retail AI Assistant", layout="wide")


@st.cache_resource
def runtime() -> Runtime:
    return Runtime.from_env()


def run(mode: str, question: str, history: list[dict]) -> dict:
    rt = runtime()
    if mode == "Analytics agent":
        if rt.analytics is None:
            return {"answer": "No SQL warehouse is configured for this app.", "kind": "error"}
        r = rt.analytics.run(question, history=history)
        return {"answer": r.answer, "kind": "analytics", "steps": r.steps}
    if mode == "Document RAG":
        r = rt.rag.run(question, history=history)
        return {
            "answer": r.answer,
            "kind": "rag",
            "grounded": r.grounded,
            "passages": r.passages,
            "trace": r.trace,
        }
    r = rt.agentic.run(question, history=history)
    return {
        "answer": r.answer,
        "kind": "agentic",
        "grounded": r.grounded,
        "evidence": r.evidence,
        "trace": r.steps,
        "routes": r.routes,
    }


def render_details(out: dict) -> None:
    kind = out.get("kind")
    if kind in ("rag", "agentic") and out.get("grounded") is False:
        st.caption("No well-cited answer was possible, so the assistant declined rather than guess.")
    if kind == "agentic":
        if out.get("routes"):
            st.caption("Routed to: " + ", ".join(out["routes"]))
        evidence = out.get("evidence") or []
        if evidence:
            with st.expander(f"Evidence ({len(evidence)})"):
                for i, e in enumerate(evidence, 1):
                    st.markdown(f"**[{i}] {e.header()}**")
                    for q in e.sql:
                        st.code(q, language="sql")
                    if e.rows:
                        st.dataframe(pd.DataFrame(e.rows[:50]), use_container_width=True)
                    elif e.passage:
                        meta = e.passage.url or e.passage.doc_id
                        st.caption(f"{meta} · {e.passage.chunk_id}")
                        st.markdown(e.content[:1200])
                    else:
                        st.text(e.content[:1200])
    if kind == "rag" and out.get("passages"):
        with st.expander(f"Sources ({len(out['passages'])})"):
            for i, p in enumerate(out["passages"], 1):
                st.markdown(f"**[{i}] {p.title}** · {p.section.split(' > ', 1)[-1]}")
                st.caption(p.content[:500])
    if kind == "analytics" and out.get("steps"):
        with st.expander(f"Tool calls ({len(out['steps'])})"):
            for i, s in enumerate(out["steps"], 1):
                st.markdown(f"**{i}. `{s.tool}`** · {s.seconds:.2f}s · {'ok' if s.ok else 'error'}")
                if s.sql:
                    st.code(s.sql, language="sql")
                if s.ok and s.result.get("rows"):
                    st.dataframe(pd.DataFrame(s.result["rows"]), use_container_width=True)
                elif not s.ok:
                    st.warning(s.result["error"])
    if out.get("trace"):
        with st.expander("Trace"):
            st.json(out["trace"], expanded=False)


with st.sidebar:
    st.header("Retail AI Assistant")
    mode = st.radio("Mode", list(MODES), index=0)
    st.caption(MODES[mode])
    rt = runtime()
    st.markdown(
        f"**Schema:** `{rt.catalog}.{rt.schema}`  \n**Model:** `{rt.endpoint}`  \n"
        f"**Retriever:** `{rt.retriever_kind}`"
    )
    st.subheader("Try asking")
    for q in EXAMPLES[mode]:
        if st.button(q, use_container_width=True, key=f"ex-{mode}-{q}"):
            st.session_state.pending = q
    if st.button("Clear chat"):
        st.session_state.messages = []

st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("out"):
            render_details(m["out"])

question = st.chat_input("Ask about sales, policies or other retailers' filings") or st.session_state.pop(
    "pending", None
)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]][-6:]
    with st.chat_message("assistant"), st.spinner("Working..."):
        try:
            out = run(mode, question, history)
        except Exception as e:
            out = {"answer": f"Something went wrong: `{e}`", "kind": "error"}
        st.markdown(out["answer"])
        render_details(out)
    st.session_state.messages.append({"role": "assistant", "content": out["answer"], "out": out})
