import json

import pytest

from retail_insights.agent.core import RetailAgent
from retail_insights.agent.executor import QueryResult
from retail_insights.agent.fakes import FakeExecutor, RuleLLM, call, say
from retail_insights.agent.tools import Toolbox
from retail_insights.agentic.graph import AgenticRAG, heuristic_plan, parse_json, resolve_ticker
from retail_insights.knowledge.corpus import policy_chunks
from retail_insights.rag.chain import REFUSAL
from retail_insights.rag.retrieval import BM25Retriever

PLANNER = "You plan how to answer"
GRADER = "You check whether retrieved passages"
SYNTH = "Answer the user's question using ONLY the numbered evidence"
ANALYST = "retail analytics assistant"
REWRITE = "A search over"


def analyst(system, messages):
    if messages[-1]["role"] == "tool":
        return say("42 Electronics lines had a 20% discount.")
    return call("run_sql", query="SELECT COUNT(*) AS lines FROM silver_sales WHERE discount_pct > 0.15")


def build(rules, retriever=None, executor=None):
    llm = RuleLLM([*rules, (ANALYST, analyst)])
    ex = executor or FakeExecutor(lambda sql: QueryResult(["lines"], [[42]]))
    return (
        AgenticRAG(
            llm,
            retriever or BM25Retriever(policy_chunks()),
            analytics=RetailAgent(llm, Toolbox(ex, "workspace", "retail_dev")),
            executor=ex,
            filing_lookup_fn="workspace.retail_dev.filing_lookup",
        ),
        llm,
        ex,
    )


def plan(*tasks):
    return lambda s, m: json.dumps(
        {"tasks": [dict(zip(("route", "query", "ticker"), t, strict=False)) for t in tasks]}
    )


def test_cross_source_question_is_grounded_in_data_and_policy():
    agent, _llm, _ex = build(
        [
            (
                PLANNER,
                plan(
                    ("analytics", "Electronics lines above 15% discount"),
                    ("policies", "maximum discount Electronics"),
                ),
            ),
            (GRADER, lambda s, m: '{"relevant": [1]}'),
            (SYNTH, lambda s, m: "42 lines were discounted 20% [1]. The Electronics cap is 15% [2]."),
        ]
    )
    r = agent.run("Did we break the Electronics discount policy?")
    assert r.grounded and r.routes == ["analytics", "policies"]
    assert [e.kind for e in r.evidence] == ["data", "policy"]
    assert r.evidence[0].sql and "silver_sales" in r.evidence[0].sql[0]
    assert r.evidence[1].passage.doc_id == "POL-PRC-003"
    assert [s["node"] for s in r.steps] == ["plan", "gather", "grade", "synthesize", "verify", "finalize"]


def test_rewrite_loop_when_nothing_relevant():
    grades = iter(['{"relevant": []}', '{"relevant": [1]}'])
    agent, llm, _ = build(
        [
            (PLANNER, plan(("policies", "stuff about giving money back"))),
            (GRADER, lambda s, m: next(grades)),
            (REWRITE, lambda s, m: "refund method cash purchases"),
            (SYNTH, lambda s, m: "Cash refunds above 200 USD are paid by store credit or check [1]."),
        ]
    )
    r = agent.run("how do we give money back?")
    nodes = [s["node"] for s in r.steps]
    assert nodes == [
        "plan",
        "gather",
        "grade",
        "rewrite",
        "gather",
        "grade",
        "synthesize",
        "verify",
        "finalize",
    ]
    assert r.grounded and llm.calls.count(REWRITE) == 1


def test_rewrite_is_bounded():
    agent, llm, _ = build(
        [
            (PLANNER, plan(("policies", "zzz"))),
            (GRADER, lambda s, m: '{"relevant": []}'),
            (REWRITE, lambda s, m: "still nothing"),
            (SYNTH, lambda s, m: "unused"),
        ]
    )
    r = agent.run("zzz")
    assert llm.calls.count(REWRITE) == 1 and not r.grounded and r.answer == REFUSAL


def test_one_revision_then_refuse():
    drafts = iter(["Electronics have 15 days.", "Electronics have 15 days [7]."])
    agent, llm, _ = build(
        [
            (PLANNER, plan(("policies", "electronics return window"))),
            (GRADER, lambda s, m: '{"relevant": [1]}'),
            (SYNTH, lambda s, m: next(drafts)),
        ]
    )
    r = agent.run("electronics return window?")
    assert llm.calls.count(SYNTH) == 2 and not r.grounded and r.answer == REFUSAL
    verify = [s for s in r.steps if s["node"] == "verify"]
    assert verify[-1]["invalid"] == [7]


def test_revision_can_fix_the_draft():
    drafts = iter(["Electronics have 15 days.", "Electronics have 15 days [1]."])
    agent, _llm, _ = build(
        [
            (PLANNER, plan(("policies", "electronics return window"))),
            (GRADER, lambda s, m: '{"relevant": [1]}'),
            (SYNTH, lambda s, m: next(drafts)),
        ]
    )
    r = agent.run("electronics return window?")
    assert r.grounded and r.answer.endswith("[1].")


def test_unparseable_plan_falls_back_to_keyword_router():
    agent, _, _ = build(
        [
            (PLANNER, lambda s, m: "Sure! I'd search policies."),
            (GRADER, lambda s, m: '{"relevant": [1, 2]}'),
            (SYNTH, lambda s, m: "Gold members are exempt from the restocking fee [1]."),
        ]
    )
    r = agent.run("What is the restocking fee policy for Gold members?")
    assert r.routes == ["policies"] and r.steps[0]["planner"] == "heuristic" and r.grounded


def test_off_topic_gets_a_reply_and_no_tools():
    agent, _llm, ex = build([(PLANNER, lambda s, m: '{"tasks": [], "reply": "I only cover retail."}')])
    r = agent.run("What's the capital of France?")
    assert r.answer == "I only cover retail." and r.evidence == [] and ex.queries == []


class RecordingRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, k=5, source=None, ticker=None):
        self.calls.append((source, ticker))
        return []


def test_filing_tasks_are_scoped_to_the_resolved_ticker():
    rec = RecordingRetriever()
    agent, _, _ = build(
        [
            (PLANNER, plan(("filings", "Best Buy returns policy", "Best Buy"))),
            (REWRITE, lambda s, m: "returns"),
            (SYNTH, lambda s, m: "x"),
        ],
        retriever=rec,
    )
    agent.run("What does Best Buy say about returns?")
    assert rec.calls == [("filing", "BBY"), ("filing", "BBY")]  # original + one rewrite


def test_filing_meta_calls_the_uc_function():
    ex = FakeExecutor(lambda sql: QueryResult(["ticker", "filing_date"], [["WMT", "2026-03-14"]]))
    agent, _, _ = build(
        [
            (PLANNER, plan(("filing_meta", "Walmart latest 10-K date", "WMT"))),
            (SYNTH, lambda s, m: "Walmart filed its latest 10-K on 2026-03-14 [1]."),
        ],
        executor=ex,
    )
    r = agent.run("When did Walmart file its latest 10-K?")
    assert ex.queries == ["SELECT * FROM workspace.retail_dev.filing_lookup('WMT')"]
    assert r.grounded and r.evidence[0].kind == "filing_meta"


def test_helpers():
    assert parse_json('Here you go:\n```json\n{"tasks": [{"route": "x"}]}\n```') == {
        "tasks": [{"route": "x"}]
    }
    with pytest.raises(ValueError):
        parse_json("no json")
    assert resolve_ticker(None, "What does Home Depot say?") == "HD"
    assert resolve_ticker("tgt") == "TGT" and resolve_ticker("NOPE", "nothing") is None
    assert [t.route for t in heuristic_plan("When did Costco file its 10-K?")] == ["filing_meta"]
    assert [t.route for t in heuristic_plan("Top products by revenue and our discount policy")] == [
        "analytics",
        "policies",
    ]
