import json
from datetime import date

from retail_insights.agent.core import RetailAgent
from retail_insights.agent.executor import QueryResult
from retail_insights.agent.fakes import FakeExecutor, ScriptedLLM, call, say
from retail_insights.agent.tools import Toolbox
from retail_insights.evals.runner import answer_matches, run_offline


def toolbox(responder=None):
    ex = FakeExecutor(responder or (lambda sql: QueryResult(["net_revenue"], [["1234.50"]])))
    return Toolbox(ex, "workspace", "retail_dev"), ex


def test_sales_kpis_builds_scoped_sql():
    tb, ex = toolbox()
    out = tb.call(
        "get_sales_kpis",
        {
            "start_date": "2026-01-01",
            "end_date": "2026-01-07",
            "group_by": "region",
            "region": "West",
        },
    )
    assert out["rows"] == [{"net_revenue": "1234.50"}]
    sql = ex.queries[-1]
    assert "workspace.retail_dev.gold_daily_sales" in sql
    assert "DATE'2026-01-01' AND DATE'2026-01-07'" in sql and "region = 'West'" in sql


def test_tool_errors_are_returned_not_raised():
    tb, ex = toolbox()
    assert "YYYY-MM-DD" in tb.call("get_sales_kpis", {"start_date": "last week", "end_date": "x"})["error"]
    assert "unknown tool" in tb.call("drop_everything", {})["error"]
    assert "only SELECT" in tb.call("run_sql", {"query": "DROP TABLE silver_sales"})["error"]
    assert (
        "region"
        in tb.call(
            "get_sales_kpis",
            {"start_date": "2026-01-01", "end_date": "2026-01-02", "region": "Mars"},
        )["error"]
    )
    assert ex.queries == []  # nothing unsafe reached the warehouse


def test_describe_table_rejects_unknown_tables():
    tb, ex = toolbox()
    assert "unknown table" in tb.call("describe_table", {"table": "users; DROP"})["error"]
    assert ex.queries == []


def test_warehouse_failure_becomes_tool_error():
    def boom(sql):
        raise RuntimeError("warehouse is stopped")

    tb, _ = toolbox(boom)
    assert "warehouse is stopped" in tb.call("top_products", {"metric": "net_revenue"})["error"]


def test_agent_loop_calls_tool_then_answers():
    tb, _ex = toolbox()
    llm = ScriptedLLM(
        [
            call("get_sales_kpis", start_date="2026-01-01", end_date="2026-01-07"),
            say("Net revenue was $1,234.50."),
        ]
    )
    result = RetailAgent(llm, tb, today=date(2026, 1, 8)).run("revenue last week?")

    assert result.answer == "Net revenue was $1,234.50."
    assert result.tools_used == ["get_sales_kpis"] and result.stopped_reason == "answered"
    # the second LLM call saw the tool result
    tool_msg = llm.seen[1][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["rows"]
    assert "2026-01-08" in llm.seen[0][0]["content"]  # today's date is in the system prompt


def test_agent_recovers_from_bad_call():
    tb, _ = toolbox()
    llm = ScriptedLLM(
        [
            call("run_sql", query="DELETE FROM silver_sales"),
            call("top_products", metric="net_revenue", limit=1),
            say("Top product is Earbuds."),
        ]
    )
    result = RetailAgent(llm, tb).run("best product?")
    assert [s.ok for s in result.steps] == [False, True]
    assert result.answer == "Top product is Earbuds."


def test_agent_stops_at_step_limit():
    tb, _ = toolbox()
    llm = ScriptedLLM([call("list_tables")])  # never answers
    result = RetailAgent(llm, tb, max_steps=3).run("loop forever")
    assert result.stopped_reason == "max_steps" and len(result.steps) == 3


def test_answer_matching():
    assert answer_matches("Net revenue was $12,345.67 last week", "12345.67", "number")
    assert answer_matches("Margin was 23.4%", 0.234, "number")
    assert not answer_matches("Net revenue was $99", "12345.67", "number")
    assert answer_matches("The West region led", "West", "text")


def test_offline_eval_is_green():
    report = run_offline("workspace", "retail_dev")
    assert report["pass_rate"] == 1.0 and report["routing_rate"] == 1.0
    assert report["red_team"]["leaks"] == []
