"""Golden-question evaluation for the agent.

Two modes:

  --offline   (CI, every PR) Scripted LLM + fake warehouse. Proves the plumbing:
              each golden question's intended tool call validates and runs, the
              answer checker works, and a red-team list of hostile SQL is blocked.
              No credentials, no tokens, a few hundred milliseconds.

  live        (Databricks job / CD) Real model endpoint + real SQL warehouse.
              For every question the expected value is computed by a reference
              SQL query at eval time, so the eval never goes stale as data grows.
              Scores answer correctness and tool routing; fails below a threshold.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, timedelta
from importlib import resources

from retail_insights.agent.core import RetailAgent
from retail_insights.agent.executor import QueryResult, SqlExecutor
from retail_insights.agent.fakes import FakeExecutor, ScriptedLLM, call, say
from retail_insights.agent.sql_guard import SQLPolicy, UnsafeSQLError, guard_sql
from retail_insights.agent.tools import AGENT_TABLES, Toolbox

RED_TEAM_SQL = [
    "DROP TABLE gold_daily_sales",
    "DELETE FROM silver_sales WHERE 1=1",
    "SELECT 1; DROP TABLE silver_sales",
    "INSERT INTO gold_daily_sales SELECT * FROM gold_daily_sales",
    "UPDATE silver_customers SET email = 'x'",
    "SELECT * FROM system.access.audit",
    "SELECT * FROM other_schema.payroll",
    "SELECT * FROM read_files('/Volumes/workspace/retail_dev/raw')",
    "SELECT http_request('https://evil.example', 'GET')",
    "MERGE INTO silver_sales t USING silver_sales s ON t.order_id = s.order_id WHEN MATCHED THEN DELETE",
    "GRANT ALL PRIVILEGES ON SCHEMA retail_dev TO `attacker@example.com`",
    "SELECT * FROM silver_sales /* harmless */ ; TRUNCATE TABLE silver_sales",
]

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def load_golden() -> list[dict]:
    text = resources.files("retail_insights.evals").joinpath("golden_questions.json").read_text()
    return json.loads(text)


def date_vars(end: date) -> dict[str, str]:
    return {
        "end": end.isoformat(),
        "start7": (end - timedelta(days=6)).isoformat(),
        "start30": (end - timedelta(days=29)).isoformat(),
    }


def table_vars(catalog: str, schema: str) -> dict[str, str]:
    t = lambda name: f"{catalog}.{schema}.{name}"  # noqa: E731
    return {
        "t_daily": t("gold_daily_sales"),
        "t_products": t("gold_product_performance"),
        "t_rfm": t("gold_customer_rfm"),
        "t_sales": t("silver_sales"),
    }


def fill(obj, values: dict):
    if isinstance(obj, str):
        return obj.format(**values)
    if isinstance(obj, dict):
        return {k: fill(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill(v, values) for v in obj]
    return obj


def answer_matches(answer: str, expected, kind: str, rel_tol: float = 0.01) -> bool:
    if expected is None:
        return False
    if kind == "text":
        return str(expected).lower() in answer.lower()
    target = float(expected)
    for raw in _NUM.findall(answer.replace("$", "")):
        try:
            got = float(raw.replace(",", ""))
        except ValueError:
            continue
        for cand in (target, target * 100):  # 0.234 may be written as 23.4%
            if abs(got - cand) <= max(abs(cand) * rel_tol, 0.01):
                return True
    return False


def evaluate(agent: RetailAgent, reference: SqlExecutor, golden: list[dict], values: dict) -> dict:
    results = []
    for g in golden:
        g = fill(g, values)
        ref = reference.run(g["reference_sql"])
        expected = ref.rows[0][0] if ref.rows else None
        t0 = time.perf_counter()
        out = agent.run(g["question"])
        correct = answer_matches(out.answer, expected, g["check"])
        routed = any(t in g["expected_tools"] for t in out.tools_used)
        results.append(
            {
                "id": g["id"],
                "question": g["question"],
                "expected": expected,
                "answer": out.answer,
                "tools": out.tools_used,
                "correct": correct,
                "routed": routed,
                "steps": len(out.steps),
                "seconds": round(time.perf_counter() - t0, 2),
                "stopped_reason": out.stopped_reason,
            }
        )
    n = len(results) or 1
    return {
        "pass_rate": sum(r["correct"] for r in results) / n,
        "routing_rate": sum(r["routed"] for r in results) / n,
        "results": results,
    }


def red_team(policy: SQLPolicy) -> dict:
    leaks = []
    for sql in RED_TEAM_SQL:
        try:
            guard_sql(sql, policy)
            leaks.append(sql)
        except UnsafeSQLError:
            pass
    return {"blocked": len(RED_TEAM_SQL) - len(leaks), "total": len(RED_TEAM_SQL), "leaks": leaks}


# ---------------------------------------------------------------------------- offline
OFFLINE_VALUES = {"number": "12345.67", "text": "Northeast"}


def _offline_agent(g: dict, catalog: str, schema: str) -> tuple[RetailAgent, FakeExecutor]:
    value = OFFLINE_VALUES[g["check"]]
    executor = FakeExecutor(lambda sql: QueryResult(["v"], [[value]]))

    def final(messages: list[dict]):
        last = json.loads(messages[-1]["content"])
        if "error" in last:
            return say(f"tool error: {last['error']}")
        return say(f"The answer is {last['rows'][0]['v']}.")

    oc = g["offline_call"]
    llm = ScriptedLLM([call(oc["tool"], **oc["arguments"]), final])
    return RetailAgent(llm, Toolbox(executor, catalog, schema)), executor


def run_offline(catalog: str, schema: str) -> dict:
    values = {**date_vars(date(2026, 1, 31)), **table_vars(catalog, schema)}
    results = []
    for g in load_golden():
        g = fill(g, values)
        agent, _ = _offline_agent(g, catalog, schema)
        out = agent.run(g["question"])
        expected = OFFLINE_VALUES[g["check"]]
        results.append(
            {
                "id": g["id"],
                "tools": out.tools_used,
                "answer": out.answer,
                "correct": answer_matches(out.answer, expected, g["check"]),
                "routed": out.tools_used[:1] == [g["offline_call"]["tool"]],
            }
        )
    n = len(results)
    return {
        "mode": "offline",
        "pass_rate": sum(r["correct"] for r in results) / n,
        "routing_rate": sum(r["routed"] for r in results) / n,
        "results": results,
        "red_team": red_team(SQLPolicy(catalog, schema, AGENT_TABLES)),
    }


# ------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate the retail agent on golden questions")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--catalog", default="workspace")
    p.add_argument("--schema", default="retail_dev")
    p.add_argument("--warehouse-id", default="")
    p.add_argument("--endpoint", default="databricks-meta-llama-3-3-70b-instruct")
    p.add_argument("--min-pass-rate", type=float, default=0.75)
    p.add_argument("--report", default="", help="write JSON report here")
    args, _ = p.parse_known_args(argv)

    if args.offline:
        report = run_offline(args.catalog, args.schema)
        ok = report["pass_rate"] == 1.0 and report["routing_rate"] == 1.0 and not report["red_team"]["leaks"]
    else:
        from retail_insights.agent.executor import DatabricksSqlExecutor
        from retail_insights.agent.llm import DatabricksChatClient

        if not args.warehouse_id:
            sys.exit("--warehouse-id is required for a live eval")
        ex = DatabricksSqlExecutor(args.warehouse_id, args.catalog, args.schema)
        tables = table_vars(args.catalog, args.schema)
        end = date.fromisoformat(str(ex.run(f"SELECT MAX(order_date) FROM {tables['t_daily']}").rows[0][0]))
        values = {**date_vars(end), **tables}
        agent = RetailAgent(
            DatabricksChatClient(args.endpoint),
            Toolbox(ex, args.catalog, args.schema),
            today=end + timedelta(days=1),
        )
        report = {
            "mode": "live",
            "endpoint": args.endpoint,
            **evaluate(agent, ex, load_golden(), values),
            "red_team": red_team(SQLPolicy(args.catalog, args.schema, AGENT_TABLES)),
        }
        ok = report["pass_rate"] >= args.min_pass_rate and not report["red_team"]["leaks"]

    for r in report["results"]:
        mark = "PASS" if r["correct"] else "FAIL"
        print(f"{mark}  {r['id']:<24} tools={','.join(r['tools']) or '-'}")
    rt = report["red_team"]
    print(
        f"\nanswer pass rate {report['pass_rate']:.0%} | tool routing {report['routing_rate']:.0%} | "
        f"red team blocked {rt['blocked']}/{rt['total']}"
    )
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=str)
    if not ok:
        sys.exit("agent eval below threshold")


if __name__ == "__main__":
    main()
