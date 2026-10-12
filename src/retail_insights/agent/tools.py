"""The agent's tools.

Two kinds, on purpose:

  * Curated tools (get_sales_kpis, top_products, customer_segments) wrap
    parameterised SQL written by a human. They are fast, correct by
    construction and cover the questions people ask most.
  * Exploratory tools (list_tables, describe_table, run_sql) let the model
    answer anything else, with run_sql going through the SQL guard.

The system prompt tells the model to prefer curated tools. In practice this
is the single biggest lever on answer accuracy for analytics agents.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from retail_insights.agent.executor import QueryResult, SqlExecutor
from retail_insights.agent.sql_guard import SQLPolicy, UnsafeSQLError, guard_sql

# silver_customers is deliberately absent: it holds names and emails, which the data
# access policy (POL-AI-010) keeps away from the assistant. gold_customer_rfm has the
# customer attributes it may use (id, tier, home state).
AGENT_TABLES = frozenset(
    {
        "gold_daily_sales",
        "gold_product_performance",
        "gold_customer_rfm",
        "silver_products",
        "silver_stores",
        "silver_sales",
    }
)
METRICS = ("net_revenue", "units_sold", "gross_margin", "margin_pct", "return_rate")
REGIONS = ("Northeast", "South", "Midwest", "West")
SEGMENTS = ("Champions", "Loyal", "New or Promising", "At Risk", "Needs Attention")
MAX_RESULT_CHARS = 6000
_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


class ToolError(ValueError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], dict]

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _date(value: str, field: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as e:
        raise ToolError(f"{field} must be YYYY-MM-DD, got {value!r}") from e


def _result_payload(sql: str, res: QueryResult) -> dict:
    records = res.as_records()
    text = json.dumps(records, default=str)
    clipped = len(text) > MAX_RESULT_CHARS
    if clipped:
        # keep whole rows, not a half-cut JSON string
        while records and len(json.dumps(records, default=str)) > MAX_RESULT_CHARS:
            records = records[: max(1, len(records) // 2)] if len(records) > 1 else []
    return {
        "sql": sql,
        "row_count": len(res.rows),
        "rows": records,
        "truncated": clipped or res.truncated,
    }


class Toolbox:
    def __init__(self, executor: SqlExecutor, catalog: str, schema: str, max_rows: int = 200):
        self.executor = executor
        self.catalog, self.schema = catalog, schema
        self.policy = SQLPolicy(catalog, schema, AGENT_TABLES, max_rows)
        self.tools: dict[str, Tool] = {t.name: t for t in self._build()}

    # -- plumbing -------------------------------------------------------------
    def fq(self, table: str) -> str:
        return f"{self.catalog}.{self.schema}.{table}"

    def _query(self, sql: str) -> dict:
        return _result_payload(sql, self.executor.run(sql))

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def call(self, name: str, args: dict) -> dict:
        """Never raises: errors go back to the model as data so it can recover."""
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}. Available: {sorted(self.tools)}"}
        if "_raw" in args:
            return {"error": "arguments were not valid JSON"}
        try:
            return tool.handler(args)
        except (ToolError, UnsafeSQLError) as e:
            return {"error": str(e)}
        except Exception as e:  # warehouse/network errors
            return {"error": f"{type(e).__name__}: {e}"}

    # -- tools ----------------------------------------------------------------
    def _build(self) -> list[Tool]:
        return [
            Tool(
                "get_sales_kpis",
                "Revenue KPIs for a date range: orders, units, net revenue, gross margin, "
                "average order value, returns. Optional breakdown by region, channel, store or day.",
                {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "YYYY-MM-DD inclusive"},
                        "end_date": {"type": "string", "description": "YYYY-MM-DD inclusive"},
                        "group_by": {
                            "type": "string",
                            "enum": ["none", "region", "channel", "store", "day"],
                        },
                        "region": {"type": "string", "enum": list(REGIONS)},
                    },
                    "required": ["start_date", "end_date"],
                },
                self._sales_kpis,
            ),
            Tool(
                "top_products",
                "Rank products by a metric (lifetime). Optionally filter to one category.",
                {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "enum": list(METRICS)},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                        "ascending": {"type": "boolean", "description": "true = worst first"},
                        "category": {"type": "string"},
                    },
                    "required": ["metric"],
                },
                self._top_products,
            ),
            Tool(
                "customer_segments",
                "Customer counts, average spend and average recency per RFM segment, "
                "or list customers in one segment.",
                {
                    "type": "object",
                    "properties": {
                        "segment": {"type": "string", "enum": list(SEGMENTS)},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                },
                self._segments,
            ),
            Tool(
                "list_tables",
                "List the tables you can query, with their descriptions.",
                {"type": "object", "properties": {}},
                self._list_tables,
            ),
            Tool(
                "describe_table",
                "Columns, types and descriptions for one table.",
                {
                    "type": "object",
                    "properties": {"table": {"type": "string"}},
                    "required": ["table"],
                },
                self._describe,
            ),
            Tool(
                "run_sql",
                "Run ONE read-only Databricks SQL SELECT over the listed tables. Use only when "
                "the other tools cannot answer. Unqualified table names are fine.",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                self._run_sql,
            ),
        ]

    def _sales_kpis(self, a: dict) -> dict:
        start, end = _date(a.get("start_date"), "start_date"), _date(a.get("end_date"), "end_date")
        if start > end:
            raise ToolError("start_date is after end_date")
        dims = {
            "none": [],
            "region": ["region"],
            "channel": ["channel"],
            "store": ["store_id", "store_name"],
            "day": ["order_date"],
        }
        group = a.get("group_by", "none")
        if group not in dims:
            raise ToolError(f"group_by must be one of {list(dims)}")
        where = [f"order_date BETWEEN DATE'{start}' AND DATE'{end}'"]
        if a.get("region"):
            if a["region"] not in REGIONS:
                raise ToolError(f"region must be one of {REGIONS}")
            where.append(f"region = '{a['region']}'")
        cols = dims[group]
        select_dims = ", ".join(cols) + ", " if cols else ""
        group_sql = f"GROUP BY {', '.join(cols)} ORDER BY net_revenue DESC" if cols else ""
        sql = (
            f"SELECT {select_dims}SUM(orders) AS orders, SUM(units) AS units, "
            "ROUND(SUM(net_revenue), 2) AS net_revenue, ROUND(SUM(gross_margin), 2) AS gross_margin, "
            "ROUND(SUM(net_revenue) / NULLIF(SUM(orders), 0), 2) AS avg_order_value, "
            "ROUND(SUM(returns_amount), 2) AS returns_amount, "
            "ROUND(SUM(discounts), 2) AS discounts "
            f"FROM {self.fq('gold_daily_sales')} WHERE {' AND '.join(where)} {group_sql}"
        ).strip()
        return self._query(sql)

    def _top_products(self, a: dict) -> dict:
        metric = a.get("metric")
        if metric not in METRICS:
            raise ToolError(f"metric must be one of {METRICS}")
        limit = max(1, min(int(a.get("limit", 10)), 50))
        order = "ASC" if a.get("ascending") else "DESC"
        where = ""
        if a.get("category"):
            cat = str(a["category"]).replace("'", "")
            where = f"WHERE lower(category) = lower('{cat}')"
        sql = (
            f"SELECT product_id, product_name, category, units_sold, net_revenue, gross_margin, "
            f"margin_pct, return_rate FROM {self.fq('gold_product_performance')} {where} "
            f"ORDER BY {metric} {order} LIMIT {limit}"
        )
        return self._query(sql)

    def _segments(self, a: dict) -> dict:
        seg = a.get("segment")
        if seg:
            if seg not in SEGMENTS:
                raise ToolError(f"segment must be one of {SEGMENTS}")
            limit = max(1, min(int(a.get("limit", 20)), 50))
            sql = (
                f"SELECT customer_id, loyalty_tier, home_state, recency_days, frequency, monetary "
                f"FROM {self.fq('gold_customer_rfm')} WHERE segment = '{seg}' "
                f"ORDER BY monetary DESC LIMIT {limit}"
            )
        else:
            sql = (
                f"SELECT segment, COUNT(*) AS customers, ROUND(AVG(monetary), 2) AS avg_spend, "
                f"ROUND(AVG(recency_days), 1) AS avg_recency_days, "
                f"ROUND(AVG(frequency), 2) AS avg_orders "
                f"FROM {self.fq('gold_customer_rfm')} GROUP BY segment ORDER BY customers DESC"
            )
        return self._query(sql)

    def _list_tables(self, _: dict) -> dict:
        names = ", ".join(f"'{t}'" for t in sorted(AGENT_TABLES))
        sql = (
            f"SELECT table_name, comment FROM {self.catalog}.information_schema.tables "
            f"WHERE table_schema = '{self.schema}' AND table_name IN ({names}) ORDER BY table_name"
        )
        return self._query(sql)

    def _describe(self, a: dict) -> dict:
        table = str(a.get("table", "")).split(".")[-1].lower()
        if not _IDENT.match(table) or table not in AGENT_TABLES:
            raise ToolError(f"unknown table {a.get('table')!r}. Call list_tables.")
        res = self.executor.run(f"DESCRIBE TABLE {self.fq(table)}")
        rows = [r for r in res.as_records() if r.get("col_name") and not r["col_name"].startswith("#")]
        return {"table": table, "columns": rows}

    def _run_sql(self, a: dict) -> dict:
        safe = guard_sql(a.get("query", ""), self.policy)
        return self._query(safe)
