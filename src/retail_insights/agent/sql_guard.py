"""Guardrails for model-written SQL.

The agent may write its own SELECT queries. Before anything reaches the SQL
warehouse we parse it with sqlglot (Databricks dialect) and enforce:

  * exactly one statement, and it must be a read query (SELECT / UNION / CTE)
  * no DML or DDL anywhere in the tree
  * only tables from the allowed schema (and allowed table list, if given)
  * no risky table/scalar functions (reading files, HTTP calls, secrets)
  * a row LIMIT, added or tightened if needed

Parsing beats regex here: comments, string literals and nested subqueries are
all handled by a real grammar instead of fragile pattern matching.

This is defence in depth, not the only control. The app's service principal
should only have SELECT on the schema, so even a guard bug cannot write data.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

BLOCKED_FUNCTIONS = {
    "read_files",
    "http_request",
    "ai_query",
    "secret",
    "try_secret",
    "read_kafka",
    "read_kinesis",
    "java_method",
    "reflect",
}
WRITE_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Command,
    exp.Use,
    exp.Set,
)


class UnsafeSQLError(ValueError):
    """Raised when a query violates the guard. The message is shown to the model
    so it can correct itself."""


@dataclass(frozen=True)
class SQLPolicy:
    catalog: str
    schema: str
    allowed_tables: frozenset[str] | None = None
    max_rows: int = 200


def guard_sql(sql: str, policy: SQLPolicy) -> str:
    """Return a safe, normalised version of `sql` or raise UnsafeSQLError."""
    if not sql or not sql.strip():
        raise UnsafeSQLError("empty query")
    try:
        statements = [s for s in sqlglot.parse(sql, read="databricks") if s is not None]
    except ParseError as e:
        raise UnsafeSQLError(f"could not parse SQL: {e.errors[0]['description'] if e.errors else e}") from e

    if len(statements) != 1:
        raise UnsafeSQLError("exactly one statement is allowed")
    tree = statements[0]

    if not isinstance(tree, exp.Select | exp.SetOperation):
        raise UnsafeSQLError(f"only SELECT queries are allowed, got {tree.key.upper()}")
    for node in tree.walk():
        if isinstance(node, WRITE_NODES):
            raise UnsafeSQLError(f"{node.key.upper()} is not allowed")

    for fn in tree.find_all(exp.Func):
        name = (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()
        if name in BLOCKED_FUNCTIONS:
            raise UnsafeSQLError(f"function {name}() is not allowed")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name, db, catalog = table.name.lower(), table.db.lower(), table.catalog.lower()
        if not name:
            raise UnsafeSQLError("table-valued functions are not allowed")
        if not db and name in cte_names:
            continue
        if catalog and catalog != policy.catalog.lower():
            raise UnsafeSQLError(f"catalog {catalog} is not allowed")
        if db and db != policy.schema.lower():
            raise UnsafeSQLError(f"schema {db} is not allowed; use tables in {policy.schema}")
        if policy.allowed_tables is not None and name not in policy.allowed_tables:
            raise UnsafeSQLError(f"table {name} is not available. Call list_tables first.")

    return _enforce_limit(tree, policy.max_rows).sql(dialect="databricks")


def _enforce_limit(tree: exp.Expression, max_rows: int) -> exp.Expression:
    if isinstance(tree, exp.SetOperation):
        return exp.select("*").from_(tree.subquery("q")).limit(max_rows)
    limit = tree.args.get("limit")
    if limit is None:
        return tree.limit(max_rows)
    try:
        current = int(limit.expression.name)
    except (AttributeError, TypeError, ValueError):
        return tree.limit(max_rows)
    return tree.limit(min(current, max_rows))
