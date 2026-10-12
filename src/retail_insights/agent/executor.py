"""Runs SQL against a Databricks SQL warehouse via the Statement Execution API.

Using the SDK (instead of a JDBC/ODBC driver) means the same code authenticates
everywhere: a PAT or CLI profile on a laptop, OAuth M2M in GitHub Actions, and
the app's own service principal inside Databricks Apps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    truncated: bool = False

    def as_records(self) -> list[dict]:
        return [dict(zip(self.columns, r, strict=False)) for r in self.rows]


class SqlExecutor(Protocol):
    def run(self, sql: str) -> QueryResult: ...


@dataclass
class DatabricksSqlExecutor:
    warehouse_id: str
    catalog: str
    schema: str
    timeout_s: int = 60
    _client: object = field(default=None, repr=False)

    def _ws(self):
        if self._client is None:
            from databricks.sdk import WorkspaceClient

            self._client = WorkspaceClient()
        return self._client

    def run(self, sql: str) -> QueryResult:
        from databricks.sdk.service.sql import StatementState

        ws = self._ws()
        resp = ws.statement_execution.execute_statement(
            statement=sql,
            warehouse_id=self.warehouse_id,
            catalog=self.catalog,
            schema=self.schema,
            wait_timeout="30s",
        )
        deadline = time.monotonic() + self.timeout_s
        while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
            if time.monotonic() > deadline:
                ws.statement_execution.cancel_execution(resp.statement_id)
                raise TimeoutError(f"query exceeded {self.timeout_s}s")
            time.sleep(1)
            resp = ws.statement_execution.get_statement(resp.statement_id)

        if resp.status.state != StatementState.SUCCEEDED:
            msg = resp.status.error.message if resp.status.error else resp.status.state
            raise RuntimeError(f"query failed: {msg}")

        cols = [c.name for c in resp.manifest.schema.columns] if resp.manifest else []
        rows = (resp.result.data_array or []) if resp.result else []
        truncated = bool(resp.manifest and resp.manifest.truncated)
        return QueryResult(columns=cols, rows=rows, truncated=truncated)
