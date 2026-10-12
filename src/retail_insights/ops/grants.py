"""Give the Databricks App's service principal read access to the retail schema.

Every Databricks App runs as its own service principal. Binding a SQL warehouse
as an app resource lets the app *use* the warehouse, but Unity Catalog still
decides which tables it may read. This script grants exactly what the agents
need (USE CATALOG, USE SCHEMA, SELECT and EXECUTE on the schema, CAN_USE on the
Vector Search endpoint) and nothing that can write.

Run after `databricks bundle deploy` (the CD workflow does this for you):
    uv run retail-grant-app --app-name retail-ai-dev --catalog workspace \\
        --schema retail_dev
"""

from __future__ import annotations

import argparse
import sys

from retail_insights.config import VECTOR_SEARCH_ENDPOINT


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app-name", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--warehouse-id", default="")
    p.add_argument(
        "--warehouse-name",
        default="Serverless Starter Warehouse",
        help="used to find the warehouse when --warehouse-id is not given",
    )
    p.add_argument("--vs-endpoint", default=VECTOR_SEARCH_ENDPOINT)
    args = p.parse_args(argv)

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState

    ws = WorkspaceClient()
    # Development mode may prefix resource names, so fall back to a suffix match.
    app = next((a for a in ws.apps.list() if a.name == args.app_name), None) or next(
        (a for a in ws.apps.list() if a.name.endswith(args.app_name)), None
    )
    if app is None:
        sys.exit(f"app {args.app_name!r} not found; run `databricks bundle deploy` first")
    principal = app.service_principal_client_id
    if not principal:
        sys.exit(f"app {args.app_name} has no service principal yet; deploy it first")

    warehouse_id = args.warehouse_id or next(
        (w.id for w in ws.warehouses.list() if w.name == args.warehouse_name), None
    )
    if not warehouse_id:
        sys.exit(f"no warehouse id given and none named {args.warehouse_name!r}")

    statements = [
        # The ETL setup task creates this too; doing it here lets grants run before the first ETL run.
        f"CREATE SCHEMA IF NOT EXISTS `{args.catalog}`.`{args.schema}`",
        f"GRANT USE CATALOG ON CATALOG `{args.catalog}` TO `{principal}`",
        f"GRANT USE SCHEMA ON SCHEMA `{args.catalog}`.`{args.schema}` TO `{principal}`",
        # SELECT covers the gold/silver tables and the Vector Search index (a UC securable).
        f"GRANT SELECT ON SCHEMA `{args.catalog}`.`{args.schema}` TO `{principal}`",
        # EXECUTE lets the agent call UC functions such as filing_lookup().
        f"GRANT EXECUTE ON SCHEMA `{args.catalog}`.`{args.schema}` TO `{principal}`",
    ]
    for sql in statements:
        r = ws.statement_execution.execute_statement(
            statement=sql, warehouse_id=warehouse_id, wait_timeout="30s"
        )
        if r.status.state != StatementState.SUCCEEDED:
            msg = r.status.error.message if r.status.error else r.status.state
            sys.exit(f"failed: {sql}\n{msg}")
        print(f"ok: {sql}")

    _grant_vector_search_endpoint(ws, args.vs_endpoint, principal)


def _grant_vector_search_endpoint(ws, endpoint_name: str, principal: str) -> None:
    """CAN_USE on the Vector Search endpoint. Best effort: the endpoint only exists
    after the docs pipeline has run once, and some workspaces manage this centrally."""
    from databricks.sdk.service.vectorsearch import (
        VectorSearchEndpointAccessControlRequest,
        VectorSearchEndpointPermissionLevel,
    )

    try:
        ep = ws.vector_search_endpoints.get_endpoint(endpoint_name)
        ws.vector_search_endpoints.update_permissions(
            ep.id,
            access_control_list=[
                VectorSearchEndpointAccessControlRequest(
                    service_principal_name=principal,
                    permission_level=VectorSearchEndpointPermissionLevel.CAN_USE,
                )
            ],
        )
        print(f"ok: CAN_USE on vector search endpoint {endpoint_name}")
    except Exception as e:
        print(f"skipped vector search endpoint permission ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
