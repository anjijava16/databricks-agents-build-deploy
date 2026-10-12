"""Idempotent Vector Search provisioning used by the `vector_index` job task."""

from __future__ import annotations

import logging
import time

log = logging.getLogger("retail_insights.vs")


def ensure_endpoint(ws, name: str, timeout_s: int = 1800) -> None:
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.vectorsearch import EndpointType

    try:
        ep = ws.vector_search_endpoints.get_endpoint(name)
        state = ep.endpoint_status.state.value if ep.endpoint_status else "UNKNOWN"
        log.info("endpoint %s exists (%s)", name, state)
        deadline = time.monotonic() + timeout_s
        while state != "ONLINE":
            if time.monotonic() > deadline:
                raise TimeoutError(f"endpoint {name} not ONLINE after {timeout_s}s")
            time.sleep(20)
            ep = ws.vector_search_endpoints.get_endpoint(name)
            state = ep.endpoint_status.state.value if ep.endpoint_status else "UNKNOWN"
    except NotFound:
        log.info("creating endpoint %s (takes several minutes the first time)", name)
        ws.vector_search_endpoints.create_endpoint_and_wait(name, endpoint_type=EndpointType.STANDARD)


def ensure_index(
    ws, index_name: str, endpoint: str, source_table: str, embedding_endpoint: str, columns: list[str]
) -> bool:
    """Create a Delta Sync index with Databricks-managed embeddings if missing.
    Returns True when it was created (creation triggers the first sync)."""
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.vectorsearch import (
        DeltaSyncVectorIndexSpecRequest,
        EmbeddingSourceColumn,
        PipelineType,
        VectorIndexType,
    )

    try:
        ws.vector_search_indexes.get_index(index_name)
        log.info("index %s exists", index_name)
        return False
    except NotFound:
        pass
    log.info("creating index %s on %s", index_name, source_table)
    ws.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=endpoint,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=source_table,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[
                EmbeddingSourceColumn(name="embed_text", embedding_model_endpoint_name=embedding_endpoint)
            ],
            columns_to_sync=columns,
        ),
    )
    return True


def wait_until_synced(ws, index_name: str, expected_rows: int, timeout_s: int = 2400) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        idx = ws.vector_search_indexes.get_index(index_name)
        status = idx.status
        ready = bool(status and status.ready)
        rows = (status.indexed_row_count or 0) if status else 0
        log.info("index %s ready=%s rows=%s/%s", index_name, ready, rows, expected_rows)
        if ready and rows >= expected_rows:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"index {index_name} not synced: {rows}/{expected_rows} rows")
        time.sleep(30)
