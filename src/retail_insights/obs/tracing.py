"""MLflow Tracing, made optional.

The agents call `span(...)` around every step that matters: retrieval, LLM
calls, tool calls, routing decisions. When tracing is switched on (eval and
serving jobs, or the app with RETAIL_TRACING=on), each call becomes an MLflow
span; otherwise it is a no-op, so unit tests and laptops never need MLflow
or a tracking server.

Retriever spans record the retrieved passages as `mlflow.entities.Document`s:
that is the shape the RetrievalGroundedness / RetrievalRelevance judges read.

On Databricks, traces go to Unity Catalog Delta tables (an experiment bound to
a UnityCatalog trace location) so they are governed and queryable with SQL.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass

log = logging.getLogger("retail_insights.tracing")
_ENABLED = os.environ.get("RETAIL_TRACING", "off").lower() == "on"


def enabled() -> bool:
    return _ENABLED


def setup_tracing(
    experiment_name: str,
    catalog: str | None = None,
    schema: str | None = None,
    warehouse_id: str | None = None,
) -> str | None:
    """Turn tracing on. With catalog+schema+warehouse, traces land in UC tables.
    Returns the experiment id, or None if MLflow is unavailable."""
    global _ENABLED
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed; tracing stays off")
        return None
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))
    kwargs = {}
    if catalog and schema and warehouse_id:
        from mlflow.entities.trace_location import UnityCatalog

        os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouse_id
        kwargs["trace_location"] = UnityCatalog(catalog_name=catalog, schema_name=schema)
        log.info("tracing to Unity Catalog %s.%s via warehouse %s", catalog, schema, warehouse_id)
    else:
        log.warning("no catalog/schema/warehouse given: traces use experiment storage")
    exp = mlflow.set_experiment(experiment_name=experiment_name, **kwargs)
    _ENABLED = True
    return exp.experiment_id


def _jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, list):
        return [_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


class _Span:
    def __init__(self, inner=None, span_type: str = ""):
        self._inner, self._type = inner, span_type

    def output(self, value) -> None:
        if self._inner is None:
            return
        if self._type == "RETRIEVER" and isinstance(value, list):
            from mlflow.entities import Document

            value = [
                Document(
                    page_content=p.content,
                    id=p.chunk_id,
                    metadata={
                        "doc_id": p.doc_id,
                        "title": p.title,
                        "section": p.section,
                        "source": p.source,
                        "score": p.score,
                    },
                )
                for p in value
            ]
            self._inner.set_outputs(value)
            return
        self._inner.set_outputs(_jsonable(value))

    def attr(self, **attrs) -> None:
        if self._inner is not None:
            self._inner.set_attributes({k: _jsonable(v) for k, v in attrs.items()})


@contextmanager
def span(name: str, span_type: str = "CHAIN", inputs: dict | None = None):
    """`with span("rag.retrieve", "RETRIEVER", {"query": q}) as s: ...; s.output(hits)`"""
    if not _ENABLED:
        yield _Span()
        return
    import mlflow
    from mlflow.entities import SpanType

    with mlflow.start_span(name=name, span_type=getattr(SpanType, span_type, SpanType.UNKNOWN)) as s:
        if inputs is not None:
            s.set_inputs(_jsonable(inputs))
        yield _Span(s, span_type)
