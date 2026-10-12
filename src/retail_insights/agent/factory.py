"""Builds every assistant from environment variables, in one place.

Inside Databricks Apps the variables come from app.yaml (`valueFrom` binds the
SQL warehouse and serving endpoint declared in resources/apps.yml). In jobs
they come from task parameters; locally you export them yourself and auth
comes from your Databricks CLI profile.

    DATABRICKS_WAREHOUSE_ID   SQL warehouse for analytics tools and UC functions
    SERVING_ENDPOINT          chat model (OpenAI-compatible Model Serving endpoint)
    RETAIL_CATALOG / RETAIL_SCHEMA
    RETAIL_RETRIEVER          "vector" (Vector Search, default) or "local" (BM25 over the
                              bundled policy library, no workspace needed)
    RETAIL_VECTOR_INDEX       override the index name
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property

from retail_insights.agent.core import RetailAgent
from retail_insights.agent.executor import DatabricksSqlExecutor, SqlExecutor
from retail_insights.agent.llm import ChatClient, DatabricksChatClient
from retail_insights.agent.tools import Toolbox
from retail_insights.config import CHAT_ENDPOINT, Layout

TARGETS = ("dev", "staging", "prod")


def resolve_schema() -> str:
    """RETAIL_SCHEMA wins. Otherwise derive it from the app name, which the bundle
    sets to retail-ai-<target>, so each target reads its own schema."""
    if os.environ.get("RETAIL_SCHEMA"):
        return os.environ["RETAIL_SCHEMA"]
    app = os.environ.get("DATABRICKS_APP_NAME", "")
    for target in TARGETS:
        if app.endswith(f"-{target}"):
            return f"retail_{target}"
    return "retail_dev"


@dataclass
class Runtime:
    """Lazily wires LLM, warehouse, retriever and the three assistants."""

    catalog: str
    schema: str
    warehouse_id: str | None
    endpoint: str
    retriever_kind: str = "vector"
    llm_override: ChatClient | None = None
    executor_override: SqlExecutor | None = None

    @classmethod
    def from_env(cls) -> Runtime:
        return cls(
            catalog=os.environ.get("RETAIL_CATALOG", "workspace"),
            schema=resolve_schema(),
            warehouse_id=os.environ.get("DATABRICKS_WAREHOUSE_ID"),
            endpoint=os.environ.get("SERVING_ENDPOINT", CHAT_ENDPOINT),
            retriever_kind=os.environ.get("RETAIL_RETRIEVER", "vector"),
        )

    @property
    def layout(self) -> Layout:
        return Layout(self.catalog, self.schema)

    @cached_property
    def llm(self) -> ChatClient:
        return self.llm_override or DatabricksChatClient(self.endpoint)

    @cached_property
    def executor(self) -> SqlExecutor | None:
        if self.executor_override:
            return self.executor_override
        if not self.warehouse_id:
            return None
        return DatabricksSqlExecutor(self.warehouse_id, self.catalog, self.schema)

    @cached_property
    def analytics(self) -> RetailAgent | None:
        if self.executor is None:
            return None
        return RetailAgent(self.llm, Toolbox(self.executor, self.catalog, self.schema))

    @cached_property
    def retriever(self):
        if self.retriever_kind == "local":
            from retail_insights.knowledge.corpus import policy_chunks
            from retail_insights.rag.retrieval import BM25Retriever

            return BM25Retriever(policy_chunks())
        from retail_insights.rag.retrieval import VectorSearchRetriever

        return VectorSearchRetriever(os.environ.get("RETAIL_VECTOR_INDEX", self.layout.vector_index))

    @cached_property
    def rag(self):
        from retail_insights.rag.chain import RagChain

        return RagChain(self.llm, self.retriever)

    @cached_property
    def agentic(self):
        from retail_insights.agentic.graph import AgenticRAG

        return AgenticRAG(
            self.llm,
            self.retriever,
            analytics=self.analytics,
            executor=self.executor,
            filing_lookup_fn=self.layout.filing_lookup_fn,
        )


def build_agent_from_env() -> RetailAgent:
    """Kept for backwards compatibility with the first version of the app."""
    rt = Runtime.from_env()
    if rt.analytics is None:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID is not set")
    return rt.analytics
