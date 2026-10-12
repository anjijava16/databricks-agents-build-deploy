"""The agentic RAG assistant packaged as an MLflow ResponsesAgent.

This file is logged with MLflow "models from code": MLflow stores the file
itself, and at load time runs it and picks up the object passed to
`mlflow.models.set_model`. That is how Mosaic AI Agent Framework expects
agents to be packaged, and what lets `databricks.agents.deploy` put it behind
a Model Serving endpoint with the Review App, inference tables and
auto-provisioned credentials for the resources declared at log time.

Configuration comes from the `model_config` given at log time (see
serving/register.py), so one artifact can point at dev or prod resources.
"""

from __future__ import annotations

import uuid

import mlflow
from mlflow.models import ModelConfig
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from retail_insights.agent.factory import Runtime

DEFAULT_CONFIG = {
    "catalog": "workspace",
    "schema": "retail_dev",
    "warehouse_id": "",
    "chat_endpoint": "databricks-meta-llama-3-3-70b-instruct",
    "retriever": "vector",
}


def _text(content) -> str:
    """Responses API content is a string or a list of typed parts."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        if isinstance(p, dict):
            parts.append(p.get("text") or "")
        else:
            parts.append(getattr(p, "text", "") or "")
    return "".join(parts)


def to_chat_messages(request: ResponsesAgentRequest) -> list[dict]:
    msgs = []
    for item in request.input:
        d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        role = d.get("role")
        if role in ("user", "assistant"):
            msgs.append({"role": role, "content": _text(d.get("content"))})
    return msgs


class RetailAgenticModel(ResponsesAgent):
    def __init__(self, runtime: Runtime | None = None, config: dict | None = None):
        if runtime is None:
            cfg = {**DEFAULT_CONFIG, **(config or {})}
            runtime = Runtime(
                catalog=cfg["catalog"],
                schema=cfg["schema"],
                warehouse_id=cfg["warehouse_id"] or None,
                endpoint=cfg["chat_endpoint"],
                retriever_kind=cfg["retriever"],
            )
        self.runtime = runtime

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        msgs = to_chat_messages(request)
        if not msgs or msgs[-1]["role"] != "user":
            raise ValueError("the last input item must be a user message")
        result = self.runtime.agentic.run(msgs[-1]["content"], history=msgs[:-1][-6:])
        sources = [
            {
                "n": i,
                "type": e.kind,
                "title": e.title,
                "chunk_id": e.passage.chunk_id if e.passage else None,
                "sql": e.sql,
            }
            for i, e in enumerate(result.evidence, 1)
        ]
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=result.answer, id=str(uuid.uuid4()))],
            custom_outputs={"grounded": result.grounded, "routes": result.routes, "sources": sources},
        )


def _load() -> RetailAgenticModel:
    # At serving time ModelConfig returns the model_config given to log_model;
    # in development it falls back to DEFAULT_CONFIG.
    cfg = ModelConfig(development_config=DEFAULT_CONFIG)
    return RetailAgenticModel(config={k: cfg.get(k) for k in DEFAULT_CONFIG})


# Construction is cheap: Runtime wires clients lazily, nothing connects until predict().
AGENT = _load()
mlflow.models.set_model(AGENT)
