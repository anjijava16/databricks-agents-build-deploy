import json

from mlflow.types.responses import ResponsesAgentRequest

from retail_insights.agent.executor import QueryResult
from retail_insights.agent.factory import Runtime, resolve_schema
from retail_insights.agent.fakes import FakeExecutor, RuleLLM
from retail_insights.serving.agent_model import RetailAgenticModel, to_chat_messages


def runtime(llm):
    return Runtime(
        "workspace",
        "retail_dev",
        "wh",
        "endpoint",
        "local",
        llm_override=llm,
        executor_override=FakeExecutor(lambda sql: QueryResult(["v"], [[1]])),
    )


def test_responses_agent_predict():
    llm = RuleLLM(
        [
            (
                "You plan how to answer",
                lambda s, m: json.dumps({"tasks": [{"route": "policies", "query": "electronics returns"}]}),
            ),
            ("You check whether retrieved passages", lambda s, m: '{"relevant": [1]}'),
            ("numbered evidence", lambda s, m: "Electronics can be returned within 15 days [1]."),
        ]
    )
    model = RetailAgenticModel(runtime=runtime(llm))
    req = ResponsesAgentRequest(
        input=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "How long do customers have to return electronics?"},
        ]
    )
    resp = model.predict(req)
    assert resp.output[0].content[0]["text"] == "Electronics can be returned within 15 days [1]."
    assert resp.custom_outputs["grounded"] is True
    assert resp.custom_outputs["routes"] == ["policies"]
    assert resp.custom_outputs["sources"][0]["type"] == "policy"


def test_message_conversion_handles_content_parts():
    req = ResponsesAgentRequest(
        input=[{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
    )
    assert to_chat_messages(req) == [{"role": "user", "content": "hello"}]


def test_schema_resolution(monkeypatch):
    monkeypatch.delenv("RETAIL_SCHEMA", raising=False)
    monkeypatch.setenv("DATABRICKS_APP_NAME", "retail-ai-prod")
    assert resolve_schema() == "retail_prod"
    monkeypatch.setenv("DATABRICKS_APP_NAME", "dev-someone-retail-ai-staging")
    assert resolve_schema() == "retail_staging"
    monkeypatch.setenv("RETAIL_SCHEMA", "custom")
    assert resolve_schema() == "custom"


def test_runtime_without_warehouse_has_no_analytics():
    rt = Runtime("workspace", "retail_dev", None, "endpoint", "local", llm_override=RuleLLM([]))
    assert rt.executor is None and rt.analytics is None
    assert rt.agentic.analytics is None
