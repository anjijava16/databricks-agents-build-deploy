"""Chat model client.

Databricks Model Serving speaks the OpenAI chat-completions protocol, so the
plain `openai` client works against Foundation Model APIs (pay-per-token), an
external-model endpoint (OpenAI, Anthropic, Groq...) or an AI Gateway endpoint.
Swapping models is a config change: the endpoint name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class ChatClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> LLMReply: ...


def _parse_args(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"_raw": raw}
    except json.JSONDecodeError:
        return {"_raw": raw}  # surfaced back to the model as a tool error


@dataclass
class DatabricksChatClient:
    endpoint: str
    temperature: float = 0.0
    max_tokens: int = 1024
    _client: object = field(default=None, repr=False)

    def _openai(self):
        if self._client is None:
            from databricks.sdk import WorkspaceClient

            self._client = WorkspaceClient().serving_endpoints.get_open_ai_client()
        return self._client

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        resp = self._openai().chat.completions.create(
            model=self.endpoint,
            messages=messages,
            tools=tools or None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=_parse_args(tc.function.arguments))
            for tc in (msg.tool_calls or [])
        ]
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        return LLMReply(content=msg.content, tool_calls=calls, usage=usage)
