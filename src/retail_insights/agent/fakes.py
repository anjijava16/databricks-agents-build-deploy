"""Deterministic stand-ins for the warehouse and the LLM.

Used by unit tests and by the offline eval in CI, so the whole agent loop
(tool routing, argument validation, SQL guard, error recovery) is exercised
on every pull request without workspace credentials or token spend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from retail_insights.agent.executor import QueryResult
from retail_insights.agent.llm import LLMReply, ToolCall


@dataclass
class FakeExecutor:
    """Returns `responder(sql)` and records every statement it was asked to run."""

    responder: Callable[[str], QueryResult] = lambda sql: QueryResult(["v"], [[1]])
    queries: list[str] = field(default_factory=list)

    def run(self, sql: str) -> QueryResult:
        self.queries.append(sql)
        return self.responder(sql)


@dataclass
class ScriptedLLM:
    """Plays back a list of replies. A callable entry receives the message list,
    which lets a script build its final answer from the last tool result."""

    script: list[LLMReply | Callable[[list[dict]], LLMReply]]
    seen: list[list[dict]] = field(default_factory=list)

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        self.seen.append(list(messages))
        step = self.script[min(len(self.seen) - 1, len(self.script) - 1)]
        return step(messages) if callable(step) else step


@dataclass
class RuleLLM:
    """Routes on the system prompt, like a switchboard: the first rule whose
    marker appears in the system message answers. Lets one fake play planner,
    grader, synthesizer and the analytics agent in a single graph run.

    A responder gets (system, messages) and returns a str or an LLMReply."""

    rules: list[tuple[str, Callable[[str, list[dict]], str | LLMReply]]]
    calls: list[str] = field(default_factory=list)

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        for marker, responder in self.rules:
            if marker in system:
                self.calls.append(marker)
                out = responder(system, messages)
                return out if isinstance(out, LLMReply) else LLMReply(content=out)
        raise AssertionError(f"no rule for system prompt: {system[:80]!r}")


def call(name: str, **arguments) -> LLMReply:
    return LLMReply(content=None, tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=arguments)])


def say(text: str) -> LLMReply:
    return LLMReply(content=text)
