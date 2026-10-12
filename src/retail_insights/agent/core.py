"""The agent loop: a plain, readable ReAct-style tool-calling loop.

    question -> LLM -> (tool calls -> results -> LLM)* -> answer

No framework on purpose. The whole control flow fits on one screen, every
step is recorded in a trace the UI shows, and there are hard limits on steps
so a confused model cannot loop forever and burn tokens.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date

from retail_insights.agent.llm import ChatClient
from retail_insights.agent.prompts import system_prompt
from retail_insights.agent.tools import Toolbox


@dataclass
class Step:
    tool: str
    arguments: dict
    result: dict
    seconds: float

    @property
    def sql(self) -> str | None:
        return self.result.get("sql") if isinstance(self.result, dict) else None

    @property
    def ok(self) -> bool:
        return "error" not in self.result


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped_reason: str = "answered"
    usage: dict = field(default_factory=dict)

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps]


class RetailAgent:
    def __init__(self, llm: ChatClient, toolbox: Toolbox, max_steps: int = 6, today: date | None = None):
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps
        self.today = today

    def run(self, question: str, history: list[dict] | None = None) -> AgentResult:
        messages: list[dict] = [{"role": "system", "content": system_prompt(self.today)}]
        messages += history or []
        messages.append({"role": "user", "content": question})
        steps: list[Step] = []
        usage: dict = {}

        for _ in range(self.max_steps):
            reply = self.llm.complete(messages, self.toolbox.specs())
            for k, v in (reply.usage or {}).items():
                if isinstance(v, int):
                    usage[k] = usage.get(k, 0) + v

            if not reply.tool_calls:
                return AgentResult(reply.content or "", steps, "answered", usage)

            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in reply.tool_calls
                    ],
                }
            )
            for call in reply.tool_calls:
                t0 = time.perf_counter()
                result = self.toolbox.call(call.name, call.arguments)
                steps.append(Step(call.name, call.arguments, result, time.perf_counter() - t0))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, default=str),
                    }
                )

        return AgentResult(
            "I could not finish this within the step limit. Try a narrower question, "
            "for example a specific date range or metric.",
            steps,
            "max_steps",
            usage,
        )
