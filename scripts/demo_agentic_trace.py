"""Stream the agentic RAG graph node by node, so you can watch
plan -> gather -> grade -> (rewrite) -> synthesize -> verify -> finalize happen.

Ported from PolicyPilot's demo_agent_trace.py.

    export DATABRICKS_WAREHOUSE_ID=... RETAIL_SCHEMA=retail_dev
    uv run python scripts/demo_agentic_trace.py "Did we break our Electronics discount policy?"

Add RETAIL_RETRIEVER=local to retrieve from the bundled policy library instead of
Vector Search (the model and warehouse still come from your workspace).
"""

from __future__ import annotations

import json
import sys

from retail_insights.agent.factory import Runtime

DEFAULT = "How many customers are At Risk, and what does our playbook say we should do for them?"
MAX = 300


def short(v) -> str:
    text = v if isinstance(v, str) else json.dumps(v, default=str)
    return text if len(text) <= MAX else text[:MAX] + "…"


def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT
    agent = Runtime.from_env().agentic
    print(f"\nQuestion: {question}\n" + "=" * 72)
    for node, update in agent.stream(question):
        print(f"\n>>> {node.upper()}")
        for step in update.get("steps", []):
            for k, v in step.items():
                if k != "node":
                    print(f"  {k}: {short(v)}")
        if node == "finalize":
            print("\n" + "=" * 72 + f"\n{update['final']}\n")


if __name__ == "__main__":
    main()
