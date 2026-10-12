"""Access to the internal policy library that ships inside the wheel.

The docs pipeline copies these files into the Unity Catalog volume (landing
zone), exactly as if another team had dropped them there. Locally, the same
files back the in-memory retriever, so RAG works on a laptop with no workspace.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

from retail_insights.knowledge.chunking import Chunk, chunk_document
from retail_insights.knowledge.documents import Document, parse_policy


def policy_files() -> dict[str, str]:
    root = resources.files("retail_insights.knowledge").joinpath("corpus", "policies")
    return {p.name: p.read_text(encoding="utf-8") for p in root.iterdir() if p.name.endswith(".md")}


def policy_documents() -> list[Document]:
    return [parse_policy(text, name) for name, text in sorted(policy_files().items())]


@cache
def policy_chunks() -> tuple[Chunk, ...]:
    return tuple(c for d in policy_documents() for c in chunk_document(d))
