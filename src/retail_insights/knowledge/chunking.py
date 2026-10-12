"""Structure-aware chunking.

PolicyPilot used fixed 1,200-word windows. That works, but a window often starts
mid-section and mixes two unrelated topics, which hurts both retrieval and
citations. Here we:

  1. split a document into sections using its own structure (Markdown headings
     for policies, "Item 1A. Risk Factors" style headings for 10-Ks),
  2. pack paragraphs into chunks of about `target_words`, never crossing a
     section boundary, with a small word overlap between neighbours,
  3. prefix every chunk with a contextual header (document title + section path)
     in a separate `embed_text` column. The header goes into the embedding so a
     chunk that just says "15 days" still retrieves for "electronics return
     window", while the clean `content` is what the model reads and cites.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from retail_insights.knowledge.documents import FILING, Document

TARGET_WORDS = 220
MAX_WORDS = 320
OVERLAP_WORDS = 40


@dataclass
class Section:
    path: str  # e.g. "Returns and Refunds Policy > Category exceptions > Electronics"
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    section: str
    chunk_index: int
    content: str
    embed_text: str
    word_count: int
    ticker: str = ""
    company: str = ""
    filing_date: str = ""
    url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------- sectioning
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def markdown_sections(text: str, title: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            names = [name for _, name in stack] or [title]
            sections.append(Section(" > ".join(names), body))
        buf.clear()

    for line in text.splitlines():
        m = _MD_HEADING.match(line)
        if m:
            flush()
            level, name = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, name))
        else:
            buf.append(line)
    flush()
    return sections


_ITEM_HEADING = re.compile(
    r"^\s*item\s+(1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16)\b[\s.:—–-]*(.{0,120})$",
    re.IGNORECASE,
)
_MIN_SECTION_WORDS = 120  # anything shorter between two item headings is a table of contents


def filing_sections(text: str, title: str) -> list[Section]:
    """Split 10-K text at 'Item N.' headings, skipping the table of contents."""
    lines = text.splitlines()
    marks = []
    for i, line in enumerate(lines):
        m = _ITEM_HEADING.match(line)
        if m and len(line) < 160:
            label = f"Item {m.group(1).upper()}"
            rest = m.group(2).strip(" .:-")
            marks.append((i, f"{label}. {rest}" if rest else label))
    if not marks:
        return [Section(title, text)] if text.strip() else []

    sections: list[Section] = []
    preamble = "\n".join(lines[: marks[0][0]]).strip()
    if len(preamble.split()) >= _MIN_SECTION_WORDS:
        sections.append(Section(f"{title} > Cover", preamble))
    for n, (start, name) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        if len(body.split()) < _MIN_SECTION_WORDS:
            continue  # TOC entry or an empty "Not applicable" item
        sections.append(Section(f"{title} > {name}", body))
    return sections


# ---------------------------------------------------------------------------- packing
def _paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[-*] )", text) if p.strip()]
    out: list[str] = []
    for p in paras:
        words = p.split()
        if len(words) <= MAX_WORDS:
            out.append(" ".join(words))
            continue
        # very long paragraph: break on sentences, then hard-wrap if needed
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(words))
        cur: list[str] = []
        for s in sentences:
            if cur and len(" ".join(cur).split()) + len(s.split()) > TARGET_WORDS:
                out.append(" ".join(cur))
                cur = []
            cur.append(s)
        if cur:
            out.append(" ".join(cur))
    final: list[str] = []
    for p in out:
        w = p.split()
        for i in range(0, len(w), MAX_WORDS):
            final.append(" ".join(w[i : i + MAX_WORDS]))
    return final


def pack(text: str, target: int = TARGET_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Greedy paragraph packing with a word overlap carried into the next chunk."""
    chunks: list[str] = []
    cur: list[str] = []
    for para in _paragraphs(text):
        cur_words = sum(len(p.split()) for p in cur)
        if cur and cur_words + len(para.split()) > target:
            chunks.append("\n\n".join(cur))
            tail = " ".join(chunks[-1].split()[-overlap:]) if overlap else ""
            cur = [tail] if tail else []
        cur.append(para)
    if cur:
        body = "\n\n".join(cur)
        # don't emit a chunk that is only the overlap tail of the previous one
        if not chunks or body.split() != chunks[-1].split()[-overlap:]:
            chunks.append(body)
    return chunks


MIN_SECTION_WORDS = 60


def merge_small_sections(sections: list[Section], min_words: int = MIN_SECTION_WORDS) -> list[Section]:
    """Fold tiny sections into their next sibling so we don't index 9-word chunks."""
    out: list[Section] = []
    for s in sections:
        prev = out[-1] if out else None
        if prev and len(prev.text.split()) < min_words:
            leaf = s.path.split(" > ")[-1]
            out[-1] = Section(f"{prev.path} + {leaf}", f"{prev.text}\n\n{leaf}:\n{s.text}")
        else:
            out.append(s)
    if len(out) > 1 and len(out[-1].text.split()) < min_words:  # tiny tail joins the previous one
        tail = out.pop()
        leaf = tail.path.split(" > ")[-1]
        out[-1] = Section(f"{out[-1].path} + {leaf}", f"{out[-1].text}\n\n{leaf}:\n{tail.text}")
    return out


# ------------------------------------------------------------------------------- main
def chunk_document(doc: Document) -> list[Chunk]:
    sections = merge_small_sections(
        filing_sections(doc.text, doc.title)
        if doc.source == FILING
        else markdown_sections(doc.text, doc.title)
    )
    m = doc.metadata
    chunks: list[Chunk] = []
    for section in sections:
        for body in pack(section.text):
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{idx:04d}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    section=section.path,
                    chunk_index=idx,
                    content=body,
                    embed_text=f"{doc.title} | {section.path}\n{body}",
                    word_count=len(body.split()),
                    ticker=m.get("ticker", ""),
                    company=m.get("company", ""),
                    filing_date=m.get("filing_date", ""),
                    url=m.get("url", ""),
                )
            )
    return chunks
