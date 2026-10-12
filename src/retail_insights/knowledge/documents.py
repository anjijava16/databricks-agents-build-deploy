"""Turning raw files into clean `Document`s.

Two sources, two parsers:

  * Internal policies: Markdown with a small front-matter block (doc_id, title,
    owner, effective_date, version). Written by people, already clean.
  * SEC 10-K filings: inline-XBRL HTML downloaded from EDGAR. Large, noisy, and
    full of hidden XBRL metadata, so they need real cleaning.

Only the standard library is used (html.parser instead of BeautifulSoup) so the
same code runs inside a Spark UDF on serverless without extra dependencies.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path

POLICY = "policy"
FILING = "filing"


@dataclass
class Document:
    doc_id: str
    source: str  # POLICY | FILING
    title: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {**asdict(self), "content_hash": self.content_hash}


# ----------------------------------------------------------------------------- policies
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Tiny `key: value` front-matter reader (no PyYAML needed at runtime)."""
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, text[m.end() :]


def parse_policy(markdown: str, path: str = "") -> Document:
    meta, body = parse_front_matter(markdown)
    doc_id = meta.pop("doc_id", "") or Path(path).stem.upper()
    title = meta.pop("title", "") or _first_heading(body) or doc_id
    return Document(
        doc_id=doc_id,
        source=POLICY,
        title=title,
        text=body.strip(),
        metadata={**meta, "path": path},
    )


def _first_heading(md: str) -> str:
    for line in md.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


# ------------------------------------------------------------------------------ filings
_BLOCK_TAGS = {
    "p",
    "div",
    "br",
    "tr",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "table",
    "section",
    "article",
    "ul",
    "ol",
    "hr",
    "title",
}
_SKIP_TAGS = {"script", "style", "head", "ix:header", "noscript"}
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.IGNORECASE)


class _TextExtractor(HTMLParser):
    """HTML -> text with line breaks at block elements; drops scripts, styles,
    the hidden inline-XBRL header and any element styled display:none."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        style = dict(attrs).get("style") or ""
        if self._skip_stack or tag in _SKIP_TAGS or _HIDDEN_STYLE.search(style):
            # void elements never get an end tag, so don't track them
            if tag not in {"br", "hr", "img", "meta", "link", "input"}:
                self._skip_stack.append(tag)
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self._skip_stack:
            if tag == self._skip_stack[-1]:
                self._skip_stack.pop()
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_stack:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    raw = "".join(parser.parts).replace("\xa0", " ")
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip(" |")
        if line and not re.fullmatch(r"[\d\s|.,$%()-]{1,6}", line):  # page numbers, stray cells
            lines.append(line)
    return "\n".join(lines)


def parse_filing(html: str, meta: dict) -> Document:
    """`meta` comes from the JSON sidecar written by the EDGAR fetcher."""
    ticker = meta.get("ticker", "").upper()
    company = meta.get("company", ticker)
    filing_date = meta.get("filing_date", "")
    accession = meta.get("accession_number", "")
    form = meta.get("form", "10-K")
    return Document(
        doc_id=f"{ticker}-{accession}" if accession else ticker,
        source=FILING,
        title=f"{company} {form} (filed {filing_date})",
        text=html_to_text(html),
        metadata={
            "ticker": ticker,
            "company": company,
            "form": form,
            "filing_date": filing_date,
            "accession_number": accession,
            "url": meta.get("url", ""),
        },
    )


def parse_file(path: str, content: str, sidecar_json: str | None = None) -> Document:
    """Dispatch on extension. Used by the Spark UDF in knowledge/jobs.py."""
    if path.endswith(".md"):
        return parse_policy(content, path)
    if path.endswith((".htm", ".html")):
        meta = json.loads(sidecar_json) if sidecar_json else {}
        return parse_filing(content, meta)
    raise ValueError(f"unsupported document type: {path}")
