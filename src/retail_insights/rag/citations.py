"""The citation gate.

PolicyPilot's verify step rejected any answer without at least one "[n]". That
stops the worst failure (an uncited answer from general knowledge) but lets
through citations to sources that don't exist ("[7]" with five passages) and
answers where one sentence is cited and five are not.

This verifier checks three things:
  * every [n] points at a real evidence item,
  * at least one valid citation exists,
  * enough factual sentences carry a citation (coverage).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION = re.compile(r"\[(\d{1,3})\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[*-])")
_NON_CLAIM = re.compile(
    r"^(i (do not|don't|cannot|can't)|sorry|note:|sources?:|here is|here's)", re.IGNORECASE
)


@dataclass
class CitationReport:
    cited: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    claim_sentences: int = 0
    cited_sentences: int = 0

    @property
    def coverage(self) -> float:
        return self.cited_sentences / self.claim_sentences if self.claim_sentences else 0.0

    def grounded(self, min_coverage: float = 0.5) -> bool:
        return bool(self.cited) and not self.invalid and self.coverage >= min_coverage


def _sentences(text: str) -> list[str]:
    out = []
    for block in re.split(r"\n+", text):
        block = block.strip().lstrip("-*• ").strip()
        if not block or block.startswith("|---"):
            continue
        out.extend(s.strip() for s in _SENTENCE.split(block) if s.strip())
    return out


def check_citations(answer: str, n_evidence: int) -> CitationReport:
    report = CitationReport()
    for s in _sentences(answer):
        words = re.sub(CITATION, "", s).split()
        is_claim = len(words) >= 4 and not _NON_CLAIM.match(s) and not s.endswith(":")
        nums = [int(n) for n in CITATION.findall(s)]
        if is_claim:
            report.claim_sentences += 1
            if any(1 <= n <= n_evidence for n in nums):
                report.cited_sentences += 1
        for n in nums:
            (report.cited if 1 <= n <= n_evidence else report.invalid).append(n)
    report.cited = sorted(set(report.cited))
    report.invalid = sorted(set(report.invalid))
    return report
