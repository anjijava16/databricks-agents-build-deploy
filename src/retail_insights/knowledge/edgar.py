"""SEC EDGAR client, ported from PolicyPilot.

Public API, no key, but SEC's fair-access policy requires a descriptive
User-Agent with a real contact address on every request and asks clients to
stay under 10 requests per second. Runs inside a Databricks job; serverless
compute needs outbound internet access to sec.gov for this step (if the
workspace blocks egress, the docs pipeline still works with policies only).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"

# Public retail peers, so the agent can put internal numbers next to what the
# big listed retailers disclose (risk factors, return policies, segments...).
RETAIL_PEERS = ["WMT", "COST", "TGT", "HD", "LOW", "AMZN", "KR", "BBY", "DG", "TJX"]


@dataclass
class FilingRef:
    ticker: str
    cik: int
    company: str
    form: str
    filing_date: str
    accession_number: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVES_URL.format(
            cik=self.cik,
            accession_nodash=self.accession_number.replace("-", ""),
            document=self.primary_document,
        )

    def sidecar(self) -> dict:
        return {
            "ticker": self.ticker,
            "cik": self.cik,
            "company": self.company,
            "form": self.form,
            "filing_date": self.filing_date,
            "accession_number": self.accession_number,
            "url": self.url,
        }


class EdgarClient:
    def __init__(self, user_agent: str, pause_s: float = 0.15, session=None):
        if "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent with a contact email, e.g. 'Acme ops@acme.com'")
        if session is None:
            import requests

            session = requests.Session()
        self._session = session
        self._session.headers.update({"User-Agent": user_agent})
        self._pause = pause_s
        self._tickers: dict[str, int] | None = None

    def _get(self, url: str):
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        time.sleep(self._pause)
        return resp

    def cik(self, ticker: str) -> int:
        if self._tickers is None:
            data = self._get(TICKERS_URL).json()
            self._tickers = {e["ticker"].upper(): int(e["cik_str"]) for e in data.values()}
        try:
            return self._tickers[ticker.upper()]
        except KeyError as e:
            raise ValueError(f"no CIK for ticker {ticker!r}") from e

    def latest_filing(self, ticker: str, form: str = "10-K") -> FilingRef:
        cik = self.cik(ticker)
        data = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
        recent = data["filings"]["recent"]
        for i, f in enumerate(recent["form"]):
            if f == form:
                return FilingRef(
                    ticker=ticker.upper(),
                    cik=cik,
                    company=data.get("name", ticker),
                    form=form,
                    filing_date=recent["filingDate"][i],
                    accession_number=recent["accessionNumber"][i],
                    primary_document=recent["primaryDocument"][i],
                )
        raise ValueError(f"no {form} in recent filings for {ticker}")

    def download(self, ref: FilingRef) -> str:
        return self._get(ref.url).text
