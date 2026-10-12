import pytest

from retail_insights.knowledge.edgar import EdgarClient


class FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload, self.text = payload, text

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.headers, self.urls = {}, []

    def get(self, url, timeout):
        self.urls.append(url)
        if url.endswith("company_tickers.json"):
            return FakeResponse({"0": {"ticker": "BBY", "cik_str": 764478}})
        if "submissions" in url:
            return FakeResponse(
                {
                    "name": "BEST BUY CO INC",
                    "filings": {
                        "recent": {
                            "form": ["8-K", "10-K", "10-K"],
                            "filingDate": ["2026-05-01", "2026-03-20", "2025-03-21"],
                            "accessionNumber": ["a-1", "0000764478-26-000010", "old"],
                            "primaryDocument": ["x.htm", "bby-20260131.htm", "y.htm"],
                        }
                    },
                }
            )
        return FakeResponse(text="<html>10-K</html>")


def test_latest_10k_and_url():
    s = FakeSession()
    client = EdgarClient("Cedarline Data ops@example.com", pause_s=0, session=s)
    ref = client.latest_filing("bby")
    assert (ref.ticker, ref.cik, ref.filing_date) == ("BBY", 764478, "2026-03-20")
    assert ref.url.endswith("/764478/000076447826000010/bby-20260131.htm")
    assert client.download(ref) == "<html>10-K</html>"
    assert s.headers["User-Agent"].endswith("ops@example.com")
    assert ref.sidecar()["company"] == "BEST BUY CO INC"


def test_requires_contact_in_user_agent():
    with pytest.raises(ValueError, match="contact email"):
        EdgarClient("no contact", session=FakeSession())


def test_unknown_ticker():
    with pytest.raises(ValueError, match="no CIK"):
        EdgarClient("a b@c.d", pause_s=0, session=FakeSession()).latest_filing("ZZZZ")
