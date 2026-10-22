"""SEC EDGAR client (official, free). Filings list, filing timestamps and XBRL company facts.

SEC fair-access policy: send a descriptive User-Agent with a contact and stay under 10 requests/second.
Set SEC_USER_AGENT="Your Name your@email.com" in .env. Without it, SEC features are disabled (never guessed).
Parsing code is covered by unit tests on fixtures with the documented JSON shape; it has NOT been run against
the live service from the development sandbox.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

import config  # noqa: F401  (loads .env so SEC_USER_AGENT is available)
from data.market_data import DataError

log = logging.getLogger(__name__)
CACHE_DIR = Path(".cache_sec")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

# Form 8-K item numbers -> (meaning, weight). weight: high / medium / low evidence value for a price move.
ITEMS: dict[str, tuple[str, str]] = {
    "1.01": ("Entry into a material definitive agreement", "high"),
    "1.02": ("Termination of a material agreement", "high"),
    "1.03": ("Bankruptcy or receivership", "high"),
    "2.01": ("Completion of acquisition or disposition", "high"),
    "2.02": ("Results of operations (earnings release)", "high"),
    "2.03": ("Creation of a direct financial obligation", "medium"),
    "2.04": ("Triggering events that accelerate an obligation", "high"),
    "2.05": ("Exit or disposal costs (restructuring)", "medium"),
    "2.06": ("Material impairments", "medium"),
    "3.01": ("Delisting notice / failure to satisfy a listing rule", "high"),
    "3.02": ("Unregistered sales of equity securities (dilution)", "high"),
    "3.03": ("Material modification to rights of holders", "medium"),
    "4.01": ("Change of certifying accountant", "medium"),
    "4.02": ("Non-reliance on previously issued financials", "high"),
    "5.01": ("Change in control", "high"),
    "5.02": ("Departure/appointment of directors or officers", "medium"),
    "5.07": ("Shareholder vote results", "low"),
    "7.01": ("Regulation FD disclosure", "low"),
    "8.01": ("Other events", "low"),
    "9.01": ("Financial statements and exhibits", "low"),
}
RISK_ITEMS = {"1.03": "bankruptcy", "3.01": "listing deficiency", "4.02": "non-reliance on financials",
              "3.02": "equity issuance / dilution", "2.06": "impairment", "2.04": "obligation acceleration",
              "1.02": "agreement termination"}


@dataclass(frozen=True)
class Filing:
    cik: int
    form: str
    filed: str                 # YYYY-MM-DD
    accepted_utc: datetime | None
    accession: str
    items: tuple[str, ...]
    document: str
    description: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, acc=self.accession.replace("-", ""), doc=self.document)

    def weight(self) -> str:
        if self.form.startswith("8-K"):
            ranks = {"high": 3, "medium": 2, "low": 1}
            best = max((ITEMS.get(i, ("", "low"))[1] for i in self.items), key=lambda w: ranks[w], default="low")
            return best
        if self.form in ("10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"):
            return "medium"
        if self.form in ("6-K",):
            return "low"
        return "low"


class SecError(DataError):
    pass


class SecClient:
    def __init__(self, user_agent: str | None = None, cache: Path = CACHE_DIR, min_interval: float = 0.15,
                 session: requests.Session | None = None):
        self.user_agent = user_agent if user_agent is not None else os.getenv("SEC_USER_AGENT", "")
        self.cache, self.min_interval = cache, min_interval
        self._s = session or requests.Session()
        self._last = 0.0

    @property
    def enabled(self) -> bool:
        return "@" in self.user_agent  # SEC asks for a contact address in the User-Agent

    def _get_json(self, url: str, cache_name: str | None = None, max_age_h: float = 12) -> dict:
        if not self.enabled:
            raise SecError("SEC features need SEC_USER_AGENT in .env (for example: Your Name you@email.com)")
        path = self.cache / cache_name if cache_name else None
        if path and path.exists() and (time.time() - path.stat().st_mtime) < max_age_h * 3600:
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = self._s.get(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=25)
            self._last = time.time()
            if r.status_code == 404:
                raise SecError("not found at SEC")
            r.raise_for_status()
            js = r.json()
        except requests.RequestException as exc:
            raise SecError(f"SEC request failed: {type(exc).__name__}") from exc
        except ValueError as exc:
            raise SecError("SEC returned invalid JSON") from exc
        if path:
            try:
                self.cache.mkdir(exist_ok=True)
                path.write_text(json.dumps(js))
            except OSError:
                pass
        return js

    # --------------------------------------------------------------------------------- lookups
    def cik_for(self, ticker: str) -> tuple[int, str]:
        js = self._get_json(TICKERS_URL, "company_tickers.json", max_age_h=24 * 7)
        return lookup_cik(js, ticker)

    def filings(self, cik: int, include_older: bool = True) -> list[Filing]:
        js = self._get_json(SUBMISSIONS_URL.format(cik=cik), f"sub_{cik}.json")
        out = parse_filings(cik, js.get("filings", {}).get("recent", {}))
        if include_older:
            for f in js.get("filings", {}).get("files", []):
                try:
                    older = self._get_json("https://data.sec.gov/submissions/" + f["name"], f"sub_{cik}_{f['name']}")
                    out += parse_filings(cik, older)
                except SecError as exc:
                    log.warning("older filings page skipped: %s", exc)
        return sorted(out, key=lambda x: x.accepted_utc or datetime.min.replace(tzinfo=timezone.utc))

    def company_facts(self, cik: int) -> dict:
        return self._get_json(FACTS_URL.format(cik=cik), f"facts_{cik}.json", max_age_h=24)


# ------------------------------------------------------------------------------------- pure parsers
def lookup_cik(company_tickers: dict, ticker: str) -> tuple[int, str]:
    t = ticker.upper().replace("-", ".")
    for rec in company_tickers.values():
        if str(rec.get("ticker", "")).upper().replace("-", ".") == t:
            return int(rec["cik_str"]), str(rec.get("title", ""))
    raise SecError(f"{ticker} not found in the SEC ticker list (not an SEC registrant or the ticker changed)")


def _parse_accepted(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_filings(cik: int, recent: dict) -> list[Filing]:
    """`recent` is the column-oriented dict from submissions JSON (parallel arrays)."""
    n = len(recent.get("accessionNumber", []))
    out = []
    for i in range(n):
        def col(name, default=""):
            arr = recent.get(name, [])
            return arr[i] if i < len(arr) else default
        items = tuple(x.strip() for x in str(col("items")).split(",") if x.strip())
        out.append(Filing(cik, str(col("form")), str(col("filingDate")), _parse_accepted(col("acceptanceDateTime", None)),
                          str(col("accessionNumber")), items, str(col("primaryDocument")),
                          str(col("primaryDocDescription"))))
    return out
