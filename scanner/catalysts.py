"""Known catalysts for a candidate: recent headlines (yfinance) and recent SEC filings (official).

Headlines are tagged with a crude keyword classifier; the tag is a hint, never a conclusion, and every item
carries its publisher, link and publication time. If nothing is found the catalyst is reported as unknown.
Headlines exist only for the present: they are never used to explain or back-test past moves.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

CATEGORIES: list[tuple[str, bool, str]] = [   # (category, negative?, regex)
    ("earnings", False, r"\b(earnings|results|guidance|revenue|eps|quarter(ly)?|outlook|forecast)\b"),
    ("contract", False, r"\b(contract|award(ed)?|agreement|partnership|deal|order|collaborat\w+)\b"),
    ("regulatory", False, r"\b(fda|approval|approved|clearance|cleared|phase\s?[123i]+|trial|patent)\b"),
    ("acquisition", False, r"\b(acquire[sd]?|acquisition|merger|buyout|takeover|to be bought)\b"),
    ("legal", True, r"\b(lawsuit|sues?|sued|litigation|investigation|subpoena|probe|class action|settlement)\b"),
    ("offering_dilution", True, r"\b(offering|dilution|convertible|warrants?|shelf|at-the-market|atm|raises?\s+(?:\$[\d.,]+\s?\w*|capital|funds)|private placement)\b"),
    ("halt", True, r"\b(halt(ed)?|suspend(ed|s)?|delist\w*)\b"),
    ("negative", True, r"\b(downgrade[sd]?|plunge[sd]?|misses?|recall|bankruptcy|default|fraud|warning)\b"),
]
_COMPILED = [(c, neg, re.compile(rx, re.I)) for c, neg, rx in CATEGORIES]


def classify_headline(title: str) -> list[dict]:
    return [{"category": c, "negative": neg} for c, neg, rx in _COMPILED if rx.search(title or "")]


def _parse_news_item(item: dict) -> dict | None:
    """Handles both the older flat yfinance shape and the newer {'content': {...}} shape."""
    c = item.get("content") if isinstance(item.get("content"), dict) else item
    title = c.get("title") or ""
    if not title:
        return None
    pub = c.get("pubDate") or c.get("providerPublishTime")
    ts = None
    if isinstance(pub, (int, float)):
        ts = datetime.fromtimestamp(pub, tz=timezone.utc)
    elif isinstance(pub, str):
        try:
            ts = datetime.fromisoformat(pub.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            ts = None
    if ts is None:
        return None   # without a publication time the item cannot be placed relative to the move: drop it
    prov = c.get("provider")
    publisher = (prov.get("displayName") if isinstance(prov, dict) else None) or c.get("publisher") or "unknown"
    url = None
    for k in ("canonicalUrl", "clickThroughUrl"):
        if isinstance(c.get(k), dict) and c[k].get("url"):
            url = c[k]["url"]
            break
    url = url or c.get("link")
    return {"title": title, "publisher": publisher, "url": url, "published": ts}


def fetch_catalysts(ticker: str, sec=None, now: datetime | None = None, hours: int = 48) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    out: list[dict] = []
    try:
        import yfinance as yf
        for it in (yf.Ticker(ticker).news or [])[:20]:
            p = _parse_news_item(it)
            if p and now - timedelta(hours=hours) <= p["published"] <= now + timedelta(minutes=5):
                tags = classify_headline(p["title"])
                out.append({**p, "source": "yfinance news", "category": tags[0]["category"] if tags else "news",
                            "negative": any(t["negative"] for t in tags), "tags": [t["category"] for t in tags]})
    except Exception as exc:   # optional enrichment must never break a scan
        log.info("news unavailable for %s: %s", ticker, type(exc).__name__)
    if sec is not None and getattr(sec, "enabled", False):
        try:
            cik, _ = sec.cik_for(ticker)
            for f in sec.filings(cik, include_older=False)[-15:]:
                if f.accepted_utc and now - timedelta(hours=hours) <= f.accepted_utc <= now + timedelta(minutes=5):
                    meaning = [__import__("history.sec", fromlist=["ITEMS"]).ITEMS.get(i, (i, ""))[0] for i in f.items]
                    out.append({"title": f"{f.form} {', '.join(f.items)} {'; '.join(meaning)}".strip(), "publisher": "SEC EDGAR",
                                "url": f.url, "published": f.accepted_utc, "source": "SEC", "category": "sec_filing",
                                "negative": any(i in ("1.03", "3.01", "4.02", "3.02", "2.04") for i in f.items), "tags": list(f.items)})
        except Exception as exc:
            log.info("SEC filings unavailable for %s: %s", ticker, type(exc).__name__)
    return sorted(out, key=lambda x: x["published"], reverse=True)
