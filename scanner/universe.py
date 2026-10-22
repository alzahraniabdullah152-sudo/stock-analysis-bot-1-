"""US-listed universe from the official Nasdaq Trader symbol directory files.

Files (public, pipe-delimited): nasdaqlisted.txt and otherlisted.txt on https://www.nasdaqtrader.com/dynamic/SymDir/
They list currently listed securities only (not delisted names), so historical studies built on them carry
survivorship bias; this is stated wherever such results are shown. A local CSV/TXT can be used instead.
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import pandas as pd
import requests

from data.market_data import DataError

log = logging.getLogger(__name__)
URLS = {
    "nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}
CACHE = Path("universe_cache.csv")
_OK = re.compile(r"^[A-Z]{1,5}$")   # plain common-stock style tickers (no warrants/units/preferred classes)
EXCHANGE_MAP = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX"}


def parse_nasdaq_listed(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
    df = df[~df.iloc[:, 0].astype(str).str.startswith("File Creation Time")]
    out = pd.DataFrame({"symbol": df["Symbol"], "name": df["Security Name"], "exchange": "NASDAQ",
                        "etf": df.get("ETF", "N"), "test": df.get("Test Issue", "N")})
    return out


def parse_other_listed(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
    df = df[~df.iloc[:, 0].astype(str).str.startswith("File Creation Time")]
    return pd.DataFrame({"symbol": df["ACT Symbol"], "name": df["Security Name"],
                         "exchange": df["Exchange"].map(EXCHANGE_MAP).fillna(df["Exchange"]),
                         "etf": df.get("ETF", "N"), "test": df.get("Test Issue", "N")})


def clean_universe(df: pd.DataFrame, include_etf: bool = False, exchanges: tuple[str, ...] = ("NASDAQ", "NYSE")) -> pd.DataFrame:
    d = df.dropna(subset=["symbol"]).copy()
    d["symbol"] = d["symbol"].str.strip().str.upper()
    d = d[d["symbol"].str.match(_OK) & (d["test"].fillna("N") != "Y")]
    if not include_etf:
        d = d[d["etf"].fillna("N") != "Y"]
    d = d[d["exchange"].isin(exchanges)]
    return d.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)


def load_universe(refresh: bool = False, include_etf: bool = False, timeout: int = 20) -> pd.DataFrame:
    """Cached universe; downloads only when asked (or when no cache exists). Raises DataError with guidance."""
    if CACHE.exists() and not refresh:
        return clean_universe(pd.read_csv(CACHE, dtype=str), include_etf)
    try:
        frames = []
        for key, parser in (("nasdaq", parse_nasdaq_listed), ("other", parse_other_listed)):
            r = requests.get(URLS[key], timeout=timeout)
            r.raise_for_status()
            frames.append(parser(r.text))
        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(CACHE, index=False)
        return clean_universe(raw, include_etf)
    except (requests.RequestException, KeyError, ValueError) as exc:
        raise DataError(f"Could not download the symbol directory ({type(exc).__name__}). Provide your own "
                        "ticker list (one per line) instead.") from exc


def parse_ticker_text(text: str) -> list[str]:
    """Free-form text (commas / spaces / newlines) -> unique valid tickers."""
    toks = re.split(r"[\s,;]+", text.upper())
    return list(dict.fromkeys(t for t in toks if _OK.match(t)))
