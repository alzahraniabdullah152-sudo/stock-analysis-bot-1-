"""Long daily price history with explicit adjustment labelling and actual coverage dates."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from data.market_data import DataError, validate_ticker

log = logging.getLogger(__name__)
COLS = ["Open", "High", "Low", "Close", "AdjClose", "Volume", "Dividends", "Splits"]


@dataclass
class PriceHistory:
    ticker: str
    df: pd.DataFrame                       # daily, tz-naive dates; Close is split-adjusted, AdjClose also dividend-adjusted
    name: str | None = None
    exchange: str | None = None
    currency: str | None = None
    first_trade_date: pd.Timestamp | None = None   # provider-reported first trade date, when available
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "yfinance (Yahoo Finance, unofficial)"
    notes: list[str] = field(default_factory=list)

    @property
    def start(self) -> pd.Timestamp:
        return self.df.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.df.index[-1]

    @property
    def splits(self) -> pd.Series:
        s = self.df["Splits"]
        return s[s > 0]

    @property
    def dividends(self) -> pd.Series:
        s = self.df["Dividends"]
        return s[s > 0]


def normalize_history(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean a yfinance daily frame (auto_adjust=False, actions=True). Returns (frame, notes)."""
    if raw is None or raw.empty:
        raise DataError("Provider returned no price history (bad ticker, delisted, or rate limit).")
    notes: list[str] = []
    d = raw.rename(columns={"Adj Close": "AdjClose", "Stock Splits": "Splits"}).copy()
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c not in d.columns:
            raise DataError(f"Missing column {c} in provider data")
    if "AdjClose" not in d.columns:
        d["AdjClose"] = d["Close"]
        notes.append("Dividend-adjusted close not provided; total-return figures use the split-adjusted close.")
    for c in ("Dividends", "Splits"):
        if c not in d.columns:
            d[c] = 0.0
    idx = pd.DatetimeIndex(d.index)
    d.index = (idx.tz_convert("America/New_York") if idx.tz is not None else idx).tz_localize(None).normalize()
    d = d[COLS].apply(pd.to_numeric, errors="coerce")
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    d = d[(d["Close"] > 0) & (d["High"] >= d["Low"])]
    d[["Volume", "Dividends", "Splits"]] = d[["Volume", "Dividends", "Splits"]].fillna(0)
    d["AdjClose"] = d["AdjClose"].fillna(d["Close"])
    if len(d) < 30:
        raise DataError(f"Only {len(d)} daily bars available; need at least 30.")
    return d, notes


def load_price_history(ticker: str) -> PriceHistory:
    ticker = validate_ticker(ticker)
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataError("yfinance is not installed. Run: pip install -r requirements.txt") from exc
    try:
        t = yf.Ticker(ticker)
        raw = t.history(period="max", interval="1d", auto_adjust=False, actions=True)
    except Exception as exc:
        raise DataError(f"yfinance failed: {type(exc).__name__}: {exc}") from exc
    df, notes = normalize_history(raw)
    meta: dict = {}
    try:
        meta = t.history_metadata or {}
    except Exception:  # metadata is optional
        log.info("history metadata unavailable for %s", ticker)
    first = meta.get("firstTradeDate")
    first_ts = pd.to_datetime(first, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize() \
        if isinstance(first, (int, float)) else None
    if first_ts is not None and first_ts < df.index[0] - pd.Timedelta(days=7):
        notes.append(f"Provider says trading began {first_ts:%Y-%m-%d} but data starts {df.index[0]:%Y-%m-%d}: "
                     "earlier history is not covered.")
    notes.append("Close is adjusted for splits; AdjClose also for dividends. Raw as-traded prices are not provided.")
    return PriceHistory(ticker, df, meta.get("longName") or meta.get("shortName"), meta.get("exchangeName"),
                        meta.get("currency"), first_ts, notes=notes)
