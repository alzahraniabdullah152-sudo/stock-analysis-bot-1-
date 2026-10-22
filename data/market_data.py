"""Market data layer: yfinance (default) and Alpha Vantage providers.

Never fabricates data. Failures raise ``DataError`` with a clear message.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import INTERVALS, INTRADAY, SETTINGS

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z])?$")


class DataError(Exception):
    """Raised when market data cannot be retrieved or is invalid."""


@dataclass
class MarketData:
    """OHLCV frame plus provenance metadata."""

    ticker: str
    interval: str
    df: pd.DataFrame
    provider: str
    fetched_at: datetime
    last_bar_time: pd.Timestamp
    data_type: str  # "historical" | "delayed" (free providers are never real-time)
    market_status: str  # "open" | "closed" | "pre/post"
    last_bar_complete: bool


def validate_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if not _TICKER_RE.match(t):
        raise DataError(f"Invalid ticker symbol: {ticker!r}")
    return t


def market_session(now: datetime | None = None) -> str:
    """US session status using the NYSE holiday / early-close calendar (data.us_calendar)."""
    from data.us_calendar import market_status

    return market_status(now)


class Provider(Protocol):
    name: str

    def history(self, ticker: str, interval: str) -> pd.DataFrame: ...


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataError("Provider returned no data (bad ticker, holiday, or rate limit).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise DataError(f"Missing columns from provider: {missing}")
    df = df[OHLCV].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        raise DataError("All rows invalid after cleaning.")
    df["Volume"] = df["Volume"].fillna(0)
    return df


class YFinanceProvider:
    name = "yfinance"

    def history(self, ticker: str, interval: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataError("yfinance is not installed. Run: pip install -r requirements.txt") from exc

        yf_interval, period, _ = INTERVALS[interval]
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                df = yf.Ticker(ticker).history(
                    period=period, interval=yf_interval, auto_adjust=False, actions=False
                )
                return _clean(df)
            except DataError:
                raise
            except Exception as exc:  # network / rate limit
                last_err = exc
                log.warning("yfinance attempt %d failed: %s", attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))
        raise DataError(f"yfinance failed after retries: {last_err}")


class AlphaVantageProvider:
    name = "alphavantage"
    _URL = "https://www.alphavantage.co/query"

    def history(self, ticker: str, interval: str) -> pd.DataFrame:
        if not SETTINGS.alphavantage_key:
            raise DataError("ALPHAVANTAGE_API_KEY is not set.")
        params: dict[str, str] = {"symbol": ticker, "apikey": SETTINGS.alphavantage_key, "outputsize": "full"}
        if interval == "1d":
            params["function"] = "TIME_SERIES_DAILY"
            key = "Time Series (Daily)"
        elif interval == "1wk":
            params["function"] = "TIME_SERIES_WEEKLY"
            key = "Weekly Time Series"
        elif interval in ("5m", "15m"):
            params["function"] = "TIME_SERIES_INTRADAY"
            params["interval"] = {"5m": "5min", "15m": "15min"}[interval]
            key = f"Time Series ({params['interval']})"
        else:
            raise DataError(f"Interval {interval} not supported by Alpha Vantage here.")
        try:
            r = requests.get(self._URL, params=params, timeout=20)
            r.raise_for_status()
            js = r.json()
        except requests.RequestException as exc:
            raise DataError(f"Alpha Vantage request failed: {exc}") from exc
        if key not in js:
            note = js.get("Note") or js.get("Information") or js.get("Error Message") or "unknown error"
            raise DataError(f"Alpha Vantage: {note}")
        df = pd.DataFrame.from_dict(js[key], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
        return _clean(df)


def get_provider(name: str | None = None) -> Provider:
    name = (name or SETTINGS.data_provider).lower()
    if name == "yfinance":
        return YFinanceProvider()
    if name in ("alphavantage", "alpha_vantage"):
        return AlphaVantageProvider()
    raise DataError(f"Unknown provider: {name}")


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    out = df.resample("4h", origin="start_day", offset="1h30min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return out.dropna(subset=["Open"])


def _bar_seconds(interval: str) -> int:
    return {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1wk": 604800}[interval]


def fetch(ticker: str, interval: str = "1d", provider: str | None = None,
          include_incomplete: bool | None = None) -> MarketData:
    """Fetch OHLCV and return it with provenance.

    By default the still-forming last candle is dropped (look-ahead/repaint safe).
    """
    ticker = validate_ticker(ticker)
    if interval not in INTERVALS:
        raise DataError(f"Unsupported interval {interval}. Options: {list(INTERVALS)}")
    prov = get_provider(provider)
    df = prov.history(ticker, interval)
    if INTERVALS[interval][2]:
        df = _resample_4h(df)
    now = datetime.now(timezone.utc)
    status = market_session(now)
    last_ts = df.index[-1]
    last_utc = last_ts.tz_convert("UTC") if last_ts.tzinfo else last_ts.tz_localize(NY).tz_convert("UTC")
    complete = (now - last_utc.to_pydatetime()).total_seconds() >= _bar_seconds(interval)
    if interval in ("1d", "1wk"):
        # daily bar of a closed session is complete; during session it is forming
        complete = status != "open" if interval == "1d" else (status != "open" and now.weekday() >= 4)
    if include_incomplete is None:
        include_incomplete = not SETTINGS.use_completed_candles
    if not complete and not include_incomplete and len(df) > 1:
        df = df.iloc[:-1]
        last_ts = df.index[-1]
        complete = True
    if len(df) < 30:
        raise DataError(f"Only {len(df)} bars available; need at least 30.")
    return MarketData(
        ticker=ticker, interval=interval, df=df, provider=prov.name, fetched_at=now,
        last_bar_time=last_ts, data_type="delayed" if interval in INTRADAY else "historical",
        market_status=status, last_bar_complete=complete,
    )
