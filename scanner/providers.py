"""Pluggable 1-minute data providers for the scanner.

A provider returns completed-or-forming 1-minute OHLCV per ticker and says honestly what it is
(`realtime=False` for every free source). Swap providers by implementing `MinuteProvider`; nothing else changes.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Protocol

import pandas as pd

from data.market_data import DataError, validate_ticker

log = logging.getLogger(__name__)
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class MinuteProvider(Protocol):
    name: str
    realtime: bool          # True only for a verified live feed. Free sources are always False.
    description: str

    def bars(self, tickers: list[str], days: int) -> dict[str, "pd.DataFrame | DataError"]: ...

    def quote(self, ticker: str) -> dict | None:
        """Real bid/ask if the source has it: {'bid','ask','bid_size','ask_size','time'} else None."""
        ...


def split_download(df: pd.DataFrame, tickers: list[str]) -> dict[str, "pd.DataFrame | DataError"]:
    """Split a yfinance `download(group_by='ticker')` frame into per-ticker OHLCV (or a DataError each)."""
    out: dict[str, pd.DataFrame | DataError] = {}
    if df is None or df.empty:
        return {t: DataError("provider returned no data (bad ticker, delisted, or rate limited)") for t in tickers}
    cols = df.columns
    for t in tickers:
        try:
            if isinstance(cols, pd.MultiIndex):
                lvl0 = set(cols.get_level_values(0))
                if t in lvl0:
                    part = df[t]
                elif len(tickers) == 1:
                    part = df.copy()
                    part.columns = part.columns.get_level_values(0)
                else:
                    raise KeyError(t)
            else:
                if len(tickers) != 1:
                    raise KeyError(t)
                part = df
            part = part[[c for c in OHLCV if c in part.columns]].dropna(how="all")
            if part.empty or any(c not in part.columns for c in OHLCV):
                raise DataError("no usable rows")
            out[t] = part
        except KeyError:
            out[t] = DataError("ticker missing from provider response")
        except DataError as exc:
            out[t] = exc
    return out


def summarize_daily(frames: dict) -> list[dict]:
    out = []
    for t, f in frames.items():
        if isinstance(f, Exception) or len(f) < 5:
            continue
        tail = f.tail(20)
        out.append({"ticker": t, "close": float(f["Close"].iloc[-1]),
                    "avg_dollar_volume_20d": float((tail["Close"] * tail["Volume"]).mean())})
    return out


class YFinanceMinuteProvider:
    name = "yfinance"
    realtime = False
    description = "Yahoo Finance via yfinance: unofficial, delayed, history capped (~7 days of 1m), may rate-limit."

    def __init__(self, chunk: int = 40, pause: float = 1.0, retries: int = 2):
        self.chunk, self.pause, self.retries = chunk, pause, retries

    def bars(self, tickers: list[str], days: int = 5) -> dict[str, "pd.DataFrame | DataError"]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataError("yfinance is not installed. Run: pip install -r requirements.txt") from exc
        days = int(min(max(days, 2), 7))  # Yahoo serves at most ~7-8 days of 1-minute data
        tickers = [validate_ticker(t) for t in tickers]
        result: dict[str, pd.DataFrame | DataError] = {}
        for i in range(0, len(tickers), self.chunk):
            part = tickers[i:i + self.chunk]
            last: Exception | None = None
            for attempt in range(self.retries + 1):
                try:
                    df = yf.download(tickers=part, period=f"{days}d", interval="1m", group_by="ticker",
                                     auto_adjust=False, prepost=False, threads=True, progress=False)
                    result.update(split_download(df, part))
                    last = None
                    break
                except Exception as exc:  # network / rate limit; reported per ticker below
                    last = exc
                    log.warning("yfinance chunk failed (attempt %d): %s", attempt + 1, exc)
                    time.sleep(self.pause * (attempt + 1))
            if last is not None:
                for t in part:
                    result[t] = DataError(f"yfinance failed after retries: {last}")
            time.sleep(self.pause)
        return result

    def quote(self, ticker: str) -> dict | None:
        return None  # Yahoo's bid/ask is neither reliable nor timely enough to be used as a spread

    def daily_summary(self, tickers: list[str]) -> pd.DataFrame:
        """Stage-1 screen: last close and 20-day average dollar volume per ticker (daily bars, batch download)."""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataError("yfinance is not installed. Run: pip install -r requirements.txt") from exc
        rows = []
        for i in range(0, len(tickers), 100):
            part = tickers[i:i + 100]
            try:
                df = yf.download(tickers=part, period="2mo", interval="1d", group_by="ticker", auto_adjust=False,
                                 threads=True, progress=False)
            except Exception as exc:
                log.warning("daily summary chunk failed: %s", exc)
                continue
            rows += summarize_daily(split_download(df, part))
            time.sleep(self.pause)
        return pd.DataFrame(rows, columns=["ticker", "close", "avg_dollar_volume_20d"])


class FrameProvider:
    """Serves bars from memory or files (backtests, tests, or data bought from a paid vendor)."""

    realtime = False

    def __init__(self, frames: dict[str, pd.DataFrame], name: str = "local-frames"):
        self.frames = frames
        self.name = name
        self.description = "Pre-loaded 1-minute bars (not a live feed)."

    def bars(self, tickers: list[str], days: int = 5) -> dict[str, "pd.DataFrame | DataError"]:
        return {t: self.frames[t] if t in self.frames else DataError("no local data for ticker") for t in tickers}

    def quote(self, ticker: str) -> dict | None:
        return None

    def daily_summary(self, tickers: list[str]) -> pd.DataFrame:
        rows = []
        for t in tickers:
            f = self.frames.get(t)
            if f is None or f.empty:
                continue
            day = f.groupby(pd.DatetimeIndex(f.index).date).agg(close=("Close", "last"), v=("Volume", "sum"))
            rows.append({"ticker": t, "close": float(day["close"].iloc[-1]),
                         "avg_dollar_volume_20d": float((f["Close"] * f["Volume"]).groupby(pd.DatetimeIndex(f.index).date).sum().tail(20).mean())})
        return pd.DataFrame(rows, columns=["ticker", "close", "avg_dollar_volume_20d"])


def load_bars_csv(path: str | Path, tz: str = "America/New_York") -> pd.DataFrame:
    """Load 1-minute bars from CSV. Needs a timestamp column plus Open/High/Low/Close/Volume.

    Timestamps without a timezone are interpreted as `tz` (default New York exchange time).
    """
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if c.lower() in ("timestamp", "datetime", "date", "time")), None)
    if tcol is None:
        raise DataError("CSV needs a timestamp/datetime column")
    ren = {c: c.capitalize() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
    df = df.rename(columns=ren)
    if any(c not in df.columns for c in OHLCV):
        raise DataError(f"CSV must contain {OHLCV}")
    idx = pd.to_datetime(df[tcol], utc=False)
    idx = idx.dt.tz_localize(tz) if idx.dt.tz is None else idx.dt.tz_convert(tz)
    out = df[OHLCV].copy()
    out.index = pd.DatetimeIndex(idx)
    return out.sort_index()
