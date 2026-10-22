"""Cheap daily pre-screen for large universes: price range + average dollar volume, before any 1-minute download."""
from __future__ import annotations

import logging
import time

import pandas as pd

from data.market_data import DataError, validate_ticker
from scanner.config import ScannerConfig
from scanner.providers import split_download

log = logging.getLogger(__name__)


def prescreen_from_daily(frames: dict, cfg: ScannerConfig) -> pd.DataFrame:
    rows = []
    for t, df in frames.items():
        if isinstance(df, Exception) or df is None or len(df) < 3:
            continue
        last = float(df["Close"].iloc[-1])
        dv = float((df["Close"] * df["Volume"]).tail(5).mean())
        chg = float((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100)
        if cfg.min_price <= last <= cfg.max_price and dv >= cfg.min_avg_dollar_volume:
            rows.append({"symbol": t, "last": last, "avg_dollar_volume_5d": dv, "change_1d_%": chg})
    out = pd.DataFrame(rows)
    return out.sort_values("avg_dollar_volume_5d", ascending=False).reset_index(drop=True) if not out.empty else out


def daily_prescreen(tickers: list[str], cfg: ScannerConfig, chunk: int = 200, pause: float = 1.0) -> pd.DataFrame:
    """Download ~10 days of daily bars in batches (yfinance) and keep tickers inside the price/liquidity range."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataError("yfinance is not installed. Run: pip install -r requirements.txt") from exc
    tickers = [validate_ticker(t) for t in tickers]
    frames: dict = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            df = yf.download(tickers=part, period="10d", interval="1d", group_by="ticker", auto_adjust=False,
                             threads=True, progress=False)
            frames.update(split_download(df, part))
        except Exception as exc:
            log.warning("daily prescreen chunk failed: %s", exc)
        time.sleep(pause)
    return prescreen_from_daily(frames, cfg)
