"""Synthetic 1-minute bars for TESTS ONLY (never shown in the app as market data)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from data.us_calendar import NY, previous_trading_day, session_hours


def trading_days(end: date, n: int) -> list[date]:
    out, d = [], end
    while len(out) < n:
        hrs = session_hours(d)
        if hrs is not None:
            out.append(d)
        d = previous_trading_day(d) if hrs is not None else d.fromordinal(d.toordinal() - 1)
    return sorted(out)


def make_minute_bars(end: date = date(2026, 10, 20), sessions: int = 6, seed: int = 3, price: float = 10.0,
                     base_volume: float = 8_000, last_session_minutes: int | None = None,
                     surge_at: int | None = None, surge_pct: float = 0.0, surge_vol_mult: float = 1.0,
                     drift: float = 0.0, surges: list | None = None) -> pd.DataFrame:
    """Random-walk bars with a U-shaped intraday volume curve.

    surge_at: minute-of-session (last session) where a price/volume surge starts (ramp over 8 minutes).
    surges: list of (session_index, minute, pct, volume_multiplier) for additional surges in any session.
    last_session_minutes: truncate the final session (simulate 'now' mid-session).
    """
    rng = np.random.default_rng(seed)
    frames = []
    px = price
    days = trading_days(end, sessions)
    for si, day in enumerate(days):
        o, c = session_hours(day)
        n = int((c - o).total_seconds() // 60)
        if si == len(days) - 1 and last_session_minutes is not None:
            n = min(n, last_session_minutes)
        idx = pd.date_range(o, periods=n, freq="min", tz=NY)
        m = np.arange(n)
        shape = 1.0 + 1.6 * np.exp(-m / 25) + 1.2 * np.exp(-(n - m) / 25)
        vol = rng.poisson(base_volume * shape / shape.mean()).astype(float)
        ret = rng.normal(drift, 0.0006, n)
        if si == len(days) - 1 and surge_at is not None and surge_at < n:
            ramp = np.zeros(n)
            span = np.arange(surge_at, min(n, surge_at + 8))
            ramp[span] = surge_pct / 100 / 8
            ret = ret + ramp
            vol[surge_at:] *= surge_vol_mult
        for (ssi, at, pct, vm) in (surges or []):
            if ssi == si and at < n:
                span = np.arange(at, min(n, at + 8))
                ret[span] += pct / 100 / 8
                vol[at:] *= vm
        close = px * np.exp(np.cumsum(ret))
        open_ = np.r_[px, close[:-1]]
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0003, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0003, n)))
        frames.append(pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx))
        px = close[-1] * (1 + rng.normal(0, 0.002))  # overnight gap
    return pd.concat(frames)


def make_daily_bars(n: int = 700, seed: int = 5, price: float = 50.0, end: date | None = None) -> pd.DataFrame:
    """Random-walk daily OHLCV on real trading days (tests only)."""
    rng = np.random.default_rng(seed)
    days = trading_days(end or date(2026, 10, 16), n)
    r = rng.normal(0.0004, 0.016, n)
    c = price * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    v = rng.integers(2_000_000, 6_000_000, n).astype(float)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=pd.DatetimeIndex(days))
