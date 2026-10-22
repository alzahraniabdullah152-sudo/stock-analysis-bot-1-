"""Vectorised intraday feature engine (1-minute bars).

Look-ahead safety: every value at bar *i* uses only bars <= i of the same session, or *prior* sessions.
  * VWAP, returns, breakout level, streaks: cumulative / shifted within the session.
  * RVOL: cumulative volume so far today / mean cumulative volume at the same minute of *earlier* sessions.
  * Liquidity baseline: mean dollar volume of *earlier* sessions only.
Missing inputs stay NaN. They are never replaced by neutral-looking values.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data.us_calendar import NY, session_hours
from indicators.technical_indicators import atr as _atr
from indicators.technical_indicators import ema as _ema
from indicators.technical_indicators import rsi as _rsi
from scanner.config import ScannerConfig

log = logging.getLogger(__name__)
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
SESSION_MINUTES = 390


class FeatureError(Exception):
    """Raised when bars are unusable (never silently patched)."""


def prepare_bars(df: pd.DataFrame, now: datetime | None = None, drop_incomplete: bool = True) -> pd.DataFrame:
    """Clean 1-minute bars: NY timezone, sorted, de-duplicated, regular session only, completed bars only."""
    if df is None or df.empty:
        raise FeatureError("no bars")
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise FeatureError(f"missing columns {missing}")
    d = df[OHLCV].apply(pd.to_numeric, errors="coerce")
    idx = pd.DatetimeIndex(d.index)
    d.index = idx.tz_localize(NY) if idx.tz is None else idx.tz_convert(NY)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    d = d[(d[["Open", "High", "Low", "Close"]] > 0).all(axis=1) & (d["High"] >= d["Low"])]
    d["Volume"] = d["Volume"].fillna(0).clip(lower=0)
    if d.empty:
        raise FeatureError("no valid bars after cleaning")
    # regular session only (holiday / early-close aware)
    dates = d.index.normalize()
    keep = pd.Series(False, index=d.index)
    for day in pd.unique(dates):
        hrs = session_hours(day.date())
        if hrs is None:
            continue
        m = (d.index >= hrs[0]) & (d.index < hrs[1])
        keep[m] = True
    d = d[keep.to_numpy()]
    if d.empty:
        raise FeatureError("no regular-session bars")
    if drop_incomplete:
        now_ts = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert(NY)
        d = d[d.index + pd.Timedelta(minutes=1) <= now_ts]
        if d.empty:
            raise FeatureError("no completed bars")
    return d


def _grouped_shift(s: pd.Series, session: pd.Series, n: int) -> pd.Series:
    return s.groupby(session).shift(n)


def _rvol_and_liquidity(d: pd.DataFrame, cfg: ScannerConfig) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Time-of-day aware RVOL.

    rvol      = volume of the last `rvol_window` minutes / mean volume of the SAME minute window in earlier sessions
                (smoothed +-5 minutes). Reacts quickly to bursts.
    rvol_day  = cumulative volume so far / mean cumulative volume at the same minute of earlier sessions.
    Also returns the earlier sessions' mean dollar volume and how many earlier sessions were used.
    """
    session = d["session"]
    minute = np.clip(d["minute"].to_numpy(), 0, SESSION_MINUTES - 1)
    sessions = list(pd.unique(session))
    S, M, W = len(sessions), SESSION_MINUTES, max(1, int(cfg.rvol_window))
    grid = np.full((S, M), np.nan)
    eligible = np.zeros(S, dtype=bool)
    sess_dollar = np.zeros(S)
    for si, s in enumerate(sessions):
        part = d[session == s]
        hrs = session_hours(s)
        length = M if hrs is None else int((hrs[1] - hrs[0]).total_seconds() // 60)
        m = part["minute"].to_numpy()
        cv = part["Volume"].cumsum().to_numpy()
        # step function of cumulative volume over the session grid; flat when no trades
        full = pd.Series(cv, index=m).groupby(level=0).last().reindex(range(length)).ffill().fillna(0.0)
        grid[si, :length] = full.to_numpy()
        sess_dollar[si] = float((part["Close"] * part["Volume"]).sum())
        eligible[si] = bool(m.max() >= length - 15 and len(part) >= 0.3 * length)  # a (nearly) complete session
    gz = np.hstack([np.zeros((S, 1)), grid])                       # gz[:, k] = cumulative volume through minute k-1
    start = np.maximum(np.arange(1, M + 1) - W, 0)
    win = gz[:, 1:] - gz[:, start]                                 # volume in the W minutes ending at minute m
    base_win = np.full((S, M), np.nan)
    base_cum = np.full((S, M), np.nan)
    prior_cnt = np.zeros(S)
    prior_dollar = np.full(S, np.nan)
    for si in range(S):
        prev = [j for j in range(si) if eligible[j]]
        prior_cnt[si] = len(prev)
        if len(prev) < cfg.min_prior_sessions:
            continue
        for src, dst in ((win, base_win), (grid, base_cum)):
            block = src[prev]
            cnt = np.sum(~np.isnan(block), axis=0)
            with np.errstate(all="ignore"):
                mean = np.nanmean(block, axis=0)
            mean[cnt < cfg.min_prior_sessions] = np.nan
            dst[si] = mean
        base_win[si] = pd.Series(base_win[si]).rolling(11, center=True, min_periods=1).mean().to_numpy()
        prior_dollar[si] = float(np.mean(sess_dollar[prev]))
    row_s = pd.Series(range(S), index=sessions).reindex(session).to_numpy()
    cur_win = win[row_s, minute]
    cur_cum = grid[row_s, minute]
    with np.errstate(divide="ignore", invalid="ignore"):
        bw, bc = base_win[row_s, minute], base_cum[row_s, minute]
        rvol = np.where(bw > 0, cur_win / bw, np.nan)
        rvol_day = np.where(bc > 0, cur_cum / bc, np.nan)
    early = d["minute"].to_numpy() < cfg.skip_open_minutes
    rvol, rvol_day = np.where(early, np.nan, rvol), np.where(early, np.nan, rvol_day)
    ix = d.index
    return (pd.Series(rvol, index=ix), pd.Series(rvol_day, index=ix),
            pd.Series(prior_dollar[row_s], index=ix), pd.Series(prior_cnt[row_s], index=ix))


def _breakout_streaks(close: np.ndarray, level: np.ndarray, session: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """bars since the breakout began, share of bars holding the level since, and a 'failed breakout' flag."""
    n = len(close)
    since = np.full(n, np.nan)
    hold = np.full(n, np.nan)
    failed = np.zeros(n, dtype=bool)
    active_level = np.nan
    start = -1
    held = 0
    cur_session = None
    for i in range(n):
        if session[i] != cur_session:
            cur_session, active_level, start, held = session[i], np.nan, -1, 0
        if np.isnan(active_level):
            if not np.isnan(level[i]) and close[i] > level[i]:
                active_level, start, held = level[i], i, 1
                since[i], hold[i] = 0, 1.0
        else:
            n_bars = i - start + 1
            if close[i] >= active_level:
                held += 1
                since[i], hold[i] = i - start, held / n_bars
            else:
                failed[i] = True  # fell back below the breakout level
                since[i], hold[i] = i - start, held / n_bars
                active_level = np.nan
    return since, hold, failed


def build_features(bars: pd.DataFrame, bench: pd.DataFrame | None = None,
                   cfg: ScannerConfig | None = None, now: datetime | None = None,
                   drop_incomplete: bool = True) -> pd.DataFrame:
    """Per-bar feature table (index = bar start, NY tz). Row i depends only on bars <= i / earlier sessions."""
    cfg = cfg or ScannerConfig()
    d = prepare_bars(bars, now, drop_incomplete)
    d["session"] = [x.date() for x in d.index]
    hh_mm = d.index.hour * 60 + d.index.minute
    d["minute"] = hh_mm - (9 * 60 + 30)
    s = d["session"]
    g = d.groupby(s)

    typical = (d["High"] + d["Low"] + d["Close"]) / 3
    cum_pv = (typical * d["Volume"]).groupby(s).cumsum()
    cum_v = g["Volume"].cumsum()
    d["vwap"] = (cum_pv / cum_v.replace(0, np.nan))
    for k in (1, 5, 15):
        d[f"ret{k}"] = (d["Close"] / _grouped_shift(d["Close"], s, k) - 1) * 100
    d["accel"] = d["ret5"] - _grouped_shift(d["ret5"], s, 5)

    L = cfg.breakout_lookback
    d["level"] = g["High"].transform(lambda x: x.shift(1).rolling(L, min_periods=L).max())
    d["breakout"] = d["Close"] > d["level"]
    d["dist_pct"] = (d["Close"] / d["level"] - 1) * 100
    d["ext_vwap_pct"] = (d["Close"] / d["vwap"] - 1) * 100

    d["ema20"], d["ema50"] = _ema(d["Close"], 20), _ema(d["Close"], 50)
    d["rsi14"] = _rsi(d["Close"], 14)
    ret1 = d["Close"].groupby(s).pct_change() * 100
    d["vol30"] = ret1.groupby(s).transform(lambda x: x.rolling(30, min_periods=15).std())
    d["atr14_pct"] = _atr(d[["Open", "High", "Low", "Close"]], 14) / d["Close"] * 100

    d["rvol"], d["rvol_day"], d["avg_dollar_vol"], d["prior_sessions"] = _rvol_and_liquidity(d, cfg)
    d["dollar_vol_5m"] = (d["Close"] * d["Volume"]).groupby(s).transform(lambda x: x.rolling(5, min_periods=1).sum())

    since, hold, failed = _breakout_streaks(d["Close"].to_numpy(), d["level"].to_numpy(),
                                            np.array([x.toordinal() for x in d["session"]]))
    d["bars_since_breakout"], d["hold_ratio"], d["failed_breakout"] = since, hold, failed

    d["rs15"] = np.nan
    if bench is not None and not bench.empty:
        try:
            b = prepare_bars(bench, now, drop_incomplete)
            bs = pd.Series([x.date() for x in b.index], index=b.index)
            b15 = (b["Close"] / b["Close"].groupby(bs).shift(15) - 1) * 100
            d["bench_ret15"] = b15.reindex(d.index)
            d["rs15"] = d["ret15"] - d["bench_ret15"]
        except FeatureError as exc:
            log.warning("benchmark unusable: %s", exc)
    d["spread_pct"] = np.nan  # not provided by free bar data; only set from a real quote
    return d.replace([np.inf, -np.inf], np.nan)


def latest(features: pd.DataFrame, quote: dict | None = None) -> pd.Series:
    """Last completed bar's features; optionally attaches a real bid/ask spread."""
    if features.empty:
        raise FeatureError("no features")
    row = features.iloc[-1].copy()
    if quote and quote.get("bid") and quote.get("ask") and quote["ask"] >= quote["bid"] > 0:
        mid = (quote["ask"] + quote["bid"]) / 2
        row["spread_pct"] = (quote["ask"] - quote["bid"]) / mid * 100
    return row


def data_age_seconds(features: pd.DataFrame, now: datetime | None = None) -> float:
    """Seconds since the last completed bar finished."""
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert(NY)
    return float((now_ts - (features.index[-1] + pd.Timedelta(minutes=1))).total_seconds())
