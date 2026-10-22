"""Price-event detection on long daily history. Every definition is configurable and recorded with the event.

Uses only past data at each point where a rolling statistic is involved (no future volatility in thresholds).
A big move is not automatically a company-specific event: each event is compared with the market benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators.technical_indicators import rsi, sma


@dataclass
class EventConfig:
    surge_1d_pct: float = 8.0          # one-day move size (abs) ...
    z_min: float = 4.0                 # ... and at least this many trailing daily sigmas
    move_5d_pct: float = 20.0          # or a 5-day move this large
    vol_window: int = 60
    gap_pct: float = 5.0
    drawdown_pct: float = 20.0
    stagnation_days: int = 252
    stagnation_band: float = 1.35      # high/low ratio over the window considered "range-bound"
    cluster_gap_days: int = 2
    ath_min_days: int = 126
    ath_min_drawdown_pct: float = 10.0
    vol_percentile: float = 95.0
    vol_min_days: int = 5
    reversal_rebound_pct: float = 20.0
    reversal_window: int = 20
    market_driven_ratio: float = 0.5   # |excess| < ratio * |move| => mostly market-driven


def _clusters(flag: np.ndarray, gap: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(flag)
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > gap + 1:
            out.append((s, p))
            s = i
        p = i
    out.append((s, p))
    return out


def _window_ret(close: pd.Series, i0: int, i1: int) -> float:
    base = close.iloc[i0 - 1] if i0 > 0 else close.iloc[0]
    return float((close.iloc[i1] / base - 1) * 100)


def _split_artifact(df: pd.DataFrame, i0: int, i1: int, change_pct: float) -> bool:
    """True if a split sits next to the move and explains its size (data may be unadjusted)."""
    lo, hi = max(0, i0 - 2), min(len(df) - 1, i1 + 1)
    sp = df["Splits"].iloc[lo:hi + 1]
    sp = sp[sp > 0]
    for ratio in sp:
        expected = (1 / ratio - 1) * 100
        if abs(change_pct - expected) <= max(15.0, abs(expected) * 0.15):
            return True
    return False


def detect_events(df: pd.DataFrame, bench: pd.DataFrame | None = None, cfg: EventConfig | None = None) -> pd.DataFrame:
    """Return a table of events sorted by start date."""
    cfg = cfg or EventConfig()
    c, o, v = df["Close"], df["Open"], df["Volume"]
    n = len(df)
    rows: list[dict] = []
    ret1 = c.pct_change()
    sigma = ret1.rolling(cfg.vol_window, min_periods=cfg.vol_window // 2).std().shift(1)
    rsi14, sma200 = rsi(c, 14), sma(c, 200)
    bclose = bench["Close"].reindex(df.index, method="ffill") if bench is not None and not bench.empty else None

    def add(kind: str, i0: int, i1: int, **extra) -> None:
        i0, i1 = max(0, i0), min(n - 1, i1)
        before = float(c.iloc[i0 - 1]) if i0 > 0 else float(c.iloc[0])
        after = float(c.iloc[i1])
        chg = (after / before - 1) * 100
        pre = v.iloc[max(0, i0 - 50):i0]
        ok_pre = len(pre) >= 10 and pre.mean() > 0
        vm = float(v.iloc[i0:i1 + 1].mean() / pre.mean()) if ok_pre else np.nan
        vpk = float(v.iloc[i0:i1 + 1].max() / pre.mean()) if ok_pre else np.nan
        vpeak = float(v.iloc[i0:i1 + 1].max() / pre.mean()) if len(pre) >= 10 and pre.mean() > 0 else np.nan
        mret = np.nan
        if bclose is not None and bclose.iloc[[max(i0 - 1, 0), i1]].notna().all():
            mret = float((bclose.iloc[i1] / bclose.iloc[max(i0 - 1, 0)] - 1) * 100)
        excess = chg - mret if pd.notna(mret) else np.nan
        driven = bool(pd.notna(excess) and abs(chg) > 0 and abs(excess) < cfg.market_driven_ratio * abs(chg)
                      and np.sign(mret) == np.sign(chg))
        rows.append({"type": kind, "start": df.index[i0], "end": df.index[i1], "trading_days": i1 - i0 + 1,
                     "price_before": before, "price_after": after, "change_pct": chg, "volume_mult": vm, "volume_peak_mult": vpk, "volume_peak_mult": vpeak,
                     "market_ret_pct": mret, "excess_pct": excess, "market_driven": driven,
                     "rsi14_at_start": float(rsi14.iloc[i0]) if pd.notna(rsi14.iloc[i0]) else np.nan,
                     "vs_sma200_pct": float((c.iloc[i0] / sma200.iloc[i0] - 1) * 100) if pd.notna(sma200.iloc[i0]) else np.nan,
                     "split_artifact": _split_artifact(df, i0, i1, chg) if kind in ("surge", "plunge", "gap_up", "gap_down") else False,
                     **extra})

    # --- sharp moves (1-day with volatility context, or 5-day) -------------------------------------
    big1 = (ret1.abs() >= cfg.surge_1d_pct / 100) & ((ret1.abs() >= cfg.z_min * sigma) | sigma.isna())
    r5 = c / c.shift(5) - 1
    big5 = r5.abs() >= cfg.move_5d_pct / 100
    for sign, kind in ((1, "surge"), (-1, "plunge")):
        f1 = (big1 & (np.sign(ret1) == sign)).to_numpy()
        f5 = (big5 & (np.sign(r5) == sign)).to_numpy()
        flag = f1 | f5
        for s, e in _clusters(flag, cfg.cluster_gap_days):
            start = s - 4 if f5[s] and not f1[s] else s
            add(kind, start, e, trigger="1-day" if f1[s:e + 1].any() else "5-day")

    # --- gaps ----------------------------------------------------------------------------------------
    gap = o / c.shift(1) - 1
    gflag = gap.abs() >= cfg.gap_pct / 100
    for i in np.flatnonzero(gflag.to_numpy()):
        prev_close = float(c.iloc[i - 1])
        look = df.iloc[i:i + 6]
        filled = bool(((look["Low"] <= prev_close) if gap.iloc[i] > 0 else (look["High"] >= prev_close)).any())
        add("gap_up" if gap.iloc[i] > 0 else "gap_down", i, i, gap_pct=float(gap.iloc[i] * 100), gap_filled_5d=filled)

    # --- drawdown episodes, recoveries, reversals ---------------------------------------------------
    peak = c.cummax()
    dd = c / peak - 1
    grp = (dd == 0).cumsum().to_numpy()
    for g in np.unique(grp):
        pos = np.flatnonzero(grp == g)
        seg = dd.iloc[pos]
        if seg.min() * 100 > -cfg.drawdown_pct:
            continue
        trough_rel = int(np.argmin(seg.to_numpy()))
        ip, it = pos[0], pos[trough_rel]
        nxt = np.flatnonzero(grp == g + 1)
        rec = int(nxt[0]) if len(nxt) else None
        add("drawdown", ip, it, peak_date=df.index[ip], trough_date=df.index[it],
            recovery_date=df.index[rec] if rec is not None else pd.NaT,
            days_to_trough=int(it - ip), days_to_recover=int(rec - it) if rec is not None else np.nan,
            max_drawdown_pct=float(seg.min() * 100))
        if it - ip <= 2 * cfg.reversal_window and it + cfg.reversal_window < n:
            reb = float((c.iloc[it + cfg.reversal_window] / c.iloc[it] - 1) * 100)
            if reb >= cfg.reversal_rebound_pct:
                add("reversal_up", it, it + cfg.reversal_window, rebound_pct=reb)

    # --- long range-bound periods -------------------------------------------------------------------
    w = cfg.stagnation_days
    if n > w:
        band = c.rolling(w).max() / c.rolling(w).min()
        flat = (band <= cfg.stagnation_band).to_numpy()
        for s, e in _clusters(flat, 0):
            add("stagnation", max(0, s - w + 1), e, range_ratio=float(band.iloc[s:e + 1].mean()))

    # --- all-time-high breaks after a long gap --------------------------------------------------------
    run_max = c.cummax().shift(1)
    new_high = (c > run_max).to_numpy()
    last_ath_idx = -1
    for i in range(1, n):
        if new_high[i]:
            if last_ath_idx >= 0 and i - last_ath_idx >= cfg.ath_min_days:
                seg_low = c.iloc[last_ath_idx:i].min()
                if (seg_low / c.iloc[last_ath_idx] - 1) * 100 <= -cfg.ath_min_drawdown_pct:
                    add("ath_break", i, i, previous_ath_days_ago=int(i - last_ath_idx))
            last_ath_idx = i

    # --- abnormal-volatility regimes (threshold uses only past volatility) ---------------------------
    vol20 = ret1.rolling(20).std() * np.sqrt(252) * 100
    thr = vol20.expanding(min_periods=252).quantile(cfg.vol_percentile / 100).shift(1)
    hv = (vol20 > thr).to_numpy()
    for s, e in _clusters(hv, 1):
        if e - s + 1 >= cfg.vol_min_days:
            add("high_volatility", s, e, avg_vol_pct=float(vol20.iloc[s:e + 1].mean()))

    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    # a long episode that contains a split-sized artifact inherits the flag: its numbers are not trustworthy
    art = ev[ev["split_artifact"]]
    for k in ev.index[~ev["split_artifact"] & ev["type"].isin(["drawdown", "reversal_up", "stagnation", "high_volatility", "ath_break"])]:
        if ((art["start"] >= ev.at[k, "start"]) & (art["start"] <= ev.at[k, "end"])).any():
            ev.at[k, "split_artifact"] = True
    ev = ev.sort_values(["start", "type"]).reset_index(drop=True)
    ev.insert(0, "id", range(1, len(ev) + 1))
    return ev
