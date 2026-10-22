"""Rule-based setup detection.

All rules use only data up to and including the evaluated bar (rolling/shift only).
Pivot-based rules are attributed to the bar where the pivot was *confirmed*
(pivot bar + window), never to the pivot bar itself, so historical signals never repaint.
Signals are evaluated on completed candles; entry in the backtester is the NEXT bar's open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.pattern_detection import candle_patterns
from analysis.risk_management import risk_reward
from analysis.support_resistance import Zone, nearest, pivots
from analysis.trend_analysis import TrendState
from config import SETTINGS, Settings
from indicators.technical_indicators import prev_period_levels

SETUP_LABELS = {
    "ema_cross": "EMA 9/21 crossover with trend confirmation",
    "ma_pullback": "Pullback to EMA21 in trend",
    "breakout": "20-bar breakout with volume",
    "macd_shift": "MACD momentum shift",
    "rsi_divergence": "RSI divergence",
    "sr_rejection": "Rejection at swing level",
    "failed_break": "Failed breakout/breakdown",
    "vwap_cross": "VWAP reclaim/rejection",
    "orb": "Opening-range breakout",
    "rvol_expansion": "Relative-volume expansion",
    "prev_level_break": "Previous-period high/low break",
}
INTRADAY_ONLY = {"vwap_cross", "orb"}


@dataclass
class Setup:
    ticker: str
    timeframe: str
    name: str
    label: str
    bias: str
    conditions: list[str]
    entry: float
    invalidation: float
    stop: float
    targets: list[float]
    rr: float | None
    supporting: list[str]
    conflicting: list[str]
    timestamp: pd.Timestamp
    freshness: str
    position: int = -1


def _s(x: pd.Series) -> pd.Series:
    return x.fillna(False).astype(bool)


def rsi_divergence(d: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    """Regular divergence between consecutive confirmed swing points and RSI."""
    n = len(d)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    ph, pl = pivots(d, window)
    rsi_v = d["rsi14"].to_numpy()
    for pv, series, is_low in ((pl, d["Low"].to_numpy(), True), (ph, d["High"].to_numpy(), False)):
        pos = np.flatnonzero(pv.notna().to_numpy())
        for a, b in zip(pos[:-1], pos[1:]):
            if not (window <= b - a <= 60) or np.isnan(rsi_v[a]) or np.isnan(rsi_v[b]):
                continue
            conf = b + window
            if conf >= n:
                continue
            if is_low and series[b] < series[a] and rsi_v[b] > rsi_v[a] and rsi_v[a] < 45:
                bull[conf] = True
            if (not is_low) and series[b] > series[a] and rsi_v[b] < rsi_v[a] and rsi_v[a] > 55:
                bear[conf] = True
    return pd.Series(bull, index=d.index), pd.Series(bear, index=d.index)


def compute_signals(d: pd.DataFrame, intraday: bool = False, cfg: Settings = SETTINGS) -> pd.DataFrame:
    """Return a frame of +1 (bullish) / -1 (bearish) / 0 per setup per bar. `d` needs add_all columns."""
    w = cfg.sr_pivot_window
    c, o, h, l = d["Close"], d["Open"], d["High"], d["Low"]
    atr = d["atr14"]
    pat = candle_patterns(d)
    prior_hi = h.rolling(20, min_periods=20).max().shift(1)
    prior_lo = l.rolling(20, min_periods=20).min().shift(1)
    rv = d["rel_vol"]
    out: dict[str, pd.Series] = {}

    def make(bull: pd.Series, bear: pd.Series) -> pd.Series:
        return _s(bull).astype(int) - _s(bear).astype(int)

    # A. trend-following
    up = (d["ema9"] > d["ema21"]) & (d["ema9"].shift(1) <= d["ema21"].shift(1))
    dn = (d["ema9"] < d["ema21"]) & (d["ema9"].shift(1) >= d["ema21"].shift(1))
    out["ema_cross"] = make(
        up & (c > d["sma50"]) & (d["adx"] >= cfg.adx_trend_min) & (d["plus_di"] > d["minus_di"]),
        dn & (c < d["sma50"]) & (d["adx"] >= cfg.adx_trend_min) & (d["minus_di"] > d["plus_di"]),
    )
    out["ma_pullback"] = make(
        (d["ema21"] > d["sma50"]) & (l <= d["ema21"]) & (c > d["ema21"]) & (c > o),
        (d["ema21"] < d["sma50"]) & (h >= d["ema21"]) & (c < d["ema21"]) & (c < o),
    )
    out["breakout"] = make(
        (c > prior_hi) & (rv >= cfg.breakout_rel_volume) & (c > o),
        (c < prior_lo) & (rv >= cfg.breakout_rel_volume) & (c < o),
    )
    # B. reversal
    m, ms = d["macd"], d["macd_signal"]
    out["macd_shift"] = make(
        (m > ms) & (m.shift(1) <= ms.shift(1)) & (m < 0),
        (m < ms) & (m.shift(1) >= ms.shift(1)) & (m > 0),
    )
    dbull, dbear = rsi_divergence(d, w)
    out["rsi_divergence"] = make(dbull, dbear)
    ph, pl = pivots(d, w)
    ph_conf, pl_conf = ph.shift(w).ffill(), pl.shift(w).ffill()
    out["sr_rejection"] = make(
        (l <= pl_conf + 0.25 * atr) & (c > pl_conf) & (pat["hammer"] | pat["bull_engulfing"] | pat["lower_wick_rejection"]),
        (h >= ph_conf - 0.25 * atr) & (c < ph_conf) & (pat["shooting_star"] | pat["bear_engulfing"] | pat["upper_wick_rejection"]),
    )
    out["failed_break"] = make(
        (c.shift(1) < prior_lo.shift(1)) & (c > prior_lo.shift(1)),
        (c.shift(1) > prior_hi.shift(1)) & (c < prior_hi.shift(1)),
    )
    # C. intraday / volume / levels
    out["rvol_expansion"] = make(
        (rv >= 2.0) & (c > o) & ((c - o) >= 0.5 * atr),
        (rv >= 2.0) & (c < o) & ((o - c) >= 0.5 * atr),
    )
    if intraday and "vwap" in d:
        vw = d["vwap"]
        out["vwap_cross"] = make(
            (c > vw) & (c.shift(1) <= vw.shift(1)) & (rv >= 1.0),
            (c < vw) & (c.shift(1) >= vw.shift(1)) & (rv >= 1.0),
        )
        out["orb"] = _opening_range_signal(d, cfg)
    pv = prev_period_levels(d, "D" if intraday else "W")
    out["prev_level_break"] = make(
        (c > pv["prev_high"]) & (c.shift(1) <= pv["prev_high"]) & (rv >= 1.0),
        (c < pv["prev_low"]) & (c.shift(1) >= pv["prev_low"]) & (rv >= 1.0),
    )
    return pd.DataFrame(out, index=d.index).fillna(0).astype(int)


def _opening_range_signal(d: pd.DataFrame, cfg: Settings) -> pd.Series:
    idx = d.index.tz_convert("America/New_York") if d.index.tz is not None else d.index
    day = pd.Series(idx.normalize(), index=d.index)
    step = pd.Series(idx).diff().dt.total_seconds().median() / 60
    if not np.isfinite(step) or step <= 0:
        return pd.Series(0, index=d.index)
    n_or = max(1, int(round(cfg.opening_range_minutes / step)))
    k = d.groupby(day).cumcount()
    in_or = k < n_or
    or_hi = d["High"].where(in_or).groupby(day).transform("max")
    or_lo = d["Low"].where(in_or).groupby(day).transform("min")
    after = ~in_or
    prev_c = d["Close"].groupby(day).shift(1)
    rv = d["rel_vol"]
    bull = after & (d["Close"] > or_hi) & (prev_c <= or_hi) & (rv >= 1.0)
    bear = after & (d["Close"] < or_lo) & (prev_c >= or_lo) & (rv >= 1.0)
    return _s(bull).astype(int) - _s(bear).astype(int)


def _evidence(r: pd.Series, bias: str) -> tuple[list[str], list[str]]:
    want_bull = bias == "bullish"
    checks: list[tuple[str, bool]] = []

    def add(label_bull: str, label_bear: str, cond: bool | None) -> None:
        if cond is None or (isinstance(cond, float) and np.isnan(cond)):
            return
        checks.append((label_bull if cond else label_bear, bool(cond)))

    def ok(*keys: str) -> bool:
        return all(k in r.index and pd.notna(r[k]) for k in keys)

    if ok("sma50"): add("Close above SMA50", "Close below SMA50", r.Close > r.sma50)
    if ok("ema9", "ema21"): add("EMA9 above EMA21", "EMA9 below EMA21", r.ema9 > r.ema21)
    if ok("macd_hist"): add("MACD histogram positive", "MACD histogram negative", r.macd_hist > 0)
    if ok("rsi14"): add(f"RSI {r.rsi14:.0f} > 50", f"RSI {r.rsi14:.0f} < 50", r.rsi14 > 50)
    if ok("plus_di", "minus_di"): add("+DI above -DI", "-DI above +DI", r.plus_di > r.minus_di)
    if ok("vwap"): add("Above VWAP", "Below VWAP", r.Close > r.vwap)
    sup = [t for t, b in checks if b == want_bull]
    con = [t for t, b in checks if b != want_bull]
    return sup, con


def _structure_stop(entry: float, atr: float, level: float | None, bull: bool, cfg: Settings) -> float:
    atr_stop = entry - cfg.atr_stop_mult * atr if bull else entry + cfg.atr_stop_mult * atr
    if level is None or not np.isfinite(level):
        return atr_stop
    s = level - 0.1 * atr if bull else level + 0.1 * atr
    dist = (entry - s) if bull else (s - entry)
    return s if 0.5 * atr <= dist <= 3 * atr else atr_stop


def _conditions(name: str, bias: str, r: pd.Series, cfg: Settings) -> list[str]:
    b = bias == "bullish"
    f = {
        "ema_cross": [f"EMA9 crossed {'above' if b else 'below'} EMA21", f"Close {'>' if b else '<'} SMA50",
                      f"ADX {r.get('adx', np.nan):.1f} >= {cfg.adx_trend_min}", f"{'+DI > -DI' if b else '-DI > +DI'}"],
        "ma_pullback": [f"EMA21 {'>' if b else '<'} SMA50 (trend)", f"Bar {'dipped to' if b else 'popped to'} EMA21",
                        f"Closed {'above' if b else 'below'} EMA21 with {'bullish' if b else 'bearish'} candle"],
        "breakout": [f"Close {'above prior 20-bar high' if b else 'below prior 20-bar low'}",
                     f"Relative volume {r.get('rel_vol', np.nan):.2f} >= {cfg.breakout_rel_volume}"],
        "macd_shift": [f"MACD crossed {'above' if b else 'below'} signal line on the {'negative' if b else 'positive'} side of zero"],
        "rsi_divergence": [f"{'Bullish' if b else 'Bearish'} RSI divergence between the last two confirmed swing points"],
        "sr_rejection": [f"{'Support' if b else 'Resistance'} swing level tested with a rejection candle"],
        "failed_break": [f"Prior bar's {'breakdown' if b else 'breakout'} closed back {'above' if b else 'below'} the 20-bar level"],
        "vwap_cross": [f"Close crossed {'above' if b else 'below'} VWAP with relative volume >= 1"],
        "orb": [f"Close broke {'above' if b else 'below'} the {cfg.opening_range_minutes}-minute opening range", "Relative volume >= 1"],
        "rvol_expansion": [f"Relative volume {r.get('rel_vol', np.nan):.2f} >= 2 with a strong {'up' if b else 'down'} bar"],
        "prev_level_break": [f"Close broke {'above previous-period high' if b else 'below previous-period low'}"],
    }
    return f.get(name, [name])


def build_setup(d: pd.DataFrame, sigs: pd.DataFrame, name: str, pos: int, ticker: str, tf: str,
                cfg: Settings = SETTINGS, zones: list[Zone] | None = None, complete: bool = True,
                pl_conf: pd.Series | None = None, ph_conf: pd.Series | None = None) -> Setup | None:
    v = int(sigs[name].iloc[pos])
    if v == 0:
        return None
    r = d.iloc[pos]
    atr = float(r["atr14"]) if pd.notna(r["atr14"]) else np.nan
    if not np.isfinite(atr) or atr <= 0:
        return None
    bull = v > 0
    entry = float(r.Close)
    lvl = None
    if pl_conf is not None and ph_conf is not None:
        lvl = pl_conf.iloc[pos] if bull else ph_conf.iloc[pos]
    stop = _structure_stop(entry, atr, lvl, bull, cfg)
    risk = abs(entry - stop)
    targets = [entry + 2 * risk, entry + 3 * risk] if bull else [entry - 2 * risk, entry - 3 * risk]
    if zones:
        s, rz = nearest(zones, entry)
        z = rz if bull else s
        if z is not None:
            zt = z.low if bull else z.high
            if abs(zt - entry) >= risk:  # only keep zone targets worth at least 1R
                targets = sorted(set(targets + [zt]), reverse=not bull)
    invalid = float(r.Low if bull else r.High)
    if name == "breakout":
        prior = d["High"].rolling(20).max().shift(1) if bull else d["Low"].rolling(20).min().shift(1)
        invalid = float(prior.iloc[pos])
    bias = "bullish" if bull else "bearish"
    sup, con = _evidence(r, bias)
    ts = d.index[pos]
    last = pos == len(d) - 1
    fresh = f"signal bar {ts}; {'completed candle' if complete or not last else 'PROVISIONAL (unfinished) candle'}"
    return Setup(
        ticker=ticker, timeframe=tf, name=name, label=SETUP_LABELS[name], bias=bias,
        conditions=_conditions(name, bias, r, cfg), entry=entry, invalidation=invalid, stop=stop,
        targets=targets, rr=risk_reward(entry, stop, targets[0]), supporting=sup, conflicting=con,
        timestamp=ts, freshness=fresh, position=pos,
    )


def confirmed_levels(d: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    """Last confirmed swing high/low available at each bar (forward-filled, lag = window)."""
    ph, pl = pivots(d, window)
    return ph.shift(window).ffill(), pl.shift(window).ffill()


def latest_setups(d: pd.DataFrame, ticker: str, tf: str, intraday: bool, zones: list[Zone] | None = None,
                  complete: bool = True, cfg: Settings = SETTINGS, sigs: pd.DataFrame | None = None) -> list[Setup]:
    sigs = compute_signals(d, intraday, cfg) if sigs is None else sigs
    ph_c, pl_c = confirmed_levels(d, cfg.sr_pivot_window)
    pos = len(d) - 1
    res = [build_setup(d, sigs, n, pos, ticker, tf, cfg, zones, complete, pl_c, ph_c) for n in sigs.columns]
    return [s for s in res if s is not None]


def historical_markers(d: pd.DataFrame, sigs: pd.DataFrame) -> pd.DataFrame:
    """Rows (time, price, bias, names) for chart markers."""
    rows = []
    for i in range(len(d)):
        row = sigs.iloc[i]
        names = [n for n, v in row.items() if v != 0]
        if not names:
            continue
        bull = sum(1 for n in names if row[n] > 0)
        bear = len(names) - bull
        bias = "bullish" if bull > bear else "bearish" if bear > bull else "mixed"
        rows.append({"time": d.index[i], "bias": bias, "names": ", ".join(SETUP_LABELS[n] for n in names),
                     "price": d["Low"].iloc[i] if bias != "bearish" else d["High"].iloc[i]})
    return pd.DataFrame(rows, columns=["time", "bias", "names", "price"])


def confluence_score(trend: TrendState, last_row: pd.Series, mtf_score: float | None,
                     cfg: Settings = SETTINGS) -> dict:
    """Heuristic bullish-alignment score 0-100 (50 = neutral). NOT a probability of profit.

    score = 50 + 50 * sum(w_i * c_i) / sum(w_i)  over components with available data,
    c_i in [-1, 1]. Weights (cfg.score_weights): trend 30, momentum 20, volume 15,
    structure 15, mtf 20. Unavailable components are excluded and the rest re-normalised.
    """
    sign = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
    comps: dict[str, float | None] = {}
    comps["trend"] = sign[trend.direction] if pd.notna(last_row.get("sma50")) else None
    comps["momentum"] = sign[trend.momentum] if pd.notna(last_row.get("rsi14")) else None
    rv = last_row.get("rel_vol")
    if pd.notna(rv):
        direction = 1.0 if last_row.Close >= last_row.Open else -1.0
        comps["volume"] = direction * min(float(rv) / 1.5, 1.0) if rv >= 1.0 else 0.0
    else:
        comps["volume"] = None
    comps["structure"] = {"HH/HL": 1.0, "LH/LL": -1.0, "mixed": 0.0}.get(trend.structure)
    comps["mtf"] = mtf_score
    num = den = 0.0
    for k, v in comps.items():
        if v is None:
            continue
        wt = cfg.score_weights[k]
        num += wt * v
        den += wt
    score = 50 + 50 * num / den if den > 0 else None
    return {
        "score": None if score is None else round(score, 1),
        "components": comps,
        "weights": dict(cfg.score_weights),
        "note": "Heuristic alignment score (0 bearish, 50 neutral, 100 bullish). Not a probability of profit.",
    }
