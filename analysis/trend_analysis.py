"""Trend, market structure, momentum, and multi-timeframe agreement."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.support_resistance import confirmed_pivots


@dataclass
class TrendState:
    direction: str  # bullish | bearish | neutral
    structure: str  # HH/HL | LH/LL | mixed | insufficient
    momentum: str  # bullish | bearish | neutral
    volume_confirms: bool | None
    adx: float | None
    reasons: list[str] = field(default_factory=list)


def market_structure(df: pd.DataFrame, window: int = 5) -> str:
    ph, pl = confirmed_pivots(df, window)
    if len(ph) < 2 or len(pl) < 2:
        return "insufficient"
    hh = ph.iloc[-1] > ph.iloc[-2]
    hl = pl.iloc[-1] > pl.iloc[-2]
    if hh and hl:
        return "HH/HL"
    if not hh and not hl:
        return "LH/LL"
    return "mixed"


def analyze_trend(d: pd.DataFrame, adx_min: float = 20.0, window: int = 5) -> TrendState:
    """`d` must already contain indicators from ``add_all``. Uses the last row only."""
    r = d.iloc[-1]
    score, reasons = 0, []

    def chk(cond: bool | float, bull_txt: str, bear_txt: str, ok: bool) -> None:
        nonlocal score
        if not ok:
            return
        if cond:
            score += 1
            reasons.append(bull_txt)
        else:
            score -= 1
            reasons.append(bear_txt)

    chk(r.Close > r.get("sma50", np.nan), "Close > SMA50", "Close < SMA50", pd.notna(r.get("sma50")))
    chk(r.get("ema9", np.nan) > r.get("ema21", np.nan), "EMA9 > EMA21", "EMA9 < EMA21", pd.notna(r.get("ema21")))
    chk(r.get("sma50", np.nan) > r.get("sma200", np.nan), "SMA50 > SMA200", "SMA50 < SMA200", pd.notna(r.get("sma200")))
    adx_v = float(r["adx"]) if pd.notna(r.get("adx")) else None
    if adx_v is not None and adx_v >= adx_min:
        chk(r.plus_di > r.minus_di, f"ADX {adx_v:.0f} with +DI > -DI", f"ADX {adx_v:.0f} with -DI > +DI", True)
    structure = market_structure(d, window)
    if structure == "HH/HL":
        score += 1; reasons.append("Higher highs & higher lows")
    elif structure == "LH/LL":
        score -= 1; reasons.append("Lower highs & lower lows")
    direction = "bullish" if score >= 2 else "bearish" if score <= -2 else "neutral"

    mom = 0
    if pd.notna(r.get("rsi14")):
        mom += 1 if r.rsi14 > 55 else -1 if r.rsi14 < 45 else 0
    if pd.notna(r.get("macd_hist")):
        mom += 1 if r.macd_hist > 0 else -1
    momentum = "bullish" if mom >= 1 else "bearish" if mom <= -1 else "neutral"

    vol_ok: bool | None = None
    if pd.notna(r.get("rel_vol")):
        up_bar = r.Close >= r.Open
        vol_ok = bool(r.rel_vol >= 1.0 and ((direction == "bullish" and up_bar) or (direction == "bearish" and not up_bar)))
    return TrendState(direction, structure, momentum, vol_ok, adx_v, reasons)


def horizon_trends(d: pd.DataFrame) -> dict[str, str]:
    """Short (EMA9/21), medium (SMA20/50) and long (SMA50/200) trend labels."""
    r = d.iloc[-1]

    def lab(a: str, b: str) -> str:
        if pd.isna(r.get(a)) or pd.isna(r.get(b)):
            return "n/a"
        return "bullish" if r[a] > r[b] else "bearish"

    return {"short": lab("ema9", "ema21"), "medium": lab("sma20", "sma50"), "long": lab("sma50", "sma200")}


def mtf_agreement(states: dict[str, TrendState]) -> tuple[str, float]:
    """Return (summary text, agreement in [-1, 1]) across timeframes.

    Higher timeframes weigh more: lower-timeframe setups against the higher-timeframe trend are less reliable.
    """
    order = ["5m", "15m", "1h", "4h", "1d", "1wk"]
    weights = {k: i + 1 for i, k in enumerate(order)}
    num = den = 0.0
    for tf, s in states.items():
        v = {"bullish": 1, "bearish": -1, "neutral": 0}[s.direction]
        w = weights.get(tf, 1)
        num += v * w
        den += w
    score = num / den if den else 0.0
    dirs = {s.direction for s in states.values()}
    if len(dirs) == 1:
        text = f"All timeframes {dirs.pop()}"
    elif "bullish" in dirs and "bearish" in dirs:
        text = "Conflict between timeframes"
    else:
        text = "Mixed / partially aligned"
    return text, score
