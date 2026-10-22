"""Vectorised candlestick pattern detection. Each flag uses only the current and prior bars."""
from __future__ import annotations

import numpy as np
import pandas as pd


def candle_patterns(df: pd.DataFrame, atr: pd.Series | None = None) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    out = pd.DataFrame(index=df.index)
    out["doji"] = body <= 0.1 * rng
    out["hammer"] = (lower >= 2 * body) & (upper <= 0.3 * rng) & (body > 0)
    out["shooting_star"] = (upper >= 2 * body) & (lower <= 0.3 * rng) & (body > 0)
    po, pc = o.shift(1), c.shift(1)
    out["bull_engulfing"] = (pc < po) & (c > o) & (c >= po) & (o <= pc)
    out["bear_engulfing"] = (pc > po) & (c < o) & (c <= po) & (o >= pc)
    out["inside_bar"] = (h < h.shift(1)) & (l > l.shift(1))
    out["upper_wick_rejection"] = (upper >= 0.5 * rng) & (c < o)
    out["lower_wick_rejection"] = (lower >= 0.5 * rng) & (c > o)
    return out.fillna(False)


def pattern_names(row: pd.Series) -> list[str]:
    return [k for k, v in row.items() if bool(v)]
