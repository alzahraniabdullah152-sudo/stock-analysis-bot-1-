"""Pivot-based support/resistance zones. Pivots are only confirmed `window` bars later."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Zone:
    low: float
    high: float
    touches: int
    kind: str  # "support" | "resistance"

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2


def pivots(df: pd.DataFrame, window: int = 5) -> tuple[pd.Series, pd.Series]:
    """Confirmed swing highs/lows. A pivot at bar i is only *known* at bar i+window.

    Returned series are indexed at the pivot bar; callers must respect confirmation lag
    (see ``confirmed_pivots``) to avoid look-ahead.
    """
    h, l = df["High"], df["Low"]
    size = 2 * window + 1
    is_hi = h == h.rolling(size, center=True, min_periods=size).max()
    is_lo = l == l.rolling(size, center=True, min_periods=size).min()
    return h.where(is_hi), l.where(is_lo)


def confirmed_pivots(df: pd.DataFrame, window: int = 5) -> tuple[pd.Series, pd.Series]:
    """Pivots usable at the last bar (the final `window` bars cannot hold confirmed pivots)."""
    ph, pl = pivots(df, window)
    return ph.dropna(), pl.dropna()


def find_zones(df: pd.DataFrame, atr_value: float, window: int = 5, zone_atr_mult: float = 0.5) -> list[Zone]:
    """Cluster pivot prices within zone_atr_mult*ATR into zones, tagged vs the last close."""
    ph, pl = confirmed_pivots(df, window)
    prices = sorted(list(ph.values) + list(pl.values))
    if not prices or not np.isfinite(atr_value) or atr_value <= 0:
        return []
    tol = atr_value * zone_atr_mult
    clusters: list[list[float]] = [[prices[0]]]
    for p in prices[1:]:
        if p - np.mean(clusters[-1]) <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    last = float(df["Close"].iloc[-1])
    zones = []
    for c in clusters:
        lo, hi = min(c), max(c)
        mid = (lo + hi) / 2
        zones.append(Zone(lo, hi, len(c), "support" if mid <= last else "resistance"))
    return zones


def nearest(zones: list[Zone], price: float) -> tuple[Zone | None, Zone | None]:
    """(nearest support at/below price, nearest resistance above price)."""
    sup = [z for z in zones if z.high <= price]
    res = [z for z in zones if z.low > price]
    s = max(sup, key=lambda z: z.high) if sup else None
    r = min(res, key=lambda z: z.low) if res else None
    return s, r
