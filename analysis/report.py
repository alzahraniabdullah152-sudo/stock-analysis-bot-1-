"""Single analysis pipeline shared by the chart, dashboard and alerts (one DataFrame, one truth)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analysis.signal_engine import (Setup, compute_signals, confluence_score, historical_markers,
                                    latest_setups)
from analysis.support_resistance import Zone, find_zones, nearest
from analysis.trend_analysis import TrendState, analyze_trend, horizon_trends, mtf_agreement
from config import INTRADAY, SETTINGS, Settings
from data.market_data import MarketData
from indicators.technical_indicators import add_all, fibonacci_levels


@dataclass
class Report:
    md: MarketData
    d: pd.DataFrame
    trend: TrendState
    horizons: dict[str, str]
    zones: list[Zone]
    support: Zone | None
    resistance: Zone | None
    sigs: pd.DataFrame
    setups: list[Setup]
    markers: pd.DataFrame
    fib: dict[str, float]
    score: dict
    last_price: float
    key_levels: dict[str, float | None] = field(default_factory=dict)


def analyze(md: MarketData, mtf_states: dict[str, TrendState] | None = None,
            cfg: Settings = SETTINGS) -> Report:
    intraday = md.interval in INTRADAY
    d = add_all(md.df, intraday=intraday)
    trend = analyze_trend(d, cfg.adx_trend_min, cfg.sr_pivot_window)
    last = d.iloc[-1]
    atr = float(last["atr14"]) if pd.notna(last["atr14"]) else float("nan")
    zones = find_zones(d, atr, cfg.sr_pivot_window, cfg.sr_zone_atr_mult)
    sup, res = nearest(zones, float(last.Close))
    sigs = compute_signals(d, intraday, cfg)
    setups = latest_setups(d, md.ticker, md.interval, intraday, zones, md.last_bar_complete, cfg, sigs)
    swing = d.iloc[-60:]
    fib = fibonacci_levels(float(swing.High.max()), float(swing.Low.min()), trend.direction != "bearish")
    mtf_score = mtf_agreement(mtf_states)[1] if mtf_states and len(mtf_states) > 1 else None
    prior_hi = d["High"].iloc[-21:-1].max()
    prior_lo = d["Low"].iloc[-21:-1].min()
    return Report(
        md=md, d=d, trend=trend, horizons=horizon_trends(d), zones=zones, support=sup, resistance=res,
        sigs=sigs, setups=setups, markers=historical_markers(d, sigs), fib=fib,
        score=confluence_score(trend, last, mtf_score, cfg), last_price=float(last.Close),
        key_levels={"breakout": float(prior_hi), "breakdown": float(prior_lo)},
    )


def scenario_text(s: Setup) -> str:
    """What confirms / invalidates a hypothetical setup."""
    up = s.bias == "bullish"
    t1 = f"{s.targets[0]:.2f}" if s.targets else "n/a"
    return (f"Confirmation: price holds {'above' if up else 'below'} {s.entry:.2f} on the next completed candle(s). "
            f"Invalidation: close {'below' if up else 'above'} {s.invalidation:.2f}; stop reference {s.stop:.2f}. "
            f"First target {t1}.")
