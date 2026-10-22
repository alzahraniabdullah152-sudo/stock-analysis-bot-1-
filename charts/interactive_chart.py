"""Plotly chart built from the SAME enriched DataFrame the dashboard analyses."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.signal_engine import Setup
from analysis.support_resistance import Zone

OVERLAYS = {
    "SMA 20": ("sma20", "#f4b400"), "SMA 50": ("sma50", "#4285f4"), "SMA 100": ("sma100", "#ab47bc"),
    "SMA 200": ("sma200", "#e53935"), "EMA 9": ("ema9", "#00acc1"), "EMA 21": ("ema21", "#43a047"),
    "EMA 50": ("ema50", "#8d6e63"),
}


def build_chart(d: pd.DataFrame, title: str, overlays: list[str], show_bb: bool, show_vwap: bool,
                zones: list[Zone] | None, markers: pd.DataFrame | None, setup: Setup | None,
                show_sr: bool = True, max_bars: int = 300) -> go.Figure:
    d = d.iloc[-max_bars:]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.55, 0.15, 0.15, 0.15],
                        vertical_spacing=0.02, subplot_titles=(title, "Volume", "RSI 14", "MACD 12/26/9"))
    # category-free time axis; hide gaps so weekends/overnight do not stretch the chart
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Price"), 1, 1)
    for name in overlays:
        col, color = OVERLAYS[name]
        if col in d and d[col].notna().any():
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=name, line=dict(width=1.2, color=color)), 1, 1)
    if show_bb and "bb_upper" in d:
        for col, nm in (("bb_upper", "BB upper"), ("bb_lower", "BB lower"), ("bb_mid", "BB mid")):
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=nm, line=dict(width=1, dash="dot", color="#9e9e9e")), 1, 1)
    if show_vwap and "vwap" in d:
        fig.add_trace(go.Scatter(x=d.index, y=d["vwap"], name="VWAP", line=dict(width=1.5, color="#ff6d00")), 1, 1)
    if show_sr and zones:
        lo, hi = d.Low.min(), d.High.max()
        for z in zones:
            if z.high < lo * 0.97 or z.low > hi * 1.03:
                continue
            fig.add_hrect(y0=z.low, y1=z.high, line_width=0, row=1, col=1,
                          fillcolor="rgba(67,160,71,0.15)" if z.kind == "support" else "rgba(229,57,53,0.15)")
    if markers is not None and not markers.empty:
        m = markers[markers.time >= d.index[0]]
        for bias, sym, color in (("bullish", "triangle-up", "#2e7d32"), ("bearish", "triangle-down", "#c62828"),
                                 ("mixed", "diamond", "#757575")):
            mm = m[m.bias == bias]
            if not mm.empty:
                fig.add_trace(go.Scatter(x=mm.time, y=mm.price, mode="markers", name=f"{bias} setup",
                                         text=mm.names, hoverinfo="text+x",
                                         marker=dict(symbol=sym, size=10, color=color)), 1, 1)
    if setup is not None:
        for y, label, color in [(setup.entry, "Entry ref", "#1565c0"), (setup.stop, "Stop ref", "#c62828"),
                                (setup.invalidation, "Invalidation", "#ef6c00")] + \
                               [(t, f"Target {i + 1}", "#2e7d32") for i, t in enumerate(setup.targets)]:
            fig.add_hline(y=y, line_dash="dash", line_color=color, annotation_text=f"{label} {y:.2f}",
                          annotation_position="right", row=1, col=1)
    up = d.Close >= d.Open
    fig.add_trace(go.Bar(x=d.index, y=d.Volume, marker_color=["#26a69a" if u else "#ef5350" for u in up],
                         name="Volume", showlegend=False), 2, 1)
    if "vol_ma20" in d:
        fig.add_trace(go.Scatter(x=d.index, y=d.vol_ma20, name="Vol MA20", line=dict(width=1, color="#ff9800")), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.rsi14, name="RSI", line=dict(color="#7e57c2")), 3, 1)
    for y in (30, 70):
        fig.add_hline(y=y, line_dash="dot", line_color="#9e9e9e", row=3, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d.macd_hist, name="Hist", marker_color="#90a4ae"), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.macd, name="MACD", line=dict(color="#1e88e5")), 4, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.macd_signal, name="Signal", line=dict(color="#fb8c00")), 4, 1)
    # remove non-trading gaps
    if len(d) > 2:
        diffs = d.index.to_series().diff().dropna()
        if diffs.median() < pd.Timedelta(days=1):
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
        else:
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, hovermode="x unified",
                      margin=dict(l=10, r=60, t=40, b=10), legend=dict(orientation="h", y=1.04))
    return fig
