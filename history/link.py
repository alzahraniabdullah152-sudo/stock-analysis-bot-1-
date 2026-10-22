"""Bridge between the live scanner and the history engine: descriptive context for a candidate ticker.

Answers: did similar daily patterns happen before, what followed, did volume breakouts here tend to hold, are there
financial/corporate risks, and does the stock behave differently from the market. It is DESCRIPTIVE history only:
statistics are withheld below `min_n` events, outcome windows only use completed past data, and nothing here predicts.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from history.sec import RISK_ITEMS, Filing

HORIZONS = (1, 5, 20)


def breakout_events(df: pd.DataFrame, lookback: int = 20, vol_mult: float = 2.0) -> pd.DataFrame:
    """Days that closed above the prior `lookback`-day high on >= vol_mult x the prior average volume.

    Detection at day t uses only days before t. Forward fields use later days and exist only once they completed.
    """
    c, h, v = df["Close"], df["High"], df["Volume"]
    prior_high = h.shift(1).rolling(lookback, min_periods=lookback).max()
    prior_vol = v.shift(1).rolling(lookback, min_periods=lookback).mean()
    hit = (c > prior_high) & (v >= vol_mult * prior_vol)
    idx = np.flatnonzero(hit.to_numpy())
    rows = []
    for i in idx:
        row = {"date": df.index[i], "close": float(c.iloc[i]), "level": float(prior_high.iloc[i]),
               "volume_x": float(v.iloc[i] / prior_vol.iloc[i])}
        for hz in HORIZONS:
            row[f"fwd_{hz}d_%"] = float((c.iloc[i + hz] / c.iloc[i] - 1) * 100) if i + hz < len(c) else np.nan
        look = c.iloc[i + 1:i + 6]
        row["failed_within_5d"] = bool((look < prior_high.iloc[i]).any()) if len(look) == 5 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def similar_patterns(df: pd.DataFrame, bench: pd.DataFrame | None = None, min_n: int = 10, **kw) -> dict:
    ev = breakout_events(df, **kw)
    out: dict = {"events": ev, "n": len(ev), "stats": None, "notes": []}
    if ev.empty:
        out["notes"].append("No comparable daily breakout on volume in this stock's available history.")
        return out
    if bench is not None and not bench.empty:
        b = bench["Close"].reindex(df.index, method="ffill")
        for hz in HORIZONS:
            col = f"fwd_{hz}d_%"
            pos = [df.index.get_loc(d) for d in ev["date"]]
            mret = [float((b.iloc[p + hz] / b.iloc[p] - 1) * 100) if p + hz < len(b) else np.nan for p in pos]
            ev[f"excess_{hz}d_%"] = ev[col] - np.array(mret)
    if len(ev) < min_n:
        out["notes"].append(f"Only {len(ev)} comparable events (fewer than {min_n}): listed for context, no statistics or odds are given.")
        return out
    stats = {}
    for hz in HORIZONS:
        s = ev[f"fwd_{hz}d_%"].dropna()
        if len(s) >= min_n:
            stats[f"{hz}d"] = {"n": int(len(s)), "median_%": float(s.median()), "share_positive": float((s > 0).mean()),
                               "p10_%": float(s.quantile(0.1)), "p90_%": float(s.quantile(0.9))}
    fail = ev["failed_within_5d"].dropna()
    stats["failed_within_5d_share"] = float(fail.mean()) if len(fail) >= min_n else None
    out["stats"] = stats
    out["notes"].append("A pattern repeating in the past does not mean it will repeat; events overlap in time and the sample is one stock.")
    return out


def candidate_context(df: pd.DataFrame, bench: pd.DataFrame | None, filings: list[Filing] | None,
                      splits: pd.Series | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    c = df["Close"]
    ctx: dict = {"as_of": df.index[-1]}
    ctx["from_52w_high_%"] = float((c.iloc[-1] / c.tail(252).max() - 1) * 100)
    ctx["return_20d_%"] = float((c.iloc[-1] / c.iloc[-21] - 1) * 100) if len(c) > 21 else np.nan
    if bench is not None and len(bench) > 21:
        b = bench["Close"].reindex(df.index, method="ffill")
        ctx["benchmark_20d_%"] = float((b.iloc[-1] / b.iloc[-21] - 1) * 100)
        ctx["vs_benchmark_20d_pp"] = ctx["return_20d_%"] - ctx["benchmark_20d_%"]
    risks = []
    for f in filings or []:
        hits = [i for i in f.items if i in RISK_ITEMS]
        if hits and f.accepted_utc and now - f.accepted_utc <= timedelta(days=90):
            risks.append({"date": f.filed, "form": f.form, "risk": "; ".join(RISK_ITEMS[i] for i in hits), "url": f.url})
    ctx["recent_filing_risks"] = risks
    if splits is not None and len(splits):
        recent = splits[splits.index >= (pd.Timestamp(now.date()) - pd.Timedelta(days=90))]
        ctx["recent_splits"] = [{"date": d, "ratio": float(r)} for d, r in recent.items()]
    else:
        ctx["recent_splits"] = []
    return ctx
