"""Strict replay-based evaluation of the scanner rules on 1-minute bars.

Rules (all enforced in code and covered by tests):
  * Features at bar i use only bars <= i (see scanner.features). Signals are evaluated on COMPLETED bars.
  * A signal on bar i is filled at the OPEN of bar i+1, plus slippage and commission (assumed, not measured:
    free bar data has no bid/ask).
  * If the target and the hypothetical stop fall inside the same 1-minute bar the order of events is unknown:
    the result is 'ambiguous' and counts as NOT a win (never the most profitable ordering).
  * Gaps through the stop fill at the open (worse price); a gap above the target still counts as the target only.
  * Nothing is tuned here. Weights/thresholds are taken from the config as given; the chronological
    development / hold-out split is only a reporting split, so tune on development sessions only.
  * Baselines (random bars, simple momentum) use the same universe, costs, targets and de-duplication.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scanner.config import ScannerConfig
from scanner.features import build_features
from scanner.scoring import score_row
from scanner.states import classify


@dataclass
class ReplayConfig:
    slippage_pct: float = 0.10          # per side, assumed
    commission_per_share: float = 0.0
    trigger_states: tuple[str, ...] = ("confirmed",)
    min_signals_for_stats: int = 30     # below this, rates are shown with a warning, never as a conclusion
    random_trials: int = 200
    seed: int = 7


# ----------------------------------------------------------------------------------------------------
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Minutes since the first bar as floats (independent of the index's datetime resolution)."""
    return np.asarray((index - index[0]) / pd.Timedelta(minutes=1), dtype="float64")


def _arrays(f: pd.DataFrame):
    return (_minutes(f.index), f["Open"].to_numpy(), f["High"].to_numpy(), f["Low"].to_numpy(),
            f["Close"].to_numpy(), np.array([x.toordinal() for x in f["session"]]))


def trade_outcome(f_arrays, i: int, target_pct: float, horizon_min: int, stop_pct: float, rc: ReplayConfig) -> dict:
    """Hypothetical trade for a signal on completed bar i; entry at open of bar i+1."""
    ts, op, hi, lo, cl, sess = f_arrays
    n = len(ts)
    j0 = i + 1
    if j0 >= n or sess[j0] != sess[i]:
        return {"status": "no_data", "net_ret": np.nan}
    entry = op[j0] * (1 + rc.slippage_pct / 100)
    cost = 2 * rc.slippage_pct + (2 * rc.commission_per_share / entry * 100 if rc.commission_per_share else 0.0)
    tgt, stp = entry * (1 + target_pct / 100), entry * (1 - stop_pct / 100)
    end_ts = ts[j0] + horizon_min
    j = j0
    last = j0
    while j < n and sess[j] == sess[j0] and ts[j] <= end_ts:
        last = j
        hit_t, hit_s = hi[j] >= tgt, lo[j] <= stp
        if hit_t and hit_s:
            return {"status": "ambiguous", "net_ret": -stop_pct - cost, "minutes": np.nan,
                    "mfe": (hi[j0:j + 1].max() / entry - 1) * 100, "mae": (lo[j0:j + 1].min() / entry - 1) * 100}
        if hit_s:
            fill = min(op[j], stp) if j > j0 else stp           # a gap through the stop fills at the open
            return {"status": "miss", "reason": "stop", "net_ret": (fill / entry - 1) * 100 - cost, "minutes": np.nan,
                    "mfe": (hi[j0:j + 1].max() / entry - 1) * 100, "mae": (lo[j0:j + 1].min() / entry - 1) * 100}
        if hit_t:
            return {"status": "hit", "reason": "target", "net_ret": target_pct - cost,
                    "minutes": float(ts[j] - ts[j0]), "mfe": (hi[j0:j + 1].max() / entry - 1) * 100,
                    "mae": (lo[j0:j + 1].min() / entry - 1) * 100}
        j += 1
    full_horizon = ts[last] >= end_ts - 1.0 or (j < n and sess[j] == sess[j0])
    if not full_horizon:  # the session ended before the horizon: not comparable, excluded from rates
        return {"status": "no_data", "net_ret": np.nan}
    return {"status": "miss", "reason": "timeout", "net_ret": (cl[last] / entry - 1) * 100 - cost, "minutes": np.nan,
            "mfe": (hi[j0:last + 1].max() / entry - 1) * 100, "mae": (lo[j0:last + 1].min() / entry - 1) * 100}


# ----------------------------------------------------------------------------------------------------
def generate_signals(f: pd.DataFrame, cfg: ScannerConfig, rc: ReplayConfig, ticker: str = "") -> pd.DataFrame:
    """Walk the bars in time order; emit a signal when a trigger state first appears, then require a reset."""
    rows = []
    in_episode, cooldown_until = False, -1.0
    ts_ns = _minutes(f.index)
    for i in range(len(f)):
        row = f.iloc[i]
        if pd.isna(row.get("avg_dollar_vol")) or pd.isna(row.get("rvol")):
            continue
        px, dv = row["Close"], row["avg_dollar_vol"]
        if not (cfg.min_price <= px <= cfg.max_price) or dv < cfg.min_avg_dollar_volume:
            in_episode = False
            continue
        sc = score_row(row, cfg)
        cl = classify(row, sc, cfg, market_open=True, data_age_s=0.0, signal_time=f.index[i].to_pydatetime())
        trig = cl.state in rc.trigger_states and cl.actionable
        if not trig:
            if cl.state in ("none", "watch", "failed", "expired"):
                in_episode = False
            continue
        if in_episode or ts_ns[i] < cooldown_until:
            continue
        in_episode = True
        cooldown_until = ts_ns[i] + cfg.cooldown_minutes
        rows.append({"ticker": ticker, "bar_time": f.index[i], "i": i, "state": cl.state, "score": sc.score,
                     "coverage": sc.coverage, "rvol": row["rvol"], "ret5": row["ret5"], "session": row["session"]})
    return pd.DataFrame(rows)


def momentum_baseline(f: pd.DataFrame, cfg: ScannerConfig, ret15_min: float = 1.5, ticker: str = "") -> pd.DataFrame:
    """Simple momentum rule with NO volume / breakout / score logic: 15-minute return above a threshold."""
    rows, last_t, in_ep = [], -1.0, False
    ts_ns = _minutes(f.index)
    for i in range(len(f)):
        r = f.iloc[i]
        ok = (pd.notna(r["avg_dollar_vol"]) and cfg.min_price <= r["Close"] <= cfg.max_price
              and r["avg_dollar_vol"] >= cfg.min_avg_dollar_volume and pd.notna(r["ret15"]) and r["ret15"] >= ret15_min)
        if not ok:
            in_ep = False
            continue
        if in_ep or ts_ns[i] < last_t:
            continue
        in_ep, last_t = True, ts_ns[i] + cfg.cooldown_minutes
        rows.append({"ticker": ticker, "bar_time": f.index[i], "i": i, "state": "momentum", "score": np.nan,
                     "session": r["session"]})
    return pd.DataFrame(rows)


def random_baseline(f: pd.DataFrame, cfg: ScannerConfig, n_signals: int, rng: np.random.Generator,
                    ticker: str = "") -> pd.DataFrame:
    """Random eligible bars (same price/liquidity filters, after the open window) as a no-skill reference."""
    ok = (f["avg_dollar_vol"].notna() & (f["avg_dollar_vol"] >= cfg.min_avg_dollar_volume)
          & f["Close"].between(cfg.min_price, cfg.max_price) & (f["minute"] >= cfg.skip_open_minutes))
    idx = np.flatnonzero(ok.to_numpy())
    if len(idx) == 0 or n_signals == 0:
        return pd.DataFrame()
    pick = np.sort(rng.choice(idx, size=min(n_signals, len(idx)), replace=False))
    return pd.DataFrame({"ticker": ticker, "bar_time": f.index[pick], "i": pick, "state": "random",
                         "score": np.nan, "session": f["session"].to_numpy()[pick]})


def evaluate(signals: pd.DataFrame, f: pd.DataFrame, cfg: ScannerConfig, rc: ReplayConfig) -> pd.DataFrame:
    if signals.empty:
        return signals.assign()
    arr = _arrays(f)
    out = []
    for s in signals.itertuples():
        for tgt, h in cfg.targets:
            o = trade_outcome(arr, int(s.i), tgt, int(h), cfg.stop_pct, rc)
            out.append({"ticker": s.ticker, "bar_time": s.bar_time, "session": s.session, "score": s.score,
                        "state": s.state, "target_pct": tgt, "horizon_min": h, **o})
    return pd.DataFrame(out)


def summarize(res: pd.DataFrame, rc: ReplayConfig, k_top: int = 3) -> pd.DataFrame:
    """Per target: counts, precision with Wilson CI, net-return stats, drawdown, Precision@Top-K per session."""
    rows = []
    if res.empty:
        return pd.DataFrame()
    for (tgt, h), g in res.groupby(["target_pct", "horizon_min"]):
        g = g[g["status"] != "no_data"].sort_values("bar_time")
        n = len(g)
        hits = int((g["status"] == "hit").sum())
        lo, hi = wilson_interval(hits, n)
        net = g["net_ret"].astype(float)
        eq = net.cumsum()
        dd = float((eq - eq.cummax()).min()) if n else float("nan")
        topk = float("nan")
        if "score" in g and g["score"].notna().any():
            tops = g.sort_values("score", ascending=False).groupby("session").head(k_top)
            topk = float((tops["status"] == "hit").mean()) if len(tops) else float("nan")
        rows.append({
            "target": f"+{tgt}% / {h}min", "signals": n, "hits": hits, "misses": int((g["status"] == "miss").sum()),
            "ambiguous": int((g["status"] == "ambiguous").sum()),
            "precision": hits / n if n else float("nan"), "ci95_low": lo, "ci95_high": hi,
            "false_signal_rate": 1 - hits / n if n else float("nan"),
            "avg_net_ret_%": float(net.mean()) if n else float("nan"), "median_net_ret_%": float(net.median()) if n else float("nan"),
            "p10_net_ret_%": float(net.quantile(0.1)) if n else float("nan"),
            "p90_net_ret_%": float(net.quantile(0.9)) if n else float("nan"),
            "sum_net_ret_%": float(net.sum()), "max_drawdown_%": dd,
            "median_minutes_to_target": float(g.loc[g["status"] == "hit", "minutes"].median()) if hits else float("nan"),
            f"precision@top{k_top}_per_session": topk,
            "avg_mfe_%": float(g["mfe"].mean()) if "mfe" in g and n else float("nan"),
            "avg_mae_%": float(g["mae"].mean()) if "mae" in g and n else float("nan"),
            "enough_samples": n >= rc.min_signals_for_stats,
        })
    return pd.DataFrame(rows)


def run_replay(bars_by_ticker: dict[str, pd.DataFrame], bench: pd.DataFrame | None, cfg: ScannerConfig | None = None,
               rc: ReplayConfig | None = None, dev_fraction: float = 0.6) -> dict:
    """Full study. Returns signals, per-trade outcomes, development / hold-out summaries and baselines."""
    cfg, rc = (cfg or ScannerConfig()).validate(), rc or ReplayConfig()
    rng = np.random.default_rng(rc.seed)
    feats, sigs, outs, base_mom, base_rand_pool = {}, [], [], [], {}
    skipped: dict[str, str] = {}
    for t, bars in bars_by_ticker.items():
        try:
            f = build_features(bars, bench, cfg, drop_incomplete=False)
        except Exception as exc:
            skipped[t] = str(exc)
            continue
        feats[t] = f
        s = generate_signals(f, cfg, rc, t)
        if not s.empty:
            sigs.append(s)
            outs.append(evaluate(s, f, cfg, rc))
        m = momentum_baseline(f, cfg, ticker=t)
        if not m.empty:
            base_mom.append(evaluate(m, f, cfg, rc))
    signals = pd.concat(sigs, ignore_index=True) if sigs else pd.DataFrame()
    outcomes = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
    mom = pd.concat(base_mom, ignore_index=True) if base_mom else pd.DataFrame()

    # random baseline: same number of signals, repeated trials, averaged
    rand_prec: dict[str, list[float]] = {}
    n_sig = len(signals)
    if n_sig and feats:
        for _ in range(rc.random_trials):
            per_ticker = max(1, int(round(n_sig / len(feats))))
            parts = [evaluate(random_baseline(f, cfg, per_ticker, rng, t), f, cfg, rc) for t, f in feats.items()]
            parts = [p for p in parts if not p.empty]
            if parts:
                s = summarize(pd.concat(parts, ignore_index=True), rc)
                for r in s.itertuples():
                    rand_prec.setdefault(r.target, []).append(r.precision)
    random_summary = pd.DataFrame([{"target": k, "random_precision_mean": float(np.nanmean(v)),
                                    "random_precision_p95": float(np.nanpercentile(v, 95)), "trials": len(v)}
                                   for k, v in rand_prec.items()])

    dev, hold = pd.DataFrame(), pd.DataFrame()
    split_info = {}
    if not outcomes.empty:
        sessions = sorted(outcomes["session"].unique())
        cut = max(1, int(len(sessions) * dev_fraction)) if len(sessions) > 1 else len(sessions)
        dev_s, hold_s = set(sessions[:cut]), set(sessions[cut:])
        # purge: nothing from the hold-out sessions is used for development and vice versa (whole sessions)
        dev = summarize(outcomes[outcomes["session"].isin(dev_s)], rc)
        hold = summarize(outcomes[outcomes["session"].isin(hold_s)], rc)
        split_info = {"development_sessions": len(dev_s), "holdout_sessions": len(hold_s)}
    notes = [
        "Costs are ASSUMED (slippage per side + commission), not measured; free bar data has no bid/ask.",
        "Same-bar target/stop collisions are 'ambiguous' and counted as non-wins.",
        "Universe built from currently listed tickers: survivorship bias (delisted names are missing).",
        "1-minute history from free providers is only a few days: treat results as a smoke test, not evidence.",
    ]
    if split_info.get("holdout_sessions", 0) == 0:
        notes.append("Not enough sessions for a separate hold-out; nothing here is out-of-sample.")
    return {"signals": signals, "outcomes": outcomes, "development": dev, "holdout": hold,
            "momentum_baseline": summarize(mom, rc) if not mom.empty else pd.DataFrame(),
            "random_baseline": random_summary, "split": split_info, "skipped": skipped, "notes": notes,
            "overall": summarize(outcomes, rc) if not outcomes.empty else pd.DataFrame()}
