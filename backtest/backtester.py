"""Event-driven bar-by-bar backtester.

Execution assumptions (conservative, no look-ahead):
* A signal on completed bar i-1 is acted on at the OPEN of bar i (plus adverse slippage).
* Stop/target levels come from bar i-1 data only.
* If one bar's range touches BOTH stop and target, the STOP is assumed to hit first.
* Gaps through the stop fill at the open (worse than stop); gaps through the target fill at the open.
* Stops and time exits suffer slippage; limit-target fills do not.
* Commission is charged per share on both sides.
Backtested results do not guarantee future performance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.risk_management import RiskInputError, position_size, risk_reward
from analysis.signal_engine import build_setup, compute_signals, confirmed_levels
from config import SETTINGS, Settings


@dataclass
class BacktestConfig:
    equity: float = 10_000.0
    risk_pct: float = 1.0
    commission_per_share: float = SETTINGS.commission_per_share
    slippage_pct: float = SETTINGS.slippage_pct
    max_hold_bars: int = 30
    allow_short: bool = True
    max_position_pct: float = 100.0
    min_rr: float = SETTINGS.min_rr
    halt_drawdown_pct: float | None = 25.0  # stop opening trades beyond this drawdown
    setups: tuple[str, ...] | None = None


def _metrics(trades: pd.DataFrame, curve: pd.Series, exposure_bars: int, n_bars: int, start_eq: float) -> dict:
    out: dict = {"trades": int(len(trades))}
    final = float(curve.iloc[-1]) if len(curve) else start_eq
    out["net_return_pct"] = (final / start_eq - 1) * 100
    dd = (curve / curve.cummax() - 1).min() * 100 if len(curve) else 0.0
    out["max_drawdown_pct"] = float(dd)
    out["exposure_pct"] = exposure_bars / n_bars * 100 if n_bars else 0.0
    if trades.empty:
        out.update(win_rate_pct=None, profit_factor=None, avg_win=None, avg_loss=None, expectancy=None, expectancy_r=None)
        return out
    wins, losses = trades[trades.pnl > 0], trades[trades.pnl <= 0]
    gp, gl = wins.pnl.sum(), -losses.pnl.sum()
    out["win_rate_pct"] = len(wins) / len(trades) * 100
    out["profit_factor"] = float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else None)
    out["avg_win"] = float(wins.pnl.mean()) if len(wins) else 0.0
    out["avg_loss"] = float(losses.pnl.mean()) if len(losses) else 0.0
    out["expectancy"] = float(trades.pnl.mean())
    out["expectancy_r"] = float(trades.r_multiple.mean())
    return out


def run_backtest(d: pd.DataFrame, sigs: pd.DataFrame | None = None, bt: BacktestConfig | None = None,
                 cfg: Settings = SETTINGS, intraday: bool = False, ticker: str = "") -> dict:
    """Backtest the rule-based setups on indicator-enriched data `d` (from add_all)."""
    bt = bt or BacktestConfig()
    sigs = compute_signals(d, intraday, cfg) if sigs is None else sigs
    names = [n for n in sigs.columns if bt.setups is None or n in bt.setups]
    ph_c, pl_c = confirmed_levels(d, cfg.sr_pivot_window)
    o, h, l, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    n = len(d)
    equity = bt.equity
    peak = equity
    curve = np.full(n, np.nan)
    trades: list[dict] = []
    pos: dict | None = None
    exposure = 0
    slip = bt.slippage_pct / 100

    for i in range(1, n):
        # ---- manage open position on bar i
        if pos is not None:
            exposure += 1
            long = pos["dir"] > 0
            stop, tgt = pos["stop"], pos["target"]
            exit_px, reason = None, None
            if long:
                if o[i] <= stop: exit_px, reason = o[i], "stop_gap"
                elif l[i] <= stop: exit_px, reason = stop, "stop"          # stop wins ties
                elif o[i] >= tgt: exit_px, reason = o[i], "target_gap"
                elif h[i] >= tgt: exit_px, reason = tgt, "target"
            else:
                if o[i] >= stop: exit_px, reason = o[i], "stop_gap"
                elif h[i] >= stop: exit_px, reason = stop, "stop"
                elif o[i] <= tgt: exit_px, reason = o[i], "target_gap"
                elif l[i] <= tgt: exit_px, reason = tgt, "target"
            if exit_px is None and i - pos["entry_i"] >= bt.max_hold_bars:
                exit_px, reason = c[i], "time"
            if exit_px is not None:
                if reason.startswith("stop") or reason == "time":
                    exit_px = exit_px * (1 - slip) if long else exit_px * (1 + slip)
                gross = (exit_px - pos["fill"]) * pos["dir"] * pos["shares"]
                fees = 2 * bt.commission_per_share * pos["shares"]
                pnl = gross - fees
                equity += pnl
                trades.append({
                    "setup": pos["setup"], "side": "long" if long else "short",
                    "entry_time": d.index[pos["entry_i"]], "exit_time": d.index[i],
                    "entry": pos["fill"], "exit": exit_px, "shares": pos["shares"], "pnl": pnl,
                    "r_multiple": pnl / pos["risk"] if pos["risk"] > 0 else 0.0, "reason": reason,
                })
                pos = None
            else:
                mtm = (c[i] - pos["fill"]) * pos["dir"] * pos["shares"]
                curve[i] = equity + mtm
        if pos is None:
            curve[i] = equity
            peak = max(peak, equity)
            halted = bt.halt_drawdown_pct is not None and equity < peak * (1 - bt.halt_drawdown_pct / 100)
            if halted or i >= n - 1:
                continue
            # ---- look for a signal on the COMPLETED bar i (acted on at open of i+1)
            # (evaluated at end of loop body so entry uses bar i+1 data only)
            cands = []
            for nm in names:
                v = int(sigs[nm].iloc[i])
                if v != 0 and (v > 0 or bt.allow_short):
                    cands.append((nm, v))
            if not cands or len({v for _, v in cands}) > 1:
                continue
            nm, v = cands[0]
            s = build_setup(d, sigs, nm, i, ticker, "", cfg, None, True, pl_c, ph_c)
            if s is None:
                continue
            nxt_open = o[i + 1]
            stop, tgt = s.stop, s.targets[0]
            rr = risk_reward(nxt_open, stop, tgt)
            if rr is None or rr < bt.min_rr:
                continue
            try:
                plan = position_size(equity, bt.risk_pct, nxt_open, stop, [tgt], bt.commission_per_share,
                                     bt.slippage_pct, False, bt.max_position_pct)
            except RiskInputError:
                continue
            if plan.shares < 1:
                continue
            pos = {"setup": nm, "dir": v, "entry_i": i + 1, "fill": plan.entry, "stop": stop, "target": tgt,
                   "shares": plan.shares, "risk": plan.dollar_risk}
            # The entry bar (i+1) is managed at the top of the next iteration.
    if pos is not None:  # close at last close
        long = pos["dir"] > 0
        px = c[-1] * (1 - slip) if long else c[-1] * (1 + slip)
        pnl = (px - pos["fill"]) * pos["dir"] * pos["shares"] - 2 * bt.commission_per_share * pos["shares"]
        equity += pnl
        trades.append({"setup": pos["setup"], "side": "long" if long else "short",
                       "entry_time": d.index[pos["entry_i"]], "exit_time": d.index[-1], "entry": pos["fill"],
                       "exit": px, "shares": pos["shares"], "pnl": pnl,
                       "r_multiple": pnl / pos["risk"] if pos["risk"] > 0 else 0.0, "reason": "end_of_data"})
        curve[-1] = equity
    cs = pd.Series(curve, index=d.index).ffill().fillna(bt.equity)
    tdf = pd.DataFrame(trades)
    return {"metrics": _metrics(tdf, cs, exposure, n, bt.equity), "trades": tdf, "equity_curve": cs}


def run_in_out_of_sample(d: pd.DataFrame, split: float = 0.7, bt: BacktestConfig | None = None,
                         cfg: Settings = SETTINGS, intraday: bool = False, ticker: str = "") -> dict:
    """Chronological split. Develop/tune on in-sample only; judge on out-of-sample.

    Indicators/signals are causal, so slicing the enriched frame does not leak future data.
    """
    if not 0.3 <= split <= 0.9:
        raise ValueError("split must be between 0.3 and 0.9")
    sigs = compute_signals(d, intraday, cfg)
    k = int(len(d) * split)
    res = {}
    for label, sl in (("in_sample", slice(0, k)), ("out_of_sample", slice(k, None))):
        res[label] = run_backtest(d.iloc[sl], sigs.iloc[sl], bt, cfg, intraday, ticker)
    return res
