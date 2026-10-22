import numpy as np
import pandas as pd
import pytest

from analysis.signal_engine import compute_signals, confluence_score, latest_setups
from analysis.trend_analysis import TrendState
from backtest.backtester import BacktestConfig, run_backtest, run_in_out_of_sample
from config import SETTINGS
from indicators.technical_indicators import add_all


def _frame(n=400, seed=7, drift=0.05):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(drift, 1.2, n))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1, n)
    vol = rng.integers(1_000, 9_000, n).astype(float)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_signals_do_not_repaint_when_future_bars_are_added():
    d = add_all(_frame())
    full = compute_signals(d)
    for cut in (250, 300, 350):
        part = compute_signals(add_all(_frame().iloc[:cut]))
        pd.testing.assert_frame_equal(full.iloc[:cut], part)  # history identical => no look-ahead


def test_signal_values_are_ternary():
    s = compute_signals(add_all(_frame()))
    assert set(np.unique(s.to_numpy())) <= {-1, 0, 1}


def test_ema_cross_requires_trend_and_adx():
    d = add_all(_frame())
    d = d.copy()
    d["adx"] = 0.0  # force ADX below the configured minimum
    s = compute_signals(d)
    assert (s["ema_cross"] == 0).all()


def test_breakout_rule_on_synthetic_bar():
    f = _frame(120, drift=0.0)
    f.iloc[-1, f.columns.get_loc("Open")] = f.Close.iloc[-2]
    f.iloc[-1, f.columns.get_loc("Close")] = f.High.iloc[-21:-1].max() + 3
    f.iloc[-1, f.columns.get_loc("High")] = f.Close.iloc[-1] + 0.5
    f.iloc[-1, f.columns.get_loc("Volume")] = 100_000.0
    d = add_all(f)
    assert compute_signals(d)["breakout"].iloc[-1] == 1


def test_setup_levels_are_consistent():
    d = add_all(_frame())
    sigs = compute_signals(d)
    found = 0
    for i in np.flatnonzero((sigs != 0).any(axis=1).to_numpy()):
        sub = d.iloc[: i + 1]
        for s in latest_setups(sub, "TEST", "1d", False):
            found += 1
            if s.bias == "bullish":
                assert s.stop < s.entry < min(s.targets)
            else:
                assert s.stop > s.entry > max(s.targets)
            assert s.rr is None or s.rr > 0
            assert s.conditions and "completed" in s.freshness
    assert found > 0


def test_confluence_score_formula_and_missing_data():
    row = pd.Series({"sma50": 1.0, "rsi14": 60.0, "rel_vol": 1.5, "Close": 11.0, "Open": 10.0})
    t = TrendState("bullish", "HH/HL", "bullish", True, 30.0)
    full = confluence_score(t, row, 1.0)
    assert full["score"] == 100.0
    no_mtf = confluence_score(t, row, None)
    assert no_mtf["score"] == 100.0  # renormalised over available components
    neutral = confluence_score(TrendState("neutral", "mixed", "neutral", None, None), row.drop("rel_vol"), 0.0)
    assert neutral["score"] == 50.0
    assert "Not a probability" in full["note"]
    assert sum(SETTINGS.score_weights.values()) == 100


def test_backtest_stop_wins_when_bar_hits_both(monkeypatch):
    # signal on bar 1 -> entry at open of bar 2 (100), stop 95, target 110; bar 2 range covers both -> stop first
    idx = pd.bdate_range("2024-01-02", periods=4)
    d = pd.DataFrame({"Open": [100, 100, 100, 100.0], "High": [101, 101, 112, 101], "Low": [99, 99, 90, 99],
                      "Close": [100, 100, 100, 100.0], "Volume": 1.0}, index=idx)
    d["atr14"] = 2.0
    sigs = pd.DataFrame({"x": [0, 1, 0, 0]}, index=idx)
    import backtest.backtester as bt_mod
    from analysis.signal_engine import Setup

    def fake_build(*a, **k):
        return Setup("T", "", "x", "x", "bullish", ["c"], 100, 99, 95.0, [110.0], 2.0, [], [], idx[0], "f", 0)

    monkeypatch.setattr(bt_mod, "build_setup", fake_build)
    cfg = BacktestConfig(slippage_pct=0, commission_per_share=0, min_rr=1.0)
    res = run_backtest(d, sigs, cfg)
    t = res["trades"].iloc[0]
    assert t.reason == "stop" and t.pnl < 0


def test_backtest_metrics_and_oos_split_run():
    d = add_all(_frame(600))
    res = run_in_out_of_sample(d, 0.7, BacktestConfig())
    for part in ("in_sample", "out_of_sample"):
        m = res[part]["metrics"]
        assert m["trades"] >= 0 and m["max_drawdown_pct"] <= 0
        assert 0 <= m["exposure_pct"] <= 100
    assert res["in_sample"]["equity_curve"].index[-1] < res["out_of_sample"]["equity_curve"].index[0]
    with pytest.raises(ValueError):
        run_in_out_of_sample(d, 0.99)
