import numpy as np
import pandas as pd
import pytest

from scanner.backtest import ReplayConfig, _arrays, run_replay, summarize, trade_outcome, wilson_interval
from scanner.calibration import calibration_report
from scanner.config import ScannerConfig
from scanner.features import build_features
from tests.synth import make_minute_bars

RC = ReplayConfig(slippage_pct=0.0, commission_per_share=0.0)


def _frame(opens, highs, lows, closes, start="2026-10-20 10:00"):
    idx = pd.date_range(start, periods=len(opens), freq="min", tz="America/New_York")
    f = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 1.0}, index=idx)
    f["session"] = [x.date() for x in idx]
    return f


def test_entry_is_next_open_and_target_hit_is_not_the_signal_bar():
    f = _frame([10, 10, 10, 10], [10.1, 10.1, 10.3, 10.1], [9.9, 9.95, 9.95, 9.95], [10, 10, 10, 10])
    r = trade_outcome(_arrays(f), 0, 2.0, 15, 1.5, RC)      # signal on bar 0, entry open of bar 1 = 10
    assert r["status"] == "hit" and r["minutes"] == 1.0 and abs(r["net_ret"] - 2.0) < 1e-9


def test_same_bar_target_and_stop_is_ambiguous_not_a_win():
    f = _frame([10, 10, 10], [10, 10.1, 10.5], [10, 9.95, 9.5], [10, 10, 10])
    r = trade_outcome(_arrays(f), 0, 2.0, 15, 1.5, RC)
    assert r["status"] == "ambiguous" and r["net_ret"] < 0


def test_gap_through_stop_fills_at_the_worse_open_and_costs_are_charged():
    f = _frame([10, 10, 9.6, 9.6], [10, 10.05, 9.7, 9.7], [10, 9.95, 9.5, 9.5], [10, 10, 9.6, 9.6])
    r = trade_outcome(_arrays(f), 0, 2.0, 15, 1.5, RC)
    assert r["status"] == "miss" and r["reason"] == "stop" and r["net_ret"] < -3.9
    paid = trade_outcome(_arrays(f), 0, 2.0, 15, 1.5, ReplayConfig(slippage_pct=0.1))
    assert paid["net_ret"] < r["net_ret"] - 0.19


def test_horizon_uses_minutes_not_raw_integer_timestamps():
    # regression: with pandas' microsecond index resolution a nanosecond-based window made +2% in 15min look easy
    n = 60
    highs = [10.05] * n
    highs[40] = 10.5                                            # +5% only 40 minutes after entry
    f = _frame([10.0] * n, highs, [9.99] * n, [10.0] * n)
    r = trade_outcome(_arrays(f), 0, 2.0, 15, 1.5, RC)
    assert r["status"] == "miss" and r["reason"] == "timeout"


def test_session_end_before_horizon_is_excluded_from_rates():
    f = _frame([10.0] * 5, [10.05] * 5, [9.99] * 5, [10.0] * 5, start="2026-10-20 15:55")
    assert trade_outcome(_arrays(f), 0, 2.0, 15, 1.5, RC)["status"] == "no_data"


def test_wilson_interval_widens_for_small_samples():
    lo1, hi1 = wilson_interval(2, 2)
    lo2, hi2 = wilson_interval(100, 100)
    assert lo1 < 0.5 and hi1 == 1.0 and lo2 > 0.95
    assert np.isnan(wilson_interval(0, 0)[0])


def _study():
    surges = [(si, 60 + (si * 37) % 250, 3.0 + (si % 3), 5) for si in range(4, 14)]
    data = {"AAA": make_minute_bars(sessions=14, seed=11, surges=surges),
            "BBB": make_minute_bars(sessions=14, seed=12, price=20, surges=[(si, 100, 2.5, 4) for si in (6, 9, 12)])}
    bench = make_minute_bars(sessions=14, seed=9, price=400)
    return run_replay(data, bench, ScannerConfig(), ReplayConfig(random_trials=40))


def test_replay_runs_splits_sessions_and_beats_random_on_planted_moves():
    res = _study()
    assert len(res["signals"]) >= 5
    assert res["split"]["development_sessions"] >= 1 and res["split"]["holdout_sessions"] >= 1
    dev_sessions = set(res["outcomes"]["session"].unique())
    ov = res["overall"].set_index("target")
    rnd = res["random_baseline"].set_index("target")
    assert ov.loc["+2.0% / 15min", "precision"] > rnd.loc["+2.0% / 15min", "random_precision_mean"] + 0.1
    assert all(res["overall"]["enough_samples"] == False)       # tiny sample must be flagged, never claimed as evidence
    assert any("survivorship" in n.lower() for n in res["notes"]) and any("ASSUMED" in n for n in res["notes"])


def test_replay_signals_use_only_past_data():
    bars = make_minute_bars(sessions=10, seed=11, surges=[(6, 120, 4.0, 6), (8, 200, 4.0, 6)])
    cfg = ScannerConfig()
    full = run_replay({"AAA": bars}, None, cfg, ReplayConfig(random_trials=5))["signals"]
    cut = bars.index[-1500]
    part = run_replay({"AAA": bars[bars.index <= cut]}, None, cfg, ReplayConfig(random_trials=5))["signals"]
    if not part.empty:
        merged = set(part["bar_time"]) - set(full["bar_time"])
        assert merged == set()                                   # truncating the future never creates new past signals


def test_calibration_refuses_small_samples_and_unskilled_models():
    rows = [{"target_pct": 2.0, "horizon_min": 15, "status": "hit" if i % 3 == 0 else "miss", "score": 50 + (i % 40),
             "session": pd.Timestamp("2026-01-01").date() + pd.Timedelta(days=i // 20)} for i in range(150)]
    small = calibration_report(pd.DataFrame(rows), 2.0, 15)
    assert not small["ok"] and "at least" in small["reason"]
    rng = np.random.default_rng(0)
    big = [{"target_pct": 2.0, "horizon_min": 15, "status": "hit" if rng.random() < 0.3 else "miss",
            "score": float(rng.uniform(40, 100)), "session": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=i // 25)).date()}
           for i in range(600)]
    noise = calibration_report(pd.DataFrame(big), 2.0, 15)          # score carries no information
    assert not noise["ok"]


def test_calibration_accepts_a_genuinely_informative_score():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(800):
        sc = float(rng.uniform(40, 100))
        p = 0.05 + 0.6 * (sc - 40) / 60
        rows.append({"target_pct": 2.0, "horizon_min": 15, "status": "hit" if rng.random() < p else "miss", "score": sc,
                     "session": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=i // 20)).date()})
    rep = calibration_report(pd.DataFrame(rows), 2.0, 15)
    assert rep["ok"] and rep["brier_model"] < rep["brier_base_rate"] and len(rep["table"]) >= 3
