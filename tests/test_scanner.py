import logging
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from data.market_data import DataError
from scanner.backtest import (ReplayConfig, generate_signals, run_replay, trade_outcome, _arrays, wilson_interval)
from scanner.calibration import calibration_report
from scanner.config import ScannerConfig
from scanner.engine import Scanner, evaluate_ticker, prefilter
from scanner.features import FeatureError, build_features, prepare_bars
from scanner.notify import format_alert
from scanner.providers import FrameProvider, split_download
from scanner.scoring import score_row
from scanner.signal_log import SignalLog, outcome_for
from scanner.states import classify
from tests.synth import make_minute_bars

NOW = datetime(2026, 10, 20, 20, 0, tzinfo=timezone.utc)   # after the close: every bar is complete


def _surge_bars(minutes=156, **kw):
    return make_minute_bars(sessions=6, last_session_minutes=minutes, surge_at=145, surge_pct=4.0, surge_vol_mult=6, **kw)


# ----------------------------------------------------------------------------- no look-ahead
def test_features_do_not_change_when_future_bars_are_added():
    bars = _surge_bars(200)
    full = build_features(bars, None, ScannerConfig(), now=NOW)
    cut = full.index[-60]
    part = build_features(bars[bars.index <= cut], None, ScannerConfig(), now=NOW)
    a = full.loc[part.index].drop(columns=["session"])
    b = part.drop(columns=["session"])
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


def test_replay_signals_do_not_change_when_future_bars_are_added():
    cfg, rc = ScannerConfig(), ReplayConfig()
    bars = make_minute_bars(sessions=10, seed=11, surges=[(si, 80 + si * 20, 4.0, 6) for si in range(4, 10)])
    full = build_features(bars, None, cfg, drop_incomplete=False)
    k = int(len(full) * 0.8)
    part = build_features(bars.iloc[:k], None, cfg, drop_incomplete=False)
    s_full, s_part = generate_signals(full, cfg, rc, "T"), generate_signals(part, cfg, rc, "T")
    early_full = s_full[s_full["i"] < k - 1][["i", "state", "score"]].reset_index(drop=True) if not s_full.empty else s_full
    early_part = s_part[s_part["i"] < k - 1][["i", "state", "score"]].reset_index(drop=True) if not s_part.empty else s_part
    assert len(early_full) > 0                      # the comparison must not be vacuous
    pd.testing.assert_frame_equal(early_full, early_part)


def test_rvol_compares_the_same_time_window_of_earlier_sessions():
    f = build_features(_surge_bars(156), None, ScannerConfig(), now=NOW)
    today = f[f.session == f.session.iloc[-1]]
    assert today["rvol"].iloc[10:140].median() == pytest.approx(1.0, abs=0.35)   # quiet period ~ normal
    assert today["rvol"].iloc[150] > 2.0                                          # burst after the surge starts


def test_incomplete_last_bar_is_dropped():
    bars = _surge_bars(100)
    last = bars.index[-1]
    now = (last + pd.Timedelta(seconds=30)).tz_convert("UTC").to_pydatetime()
    assert prepare_bars(bars, now).index[-1] == bars.index[-2]


def test_bad_input_raises_clear_errors():
    with pytest.raises(FeatureError):
        prepare_bars(pd.DataFrame())
    with pytest.raises(FeatureError):
        prepare_bars(pd.DataFrame({"Close": [1.0]}))


# ----------------------------------------------------------------------------- scoring / states
def _row_at(f, i):
    return f[f.session == f.session.iloc[-1]].iloc[i]


def test_score_is_bounded_and_missing_data_never_raises_it():
    f = build_features(_surge_bars(156), make_minute_bars(seed=9, price=400, sessions=6, last_session_minutes=156), ScannerConfig(), now=NOW)
    cfg = ScannerConfig()
    for i in (5, 60, 150, 155):
        row = _row_at(f, i)
        sc = score_row(row, cfg)
        assert 0 <= sc.score <= 100 and 0 <= sc.coverage <= 1
        worse = row.copy()
        worse["rvol"], worse["rs15"] = np.nan, np.nan
        sc2 = score_row(worse, cfg)
        assert sc2.score <= sc.score and sc2.coverage < sc.coverage


def test_all_missing_features_give_zero_score_and_no_signal():
    cfg = ScannerConfig()
    row = pd.Series({"Close": 10.0})
    sc = score_row(row, cfg)
    assert sc.score == 0 and sc.coverage == 0
    cl = classify(pd.Series({"Close": 10.0, "level": np.nan}), sc, cfg, market_open=True, data_age_s=0)
    assert cl.state == "none" and not cl.actionable and cl.blocked == "insufficient_data"


def test_score_weights_are_editable_and_validated():
    with pytest.raises(ValueError):
        ScannerConfig(weights={"rvol": -1}).validate()
    with pytest.raises(ValueError):
        ScannerConfig(weights={"bogus": 10}).validate()
    assert sum(ScannerConfig().weights.values()) == 100


def test_state_lifecycle_on_a_surge():
    cfg = ScannerConfig()
    f = build_features(_surge_bars(156), make_minute_bars(seed=9, price=400, sessions=6, last_session_minutes=156), cfg, now=NOW)
    seen = set()
    for i in range(140, 156):
        row = _row_at(f, i)
        seen.add(classify(row, score_row(row, cfg), cfg, market_open=True, data_age_s=10).state)
    assert "confirmed" in seen or "early" in seen
    assert classify(_row_at(f, 20), score_row(_row_at(f, 20), cfg), cfg, market_open=True, data_age_s=10).state in ("none", "watch")


def test_closed_market_and_stale_data_are_never_actionable():
    cfg = ScannerConfig()
    f = build_features(_surge_bars(156), None, cfg, now=NOW)
    row = _row_at(f, 152)
    sc = score_row(row, cfg)
    assert not classify(row, sc, cfg, market_open=False, data_age_s=5).actionable
    assert classify(row, sc, cfg, market_open=False, data_age_s=5).blocked == "market_closed"
    assert classify(row, sc, cfg, market_open=True, data_age_s=cfg.stale_after_seconds + 1).blocked == "data_stale"


def test_prefilter_reasons():
    cfg = ScannerConfig()
    assert "price" in prefilter(pd.Series({"Close": 0.5, "avg_dollar_vol": 1e9}), cfg)
    assert "liquidity unknown" in prefilter(pd.Series({"Close": 10.0, "avg_dollar_vol": np.nan}), cfg)
    assert "below" in prefilter(pd.Series({"Close": 10.0, "avg_dollar_vol": 1e3}), cfg)
    assert prefilter(pd.Series({"Close": 10.0, "avg_dollar_vol": 1e9, "spread_pct": 2.0}), cfg).startswith("spread")
    assert prefilter(pd.Series({"Close": 10.0, "avg_dollar_vol": 1e9}), cfg) is None


# ----------------------------------------------------------------------------- journal / alerts
class _Sink:
    enabled = True

    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)
        return True


def _scanner(sink=None, db=None):
    prov = FrameProvider({"ABCD": _surge_bars(156), "SPY": make_minute_bars(seed=9, price=400, sessions=6, last_session_minutes=156)})
    j = SignalLog(db or os.path.join(tempfile.mkdtemp(), "s.db"))
    return Scanner(prov, ScannerConfig(), j, sink, names={"ABCD": "Abcd Corp"}), j


def test_alert_is_sent_once_and_not_repeated():
    sink = _Sink()
    sc, j = _scanner(sink)
    now = datetime(2026, 10, 20, 16, 7, tzinfo=timezone.utc)
    sc.scan(["ABCD"], now=now, send_alerts=True)
    sc.scan(["ABCD"], now=now, send_alerts=True)
    sc.scan(["ABCD"], now=datetime(2026, 10, 20, 16, 8, tzinfo=timezone.utc), send_alerts=True)
    assert len(sink.sent) == 1                      # exactly one alert for the new event, none for repeats
    assert len(j.signals()) >= 1 and j.signals()["alerted"].sum() == 1


def test_alert_text_has_required_fields_and_disclaimer():
    sink = _Sink()
    sc, j = _scanner(sink)
    rep = sc.scan(["ABCD"], now=datetime(2026, 10, 20, 16, 7, tzinfo=timezone.utc), send_alerts=True)
    c = rep.candidates[0]
    text = format_alert(c, datetime(2026, 10, 20, 16, 7, tzinfo=timezone.utc))
    for needle in ("ABCD", "ET", "/100", "الحجم النسبي", "الأسباب", "المصدر", "ليست نصيحة", "وليست احتمال", "المحفز"):
        assert needle in text
    assert len(text) <= 4000


def test_missing_provider_data_is_reported_not_turned_into_a_signal():
    sc, j = _scanner()
    rep = sc.scan(["ABCD", "NOPE"], now=datetime(2026, 10, 20, 16, 7, tzinfo=timezone.utc))
    assert "NOPE" in rep.skipped and all(c.ticker != "NOPE" for c in rep.candidates)


def test_split_download_handles_multiindex_and_failures():
    idx = pd.date_range("2026-10-19 09:30", periods=3, freq="min")
    cols = pd.MultiIndex.from_product([["AAA", "BBB"], ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(np.ones((3, 10)), index=idx, columns=cols)
    df[("BBB", "Close")] = np.nan
    df.loc[:, [("BBB", c) for c in ["Open", "High", "Low", "Volume"]]] = np.nan
    out = split_download(df, ["AAA", "BBB", "CCC"])
    assert isinstance(out["AAA"], pd.DataFrame) and isinstance(out["BBB"], DataError) and isinstance(out["CCC"], DataError)
    assert all(isinstance(v, DataError) for v in split_download(pd.DataFrame(), ["X"]).values())


def test_telegram_failure_never_logs_the_token():
    import requests
    from alerts.alert_manager import TelegramNotifier
    records = []

    class H(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())
    h = H()
    logging.getLogger("alerts.alert_manager").addHandler(h)
    orig = requests.post

    def boom(*a, **k):
        raise requests.ConnectionError("https://api.telegram.org/botSECRET-TOKEN-123/sendMessage failed")
    requests.post = boom
    try:
        assert TelegramNotifier("SECRET-TOKEN-123", "1").send("x") is False
    finally:
        requests.post = orig
        logging.getLogger("alerts.alert_manager").removeHandler(h)
    assert records and not any("SECRET-TOKEN-123" in r for r in records)


# ----------------------------------------------------------------------------- outcomes / replay
def _bars(opens, highs, lows, closes):
    idx = pd.date_range("2026-10-20 09:31", periods=len(opens), freq="min", tz="America/New_York")
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 1.0}, index=idx)


def test_same_minute_target_and_stop_is_ambiguous_not_a_win():
    b = _bars([10, 10, 10], [10.1, 10.5, 10.1], [9.9, 9.7, 9.9], [10, 10, 10])
    r = outcome_for(b, b.index[0].to_pydatetime(), 10.0, 3.0, 5, 1.5, datetime(2026, 10, 20, 21, 0, tzinfo=timezone.utc))
    assert r["status"] == "ambiguous"


def test_outcome_pending_until_the_horizon_is_complete():
    b = _bars([10, 10.1], [10.2, 10.2], [9.95, 10.0], [10.1, 10.1])
    r = outcome_for(b, b.index[0].to_pydatetime(), 10.0, 3.0, 15, 1.5, datetime(2026, 10, 20, 13, 33, tzinfo=timezone.utc))
    assert r["status"] == "pending"


def test_replay_trade_costs_fill_next_open_and_gap_through_stop():
    cfg = ScannerConfig()
    ts = pd.date_range("2026-10-20 09:31", periods=6, freq="min", tz="America/New_York")
    f = pd.DataFrame({"Open": [10, 10, 9.0, 9, 9, 9], "High": [10, 10.05, 9.1, 9, 9, 9], "Low": [10, 9.95, 8.9, 9, 9, 9],
                      "Close": [10, 10, 9, 9, 9, 9], "session": pd.Timestamp("2026-10-20").date()}, index=ts)
    arr = _arrays(f)
    rc = ReplayConfig(slippage_pct=0.1)
    o = trade_outcome(arr, 0, 2.0, 5, 1.5, rc)        # entry at open of bar 1 (10 * 1.001); bar 2 gaps to 9.0
    assert o["status"] == "miss" and o["reason"] == "stop"
    assert o["net_ret"] < -1.5 - 0.2                  # gap fill is worse than the stop, plus costs


def test_wilson_interval_and_small_sample_flags():
    lo, hi = wilson_interval(2, 2)
    assert 0.3 < lo < 0.4 and hi == 1.0
    res = run_replay({"AAA": make_minute_bars(sessions=8, seed=2)}, None, ScannerConfig(), ReplayConfig(random_trials=3))
    assert {"signals", "overall", "holdout", "random_baseline", "notes"} <= set(res)
    assert any("survivorship" in n.lower() for n in res["notes"])
    assert any("same-bar" in n.lower() for n in res["notes"])


def test_calibration_refuses_small_samples_and_weak_models():
    rows = [{"target_pct": 2.0, "horizon_min": 15, "status": "hit" if i % 3 == 0 else "miss", "score": 50 + (i % 40),
             "session": pd.Timestamp("2026-01-01").date() + pd.Timedelta(days=i // 10)} for i in range(60)]
    r = calibration_report(pd.DataFrame(rows), 2.0, 15)
    assert not r["ok"] and "at least" in r["reason"]
    rng = np.random.default_rng(0)
    big = [{"target_pct": 2.0, "horizon_min": 15, "status": "hit" if rng.random() < 0.3 else "miss", "score": float(rng.uniform(40, 95)),
            "session": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=i // 25)).date()} for i in range(1500)]
    r2 = calibration_report(pd.DataFrame(big), 2.0, 15)
    assert not r2["ok"]      # score carries no information here -> must not claim a probability
