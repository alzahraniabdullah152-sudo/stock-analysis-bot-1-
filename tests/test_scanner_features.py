from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from scanner.config import ScannerConfig
from scanner.features import FeatureError, build_features, data_age_seconds, prepare_bars
from tests.synth import make_minute_bars

CFG = ScannerConfig()
LATE = datetime(2026, 10, 20, 23, 0, tzinfo=timezone.utc)   # after the close: every synthetic bar is completed


def _bars(**k):
    return make_minute_bars(sessions=7, last_session_minutes=200, **k)


def test_features_have_no_lookahead_when_future_bars_are_appended():
    bars = _bars(surge_at=150, surge_pct=4.0, surge_vol_mult=6)
    full = build_features(bars, None, CFG, now=LATE)
    num = [c for c in full.columns if c not in ("session",)]
    for cut in (len(bars) - 120, len(bars) - 40):
        part = build_features(bars.iloc[:cut], None, CFG, now=LATE)
        a, b = full.loc[part.index, num], part[num]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


def test_rvol_is_nan_without_enough_prior_sessions_and_never_defaults_to_one():
    bars = make_minute_bars(sessions=3)
    f = build_features(bars, None, CFG, now=LATE)
    assert f["rvol"].isna().all()
    assert f["avg_dollar_vol"].isna().all()


def test_rvol_detects_a_volume_burst_but_cumulative_rvol_lags():
    bars = _bars(surge_at=150, surge_pct=4.0, surge_vol_mult=6)
    f = build_features(bars, None, CFG, now=LATE)
    today = f[f["session"] == f["session"].iloc[-1]]
    assert today["rvol"].iloc[160] > 4 and today["rvol"].iloc[100] < 1.5
    assert today["rvol_day"].iloc[160] < today["rvol"].iloc[160]


def test_breakout_level_excludes_current_bar_and_vwap_is_session_scoped():
    f = build_features(_bars(), None, CFG, now=LATE)
    first_of_day = f.groupby("session").head(1)
    assert first_of_day["level"].isna().all()                # no 15-minute lookback at the open
    assert (f["vwap"].groupby(f["session"]).first() - f.groupby("session")["Close"].first()).abs().max() < 1.0
    assert (f["breakout"] == (f["Close"] > f["level"])).all()


def test_incomplete_last_bar_is_dropped_by_default():
    bars = _bars()
    last_start = bars.index[-1].tz_convert("UTC").to_pydatetime()
    now_inside_last_minute = last_start.replace(second=30)
    f = build_features(bars, None, CFG, now=now_inside_last_minute)
    assert f.index[-1] == bars.index[-2]
    assert data_age_seconds(f, now_inside_last_minute) >= 0


def test_bad_rows_are_rejected_not_repaired():
    bars = _bars().copy()
    bars.iloc[10, bars.columns.get_loc("Close")] = np.nan
    bars.iloc[20, bars.columns.get_loc("High")] = -1.0
    f = build_features(bars, None, CFG, now=LATE)
    assert bars.index[10] not in f.index and bars.index[20] not in f.index


def test_extended_hours_and_holidays_are_excluded():
    idx = pd.date_range("2026-12-25 09:30", periods=30, freq="min", tz="America/New_York")   # Christmas: closed
    df = pd.DataFrame({"Open": 10.0, "High": 10.1, "Low": 9.9, "Close": 10.0, "Volume": 1000.0}, index=idx)
    with pytest.raises(FeatureError):
        prepare_bars(df, now=datetime(2026, 12, 26, tzinfo=timezone.utc))
    pre = pd.date_range("2026-10-20 07:00", periods=30, freq="min", tz="America/New_York")
    df2 = pd.DataFrame({"Open": 10.0, "High": 10.1, "Low": 9.9, "Close": 10.0, "Volume": 1000.0}, index=pre)
    with pytest.raises(FeatureError):
        prepare_bars(df2, now=LATE)


def test_empty_and_missing_columns_raise():
    with pytest.raises(FeatureError):
        prepare_bars(pd.DataFrame())
    with pytest.raises(FeatureError):
        prepare_bars(pd.DataFrame({"Close": [1.0]}, index=pd.date_range("2026-10-20 10:00", periods=1, freq="min", tz="America/New_York")))
