import numpy as np
import pandas as pd
import pytest

from indicators import technical_indicators as ti


def _ohlc(n=300, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1, n)
    vol = rng.integers(1_000, 5_000, n).astype(float)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_sma_known_values_and_warmup():
    s = pd.Series([1.0, 2, 3, 4, 5])
    out = ti.sma(s, 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_ema_matches_recursive_definition():
    s = pd.Series([10.0, 11, 12, 11, 10, 9, 10, 12])
    n = 3
    a = 2 / (n + 1)
    exp = [s.iloc[0]]
    for x in s.iloc[1:]:
        exp.append(a * x + (1 - a) * exp[-1])
    out = ti.ema(s, n)
    np.testing.assert_allclose(out.iloc[n - 1:].to_numpy(), np.array(exp)[n - 1:], rtol=1e-9)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1.0, 60))
    assert ti.rsi(up).dropna().iloc[-1] == pytest.approx(100)
    down = pd.Series(np.arange(60.0, 1, -1))
    assert ti.rsi(down).dropna().iloc[-1] == pytest.approx(0, abs=1e-9)
    r = ti.rsi(_ohlc().Close).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_macd_hist_identity():
    c = _ohlc().Close
    m = ti.macd(c)
    assert (m.macd - m.macd_signal - m.macd_hist).dropna().abs().max() < 1e-9


def test_bollinger_ordering():
    b = ti.bollinger(_ohlc().Close).dropna()
    assert (b.bb_upper >= b.bb_mid).all() and (b.bb_mid >= b.bb_lower).all()


def test_atr_positive_and_warmup():
    d = _ohlc()
    a = ti.atr(d)
    assert a.iloc[:13].isna().all()
    assert (a.dropna() > 0).all()


def test_adx_ranges():
    x = ti.adx(_ohlc()).dropna()
    assert ((x.adx >= 0) & (x.adx <= 100)).all()
    assert ((x.plus_di >= 0) & (x.minus_di >= 0)).all()


def test_relative_volume_excludes_current_bar():
    v = pd.Series([100.0] * 20 + [300.0])
    rv = ti.relative_volume(v, 20)
    assert rv.iloc[-1] == pytest.approx(3.0)


def test_no_lookahead_indicators_unchanged_by_future_data():
    d = _ohlc()
    full = ti.add_all(d)
    part = ti.add_all(d.iloc[:200])
    cols = ["sma50", "ema21", "rsi14", "macd", "atr14", "adx", "rel_vol"]
    pd.testing.assert_frame_equal(full.iloc[:200][cols], part[cols], check_exact=False, rtol=1e-9, atol=1e-9)


def test_fibonacci_levels():
    f = ti.fibonacci_levels(110, 100, uptrend=True)
    assert f["0.000"] == 110 and f["1.000"] == 100
    assert f["0.500"] == pytest.approx(105)
    assert f["0.618"] == pytest.approx(110 - 6.18)


def test_gap_detection():
    d = pd.DataFrame({"Open": [10, 11, 9], "Close": [10, 10.5, 9.5]},
                     index=pd.bdate_range("2024-01-02", periods=3))
    g = ti.detect_gaps(d, min_pct=5)
    assert g.gap_up.tolist() == [False, True, False]
    assert g.gap_down.tolist() == [False, False, True]


def test_prev_week_levels_use_only_completed_prior_period():
    d = _ohlc(30)
    pv = ti.prev_period_levels(d, "W")
    first_week_end = d.index[d.index.to_period("W") == d.index[0].to_period("W")][-1]
    assert pv.loc[:first_week_end, "prev_high"].isna().all()
    assert pv.prev_high.notna().any()


def test_vwap_resets_each_day():
    idx = pd.date_range("2024-01-02 09:30", periods=3, freq="5min", tz="America/New_York").append(
        pd.date_range("2024-01-03 09:30", periods=2, freq="5min", tz="America/New_York"))
    d = pd.DataFrame({"High": 11.0, "Low": 9.0, "Close": [10, 10, 10, 20, 20], "Volume": 100.0}, index=idx)
    v = ti.vwap_session(d)
    assert v.iloc[3] == pytest.approx((11 + 9 + 20) / 3)
