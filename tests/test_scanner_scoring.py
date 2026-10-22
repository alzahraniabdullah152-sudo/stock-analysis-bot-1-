import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from scanner.config import ScannerConfig
from scanner.engine import evaluate_ticker, prefilter
from scanner.scoring import score_row
from scanner.states import classify
from tests.synth import make_minute_bars

CFG = ScannerConfig()
OPEN_NOW = datetime(2026, 10, 20, 16, 7, tzinfo=timezone.utc)      # 12:07 New York, regular session, Tuesday


def _row(**over):
    base = dict(Close=10.5, vwap=10.2, ema20=10.3, ema50=10.1, rvol=4.0, rvol_day=1.5, ret1=0.2, ret5=1.2, ret15=2.0,
                accel=0.6, level=10.4, dist_pct=0.96, ext_vwap_pct=2.9, avg_dollar_vol=30e6, spread_pct=0.2, rs15=1.0,
                bars_since_breakout=4.0, hold_ratio=1.0, failed_breakout=False)
    base.update(over)
    return pd.Series(base)


def test_score_is_always_between_0_and_100_and_finite():
    rng = np.random.default_rng(0)
    keys = list(_row().index)
    for _ in range(300):
        vals = {k: (np.nan if rng.random() < 0.3 else float(rng.normal(0, 50))) for k in keys if k not in ("failed_breakout",)}
        r = _row(**vals)
        s = score_row(r, CFG)
        assert 0.0 <= s.score <= 100.0 and math.isfinite(s.score)
        assert 0.0 <= s.coverage <= 1.0


def test_missing_data_lowers_score_and_coverage_instead_of_being_neutral():
    full = score_row(_row(), CFG)
    no_rvol = score_row(_row(rvol=np.nan), CFG)
    no_rs = score_row(_row(rs15=np.nan), CFG)
    assert no_rvol.score < full.score and no_rvol.coverage < full.coverage
    assert no_rs.score < full.score
    assert any(r.code == "rvol_missing" for r in no_rvol.reasons)
    empty = score_row(pd.Series({"Close": 10.0}), CFG)
    assert empty.score == 0.0 and empty.coverage == 0.0


def test_score_components_sum_to_score_and_weights_cap_points():
    s = score_row(_row(), CFG)
    assert abs(sum(c["points"] for c in s.components.values()) - s.score) < 0.2   # weights sum to 100
    for k, c in s.components.items():
        assert 0 <= c["points"] <= c["max"] + 1e-9


def test_extended_move_is_penalised_and_flagged():
    ok = score_row(_row(dist_pct=0.5), CFG)
    late = score_row(_row(dist_pct=9.0), CFG)
    assert late.extended and not ok.extended and late.score < ok.score
    assert any(r.code == "extended" for r in late.reasons)


def test_every_reason_has_arabic_and_english_text():
    for r in score_row(_row(), CFG).reasons + score_row(_row(rvol=np.nan, rs15=np.nan, dist_pct=-0.1), CFG).reasons:
        assert r.ar.strip() and r.en.strip() and r.kind in "+-!"


def _classify(row, **kw):
    sc = score_row(row, CFG)
    args = dict(market_open=True, data_age_s=30.0, signal_time=OPEN_NOW)
    args.update(kw)
    return classify(row, sc, CFG, **args), sc


def test_confirmed_requires_breakout_volume_momentum_and_score():
    c, _ = _classify(_row())
    assert c.state == "confirmed" and c.actionable and c.blocked is None
    assert _classify(_row(rvol=1.2))[0].state != "confirmed"
    assert _classify(_row(ret5=0.1))[0].state != "confirmed"
    assert _classify(_row(dist_pct=-0.1))[0].state == "near_breakout"


def test_missing_data_never_becomes_an_actionable_signal():
    c, sc = _classify(_row(rvol=np.nan, avg_dollar_vol=np.nan, rs15=np.nan, ret5=np.nan, dist_pct=np.nan))
    assert not c.actionable and c.state in ("none", "watch")
    c2, _ = _classify(pd.Series({"Close": 10.0, "level": np.nan}))
    assert c2.state == "none" and not c2.actionable


def test_market_closed_stale_and_extended_block_actions():
    assert _classify(_row(), market_open=False)[0].blocked == "market_closed"
    assert not _classify(_row(), market_open=False)[0].actionable
    st = _classify(_row(), data_age_s=3600.0)[0]
    assert st.blocked == "data_stale" and not st.actionable
    ext = _classify(_row(dist_pct=6.0))[0]
    assert ext.extended and not ext.actionable and ext.blocked == "extended"


def test_failed_breakout_and_expiry_from_prior_signal():
    prior = {"state": "confirmed", "level": 10.4, "time": OPEN_NOW - timedelta(minutes=5)}
    c, _ = _classify(_row(Close=10.3, dist_pct=-0.9, rvol=1.0), prior=prior)
    assert c.state == "failed"
    old = {"state": "near_breakout", "level": 10.4, "time": OPEN_NOW - timedelta(minutes=90)}
    c2, _ = _classify(_row(rvol=1.0, dist_pct=-1.0, ret5=0.0, accel=-0.1), prior=old)
    assert c2.state == "expired"
    assert _classify(_row(failed_breakout=True))[0].state == "failed"


def test_prefilter_reasons():
    assert "price" in prefilter(_row(Close=1.2), CFG)
    assert "price" in prefilter(_row(Close=80.0), CFG)
    assert "liquidity unknown" in prefilter(_row(avg_dollar_vol=np.nan), CFG)
    assert "dollar volume" in prefilter(_row(avg_dollar_vol=1e6), CFG)
    assert "spread" in prefilter(_row(spread_pct=2.0), CFG)
    assert prefilter(_row(), CFG) is None


def test_evaluate_ticker_end_to_end_on_planted_surge():
    bars = make_minute_bars(sessions=7, last_session_minutes=157, surge_at=145, surge_pct=4.0, surge_vol_mult=6)
    bench = make_minute_bars(seed=9, price=400, sessions=7, last_session_minutes=157)
    cand, why = evaluate_ticker("ABCD", bars, bench, CFG, now=OPEN_NOW, provider="test", delayed=True)
    assert why is None and cand is not None
    assert cand.state in ("early", "near_breakout", "confirmed") and 0 <= cand.score <= 100 and cand.delayed
    cand2, why2 = evaluate_ticker("PENNY", make_minute_bars(sessions=7, price=0.5, last_session_minutes=157), None, CFG, now=OPEN_NOW)
    assert cand2 is None and "price" in why2
