import os
import tempfile
from datetime import datetime, timedelta, timezone

import pandas as pd

from scanner.config import ScannerConfig
from scanner.engine import Scanner
from scanner.providers import FrameProvider
from scanner.signal_log import SignalLog, outcome_for
from tests.synth import make_minute_bars

CFG = ScannerConfig()
T0 = datetime(2026, 10, 20, 16, 7, tzinfo=timezone.utc)


def _log():
    return SignalLog(os.path.join(tempfile.mkdtemp(), "s.db"))


def _rec(j, ticker="ABCD", state="confirmed", score=70.0, level=10.4, t=T0, bar=None):
    return j.record(ticker=ticker, signal_time=t, bar_time=bar or t, state=state, score=score, coverage=0.9, ref_price=10.5,
                    level=level, invalidation=10.4, reasons=[{"code": "x"}], features={"rvol": 4.0, "session": "x"},
                    provider="test", data_age_s=30.0)


def test_record_is_idempotent_per_ticker_bar_state():
    j = _log()
    assert _rec(j) is not None
    assert _rec(j) is None
    assert len(j.signals()) == 1


def test_alert_dedup_rules():
    j = _log()
    assert j.should_alert("ABCD", "confirmed", 70, 10.4, T0, CFG).send                       # first alert
    sid = _rec(j, state="near_breakout", score=60.0)
    j.mark_alerted(sid)
    assert not j.should_alert("ABCD", "near_breakout", 61, 10.4, T0 + timedelta(minutes=5), CFG).send   # duplicate
    assert j.should_alert("ABCD", "confirmed", 62, 10.4, T0 + timedelta(minutes=5), CFG).send           # state upgrade
    assert j.should_alert("ABCD", "near_breakout", 80, 10.4, T0 + timedelta(minutes=5), CFG).send       # material score jump
    assert j.should_alert("ABCD", "near_breakout", 61, 10.9, T0 + timedelta(minutes=5), CFG).send       # new higher level
    assert not j.should_alert("ABCD", "watch", 99, None, T0, CFG).send                                  # not alertable
    assert not j.should_alert("ABCD", "failed", 10, 10.4, T0, CFG).send                                 # failure only after confirmed alert


def test_failure_alert_after_confirmed_and_new_episode_after_reset():
    j = _log()
    sid = _rec(j, state="confirmed")
    j.mark_alerted(sid)
    assert j.should_alert("ABCD", "failed", 20, 10.4, T0 + timedelta(minutes=10), CFG).send
    assert not j.should_alert("ABCD", "confirmed", 70, 10.4, T0 + timedelta(minutes=40), CFG).send      # no reset yet
    _rec(j, state="none", score=10.0, t=T0 + timedelta(minutes=20), bar=T0 + timedelta(minutes=20))
    assert j.should_alert("ABCD", "confirmed", 70, 10.4, T0 + timedelta(minutes=45), CFG).send          # new episode, cooldown over
    assert not j.should_alert("ABCD", "confirmed", 70, 10.4, T0 + timedelta(minutes=21), CFG).send      # reset but inside cooldown


def _outcome_bars(highs, lows, start=T0 + timedelta(minutes=1)):
    idx = pd.date_range(start, periods=len(highs), freq="min", tz="UTC")
    return pd.DataFrame({"Open": 10.0, "High": highs, "Low": lows, "Close": 10.0, "Volume": 1.0}, index=idx)


def test_outcomes_hit_miss_ambiguous_pending():
    now = T0 + timedelta(hours=3)
    hit = outcome_for(_outcome_bars([10.1, 10.3, 10.1], [9.95, 10.0, 10.0]), T0, 10.0, 2.0, 15, 1.5, now)
    assert hit["status"] == "hit" and hit["minutes_to_target"] == 2
    miss = outcome_for(_outcome_bars([10.05, 10.05], [9.7, 9.9]), T0, 10.0, 2.0, 15, 1.5, now)
    assert miss["status"] == "miss"
    amb = outcome_for(_outcome_bars([10.4], [9.7]), T0, 10.0, 2.0, 15, 1.5, now)
    assert amb["status"] == "ambiguous"                      # target and stop inside the same minute
    pend = outcome_for(_outcome_bars([10.05], [9.95]), T0, 10.0, 2.0, 15, 1.5, T0 + timedelta(minutes=3))
    assert pend["status"] == "pending"
    none = outcome_for(_outcome_bars([10.4], [9.7], start=T0 - timedelta(minutes=30)), T0, 10.0, 2.0, 15, 1.5, now)
    assert none["status"] == "no_data"


class _Notifier:
    enabled = True

    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)
        return True


def test_scanner_journals_and_alerts_once_for_the_same_event():
    bars = make_minute_bars(sessions=7, last_session_minutes=157, surge_at=145, surge_pct=4.0, surge_vol_mult=6)
    bench = make_minute_bars(seed=9, price=400, sessions=7, last_session_minutes=157)
    prov = FrameProvider({"ABCD": bars, "SPY": bench})
    j, n = _log(), _Notifier()
    sc = Scanner(prov, CFG, j, n, names={"ABCD": "Abcd Corp"})
    r1 = sc.scan(["ABCD", "ZZZ"], now=T0, send_alerts=True)
    r2 = sc.scan(["ABCD"], now=T0 + timedelta(minutes=1), send_alerts=True)
    assert "ZZZ" in r1.skipped and r1.delayed and any("delayed" in w for w in r1.warnings)
    assert len(n.sent) == 1 and "ABCD" in n.sent[0] and "ليست احتمال" in n.sent[0] and "تعليمية" in n.sent[0]
    assert r1.candidates[0].alert_sent and not r2.candidates[0].alert_sent
    assert (j.signals()["alerted"] == 1).sum() == 1


def test_closed_market_scan_is_never_actionable_and_sends_nothing():
    bars = make_minute_bars(sessions=7, last_session_minutes=157, surge_at=145, surge_pct=4.0, surge_vol_mult=6)
    prov = FrameProvider({"ABCD": bars, "SPY": make_minute_bars(seed=9, price=400, sessions=7, last_session_minutes=157)})
    n = _Notifier()
    rep = Scanner(prov, CFG, _log(), n).scan(["ABCD"], now=datetime(2026, 10, 24, 16, 0, tzinfo=timezone.utc), send_alerts=True)
    assert rep.market_status == "closed" and not any(c.actionable for c in rep.candidates) and not n.sent
