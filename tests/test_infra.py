import os
import tempfile
from datetime import datetime, timedelta, timezone

import pandas as pd

from scanner.config import ScannerConfig
from scanner.signal_log import SignalLog
from scanner.stream import Deduper, ReconnectingStream, valid_timestamp
from scanner.universe import clean_universe, parse_nasdaq_listed, parse_other_listed, parse_ticker_text

T0 = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)

NASDAQ = ("Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
          "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\nQQQ|Invesco QQQ|G|N|N|100|Y|N\nZZZT|Test Co|G|Y|N|100|N|N\n"
          "File Creation Time: 1020202608:00|||||||\n")
OTHER = ("ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
         "F|Ford Motor Company|N|F|N|100|N|F\nBRK.B|Berkshire|N|BRK.B|N|100|N|BRK-B\nSPY|SPDR|P|SPY|Y|100|N|SPY\n"
         "File Creation Time: 1020202608:00|||||||\n")


def test_universe_parsers_drop_etfs_test_issues_footers_and_odd_tickers():
    df = pd.concat([parse_nasdaq_listed(NASDAQ), parse_other_listed(OTHER)], ignore_index=True)
    assert list(clean_universe(df)["symbol"]) == ["AAPL", "F"]
    assert "QQQ" in set(clean_universe(df, include_etf=True)["symbol"])
    assert parse_ticker_text("aapl, msft  nvda;toolongtkr\nF") == ["AAPL", "MSFT", "NVDA", "F"]


def test_stream_dedups_rejects_bad_timestamps_and_reconnects_with_backoff():
    got, sleeps = [], []
    msgs = [{"symbol": "A", "price": 1.0, "size": 1, "ts": T0}, {"symbol": "A", "price": 1.0, "size": 1, "ts": T0},
            {"symbol": "A", "price": 2.0, "size": 1, "ts": T0 + timedelta(hours=1)},     # from the future
            {"symbol": "A", "price": 3.0, "size": 1, "ts": None}, {"symbol": "A", "price": 4.0, "size": 1, "ts": T0 - timedelta(seconds=2)}]
    calls = {"n": 0}

    def connect():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("dropped")
        return iter(msgs)

    s = ReconnectingStream(connect, got.append, sleep=sleeps.append, clock=lambda: T0)
    s.run(max_attempts=3)
    assert [m["price"] for m in got] == [1.0, 4.0]
    assert s.dropped_duplicates == 1 and s.dropped_invalid == 2 and s.reconnects == 3 and sleeps[:2] == [1.0, 2.0]
    assert s.errors and not s.connected


def test_deduper_is_bounded_and_timestamp_rules():
    d = Deduper(size=3)
    assert [d.seen(i) for i in (1, 2, 3, 1)] == [False, False, False, True]
    d.seen(4)
    assert d.seen(1) is False                                  # oldest key was evicted
    assert valid_timestamp(T0, T0) and not valid_timestamp(T0 - timedelta(hours=5), T0) and not valid_timestamp(datetime(2026, 1, 1), T0)


def _log():
    return SignalLog(os.path.join(tempfile.mkdtemp(), "j.db"))


def _rec(j, state, score, t, level=10.0, bar=None):
    return j.record(ticker="X", signal_time=t, bar_time=bar or t, state=state, score=score, coverage=0.9, ref_price=10.0,
                    level=level, invalidation=9.5, reasons=[], features={"rvol": 3.0, "session": "ignored"}, provider="t", data_age_s=5)


def test_journal_record_is_idempotent_and_alert_rules():
    j, cfg = _log(), ScannerConfig()
    sid = _rec(j, "near_breakout", 60, T0)
    assert sid and _rec(j, "near_breakout", 60, T0) is None                  # same ticker+bar+state is stored once
    assert j.should_alert("X", "near_breakout", 60, 10.0, T0, cfg).send       # first alert
    j.mark_alerted(sid)
    assert not j.should_alert("X", "near_breakout", 62, 10.0, T0 + timedelta(minutes=5), cfg).send   # same state, no new event
    assert j.should_alert("X", "confirmed", 70, 10.0, T0 + timedelta(minutes=5), cfg).send           # state upgraded
    assert j.should_alert("X", "near_breakout", 80, 10.0, T0 + timedelta(minutes=5), cfg).send       # material score jump
    assert j.should_alert("X", "near_breakout", 61, 10.5, T0 + timedelta(minutes=5), cfg).send       # new higher level
    assert not j.should_alert("X", "watch", 70, 10.0, T0, cfg).send and not j.should_alert("X", "expired", 70, 10.0, T0, cfg).send


def test_failed_breakout_alerts_only_after_a_confirmed_alert_and_new_episodes_after_reset():
    j, cfg = _log(), ScannerConfig()
    sid = _rec(j, "confirmed", 75, T0)
    j.mark_alerted(sid)
    assert j.should_alert("X", "failed", 30, 10.0, T0 + timedelta(minutes=10), cfg).send
    _rec(j, "none", 10, T0 + timedelta(minutes=20), bar=T0 + timedelta(minutes=20))          # setup reset
    assert j.should_alert("X", "confirmed", 76, 10.0, T0 + timedelta(minutes=cfg.cooldown_minutes + 5), cfg).send
    j2 = _log()
    s2 = _rec(j2, "near_breakout", 60, T0)
    j2.mark_alerted(s2)
    assert not j2.should_alert("X", "failed", 30, 10.0, T0 + timedelta(minutes=10), cfg).send   # never alerted confirmed


def test_scanner_config_roundtrip_and_corrupt_file_fallback():
    p = os.path.join(tempfile.mkdtemp(), "s.json")
    from pathlib import Path
    c = ScannerConfig(min_price=3.0)
    c.save(Path(p))
    assert ScannerConfig.load(Path(p)).min_price == 3.0
    Path(p).write_text("{not json")
    assert ScannerConfig.load(Path(p)).min_price == 2.0                       # defaults, no crash
