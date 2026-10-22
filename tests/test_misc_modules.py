import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from scanner.catalysts import categorize, parse_news
from scanner.config import ScannerConfig
from scanner.providers import FrameProvider, load_bars_csv, split_download
from scanner.stream import Deduper, ReconnectingStream, valid_timestamp
from scanner.universe import clean_universe, parse_nasdaq_listed, parse_other_listed, parse_ticker_text
from data.market_data import DataError
from tests.synth import make_minute_bars

NOW = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)


def test_config_validation_and_roundtrip():
    c = ScannerConfig()
    c.validate()
    p = Path(tempfile.mkdtemp()) / "s.json"
    c.min_price = 3.5
    c.save(p)
    assert ScannerConfig.load(p).min_price == 3.5
    bad = ScannerConfig()
    bad.weights = {"nonsense": 10}
    with pytest.raises(ValueError):
        bad.validate()
    bad2 = ScannerConfig(min_price=10, max_price=5)
    with pytest.raises(ValueError):
        bad2.validate()
    p.write_text("{not json")
    assert ScannerConfig.load(p).min_price == 2.0          # unreadable file falls back to defaults


def test_stream_dedup_invalid_timestamps_and_reconnect_with_backoff():
    good = lambda i, ts=NOW: {"symbol": "A", "price": 10 + i, "size": 1, "ts": ts, "id": i}
    attempts = {"n": 0}

    def connect():
        attempts["n"] += 1
        if attempts["n"] == 1:
            yield good(1)
            yield good(1)                                         # exact duplicate
            yield good(2, NOW + timedelta(minutes=5))             # from the future: invalid
            raise ConnectionError("dropped")
        if attempts["n"] == 2:
            yield good(3)
            yield {"symbol": "A", "price": 1, "ts": None}         # missing timestamp: invalid

    got, sleeps = [], []
    s = ReconnectingStream(connect, got.append, sleep=sleeps.append, clock=lambda: NOW, base_delay=1, max_delay=4)
    s.run(max_attempts=3)
    assert [m["id"] for m in got] == [1, 3]
    assert s.dropped_duplicates == 1 and s.dropped_invalid == 2 and s.reconnects == 3
    assert "ConnectionError" in s.errors[0] and sleeps[:2] == [1, 1]   # healthy traffic resets back-off
    assert not s.connected and s.last_message_at is not None and not s.stale(30)


def test_backoff_grows_when_the_connection_keeps_failing():
    def boom():
        raise OSError("down")
        yield  # pragma: no cover

    sleeps = []
    s = ReconnectingStream(boom, lambda m: None, sleep=sleeps.append, clock=lambda: NOW, base_delay=1, max_delay=4)
    s.run(max_attempts=5)
    assert sleeps == [1, 2, 4, 4, 4]


def test_deduper_is_bounded_and_timestamp_validation():
    d = Deduper(size=2)
    assert not d.seen("a") and d.seen("a") and not d.seen("b") and not d.seen("c") and not d.seen("a")
    assert valid_timestamp(NOW, NOW) and not valid_timestamp(NOW - timedelta(days=1), NOW)
    assert not valid_timestamp(datetime(2026, 10, 20, 15, 0), NOW)  # naive


NAS = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\nAAPL|Apple Inc.|Q|N|N|100|N|N\nQQQ|Invesco QQQ|G|N|N|100|Y|N\nZZZT|Test|G|Y|N|100|N|N\nFile Creation Time: 1020202608:00|||||||\n"
OTH = "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\nF|Ford|N|F|N|100|N|F\nBRK.B|Berkshire|N|BRK.B|N|100|N|BRK-B\nSPY|SPDR|P|SPY|Y|100|N|SPY\nFile Creation Time: 1020202608:00|||||||\n"


def test_universe_parsing_filters_etf_test_issues_and_other_exchanges():
    df = pd.concat([parse_nasdaq_listed(NAS), parse_other_listed(OTH)], ignore_index=True)
    assert clean_universe(df)["symbol"].tolist() == ["AAPL", "F"]
    assert "QQQ" in clean_universe(df, include_etf=True)["symbol"].tolist()
    assert parse_ticker_text("aapl, msft  nvda;TOOLONGTICKER x1") == ["AAPL", "MSFT", "NVDA"]


def test_yfinance_download_shape_is_split_per_ticker_with_errors_for_missing():
    cols = pd.MultiIndex.from_product([["AAA", "BBB"], ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(1.0, index=pd.date_range("2026-10-20 10:00", periods=5, freq="min"), columns=cols)
    df[("BBB", "Close")] = float("nan")
    for c in ("Open", "High", "Low", "Volume"):
        df[("BBB", c)] = float("nan")
    out = split_download(df, ["AAA", "BBB", "CCC"])
    assert isinstance(out["AAA"], pd.DataFrame) and isinstance(out["BBB"], DataError) and isinstance(out["CCC"], DataError)
    assert isinstance(split_download(pd.DataFrame(), ["A"])["A"], DataError)


def test_csv_loader_localises_naive_timestamps_to_new_york():
    p = Path(tempfile.mkdtemp()) / "b.csv"
    p.write_text("timestamp,open,high,low,close,volume\n2026-10-20 09:30:00,10,10.1,9.9,10.05,1000\n2026-10-20 09:31:00,10.05,10.2,10,10.1,900\n")
    d = load_bars_csv(p)
    assert str(d.index.tz) == "America/New_York" and d.index[0].hour == 9 and list(d.columns)[:2] == ["Open", "High"]
    p.write_text("a,b\n1,2\n")
    with pytest.raises(DataError):
        load_bars_csv(p)


def test_frame_provider_reports_missing_ticker_instead_of_inventing_data():
    prov = FrameProvider({"A": make_minute_bars(sessions=3)})
    out = prov.bars(["A", "B"], 5)
    assert isinstance(out["B"], DataError) and not prov.realtime


def test_news_parser_drops_future_dated_undated_and_old_items_and_tags_categories():
    raw = [{"content": {"title": "XYZ prices public offering", "pubDate": "2026-10-20T13:00:00Z"}},
           {"title": "Old", "providerPublishTime": int(NOW.timestamp()) - 5 * 86400, "link": "x"},
           {"title": "From the future", "providerPublishTime": int(NOW.timestamp()) + 6 * 3600},
           {"title": "No time at all"}]
    items = parse_news(raw, NOW)
    assert [i["title"] for i in items] == ["XYZ prices public offering"]
    assert items[0]["negative"] and "offering/dilution" in items[0]["category"]
    assert categorize("FDA approval and contract award") == ["contract", "regulatory"]
    assert categorize("Nothing special") == []
