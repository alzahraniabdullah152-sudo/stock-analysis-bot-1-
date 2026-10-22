from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from data.market_data import DataError
from history.events import detect_events
from history.explain import explain_event, timing_vs_day
from history.financials import annual_table, flags
from history.loader import PriceHistory, normalize_history
from history.report import build_report, to_html
from history.sec import Filing, SecClient, SecError, lookup_cik, parse_filings
from scanner.catalysts import _parse_news_item, classify_headline
from tests import sec_fixtures as fx


def _daily(n=3000, seed=1, events=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    r = rng.normal(0.0004, 0.015, n)
    vol = rng.integers(1e6, 2e6, n).astype(float)
    if events:
        r[1000:1010] = -0.045
        r[1500] = 0.22
        r[2000] = np.log(0.5)                 # an UNADJUSTED 2:1 split looks like a -50% crash
        vol[1500] *= 8
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.005, "Low": np.minimum(o, c) * 0.995, "Close": c,
                       "Adj Close": c, "Volume": vol, "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)
    if events:
        df.loc[idx[2000], "Stock Splits"] = 2.0
    return df, idx


def _hist(df, ticker="TEST"):
    d, notes = normalize_history(df)
    return PriceHistory(ticker, d, f"{ticker} Corp", "NMS", "USD", notes=notes)


# ------------------------------------------------------------------------------- loader
def test_normalize_history_columns_timezones_and_minimum_size():
    df, idx = _daily(60, events=False)
    df.index = df.index.tz_localize("America/New_York")
    d, notes = normalize_history(df)
    assert d.index.tz is None and {"AdjClose", "Splits"} <= set(d.columns)
    with pytest.raises(DataError):
        normalize_history(df.iloc[:10])
    with pytest.raises(DataError):
        normalize_history(pd.DataFrame())
    _, n2 = normalize_history(df.drop(columns=["Adj Close"]))
    assert any("Dividend-adjusted" in x for x in n2)


# ------------------------------------------------------------------------------- events
def test_events_detect_surge_plunge_and_flag_split_artifact():
    df, idx = _daily()
    d, _ = normalize_history(df)
    ev = detect_events(d)
    assert (ev["type"] == "surge").any() and (ev["type"] == "drawdown").any()
    plunge_art = ev[(ev["type"] == "plunge") & ev["split_artifact"]]
    assert len(plunge_art) >= 1                          # the unadjusted split is NOT reported as a real crash
    surge = ev[(ev["type"] == "surge") & (ev["start"] == idx[1500])]
    assert len(surge) == 1 and surge.iloc[0]["change_pct"] > 20 and surge.iloc[0]["volume_peak_mult"] > 5
    assert not surge.iloc[0]["split_artifact"]


def test_market_driven_move_is_marked():
    df, idx = _daily()
    d, _ = normalize_history(df)
    bench = pd.DataFrame({"Close": d["Close"] * 3.0}, index=d.index)       # benchmark moves exactly with the stock
    ev = detect_events(d, bench)
    s = ev[(ev["type"] == "surge") & (~ev["split_artifact"])]
    assert s["market_driven"].all()


def test_events_use_only_past_volatility_for_thresholds():
    df, idx = _daily()
    d, _ = normalize_history(df)
    a = detect_events(d.iloc[:1800])
    b = detect_events(d)
    key = ["type", "start", "end"]
    a = a[(a["type"] != "drawdown") & (a["type"] != "stagnation") & (a["end"] < idx[1700])]
    m = a.merge(b[key], on=key, how="left", indicator=True)
    assert (m["_merge"] == "both").all()                  # earlier events are unchanged by later data


# ------------------------------------------------------------------------------- timing / evidence
def test_timing_vs_day_handles_daylight_saving():
    day = pd.Timestamp("2026-07-15")                     # EDT: open 13:30 UTC
    assert timing_vs_day(datetime(2026, 7, 15, 13, 0, tzinfo=timezone.utc), day) == ("before_open", True)
    assert timing_vs_day(datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc), day) == ("during_session", None)
    assert timing_vs_day(datetime(2026, 7, 15, 20, 30, tzinfo=timezone.utc), day) == ("after_close", False)
    winter = pd.Timestamp("2026-01-15")                  # EST: open 14:30 UTC
    assert timing_vs_day(datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc), winter)[0] == "before_open"
    assert timing_vs_day(datetime(2026, 1, 15, 14, 45, tzinfo=timezone.utc), winter)[0] == "during_session"
    assert timing_vs_day(None, winter) == ("unknown", None)


def _event(**kw):
    base = {"id": 1, "type": "surge", "start": pd.Timestamp("2017-10-02"), "end": pd.Timestamp("2017-10-02"),
            "change_pct": 40.0, "volume_mult": 2.8, "volume_peak_mult": 8.0, "market_ret_pct": 0.2, "excess_pct": 39.8, "market_driven": False,
            "split_artifact": False}
    base.update(kw)
    return pd.Series(base)


def _filing(accepted, items=("2.02", "9.01"), form="8-K"):
    return Filing(1, form, accepted[:10], datetime.fromisoformat(accepted.replace("Z", "+00:00")), "0001-17-1", tuple(items), "d.htm", "")


def test_filing_before_the_open_gives_direct_evidence_with_a_link():
    x = explain_event(_event(), [_filing("2017-10-02T11:30:00Z")], sec_available=True)
    assert x.classification == "direct_evidence" and x.evidence[0].knowable_before_move is True
    assert x.evidence[0].url.startswith("https://www.sec.gov/Archives/edgar/data/")
    assert any("filing text was not read" in c for c in x.caveats)
    assert any("no headlines" in c.lower() for c in x.caveats)          # no news is attributed to history


def test_filing_after_the_move_is_never_direct_evidence():
    x = explain_event(_event(), [_filing("2017-10-02T21:00:00Z")], sec_available=True)
    assert x.classification != "direct_evidence"
    assert any("after the move began" in a for a in x.alternatives)


def test_filing_during_the_session_cannot_establish_order():
    x = explain_event(_event(), [_filing("2017-10-02T16:00:00Z")], sec_available=True)
    assert x.classification in ("probable", "coincident")


def test_market_driven_move_is_not_attributed_to_the_company():
    x = explain_event(_event(market_driven=True, market_ret_pct=35.0, excess_pct=5.0), [_filing("2017-10-02T11:30:00Z")], sec_available=True)
    assert x.classification == "coincident" and any("benchmark" in a for a in x.alternatives)


def test_no_sec_access_or_no_evidence_means_unresolved_never_invented():
    assert explain_event(_event(), None, sec_available=False).classification == "unresolved"
    y = explain_event(_event(volume_mult=1.0, volume_peak_mult=1.0, excess_pct=1.0), [], sec_available=True)
    assert y.classification == "unresolved" and y.evidence == []


def test_split_artifact_is_labelled_corporate_action():
    x = explain_event(_event(type="plunge", change_pct=-50.0, split_artifact=True), [], sec_available=True)
    assert x.classification == "coincident" and "split" in x.summary.lower()


# ------------------------------------------------------------------------------- SEC parsers / client
def test_sec_lookup_filings_urls_and_item_weights():
    assert lookup_cik(fx.COMPANY_TICKERS, "TEST") == (1234567, "Test Corp")
    assert lookup_cik(fx.COMPANY_TICKERS, "BRK.B")[0] == 7654321          # BRK-B and BRK.B are the same name
    with pytest.raises(SecError):
        lookup_cik(fx.COMPANY_TICKERS, "NOPE")
    fl = parse_filings(1234567, fx.submissions()["filings"]["recent"])
    assert [f.form for f in fl] == ["8-K", "10-Q", "10-K"] and fl[0].items == ("2.02", "9.01")
    assert fl[0].url == "https://www.sec.gov/Archives/edgar/data/1234567/000123456717000010/ex99.htm"
    assert fl[0].weight() == "high" and fl[1].weight() == "medium"


def test_sec_is_disabled_without_a_contact_user_agent():
    c = SecClient(user_agent="")
    assert not c.enabled
    with pytest.raises(SecError):
        c.cik_for("AAPL")
    assert SecClient(user_agent="Me me@example.com").enabled


# ------------------------------------------------------------------------------- financials
def test_annual_financials_as_reported():
    df = annual_table(fx.company_facts())
    assert list(df["revenue"]) == [1000.0, 1200.0, 900.0]
    assert df["free_cash_flow"].iloc[1] == 120.0 and df["net_margin_%"].iloc[0] == pytest.approx(10.0)
    assert df["revenue_growth_%"].iloc[2] == pytest.approx(-25.0) and pd.isna(df["revenue_growth_%"].iloc[0])
    fl = " ".join(flags(df))
    assert "Net loss" in fl and "dilution" in fl and "Current ratio" in fl


def test_financials_use_latest_restatement_and_skip_quarterly_values():
    f = fx.company_facts()
    rows = f["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    rows.append(fx._fact("2020-12-31", 1250.0, "2020-01-01", filed="2021-06-01"))             # restated later
    rows.append(fx._fact("2020-09-30", 300.0, "2020-07-01", form="10-Q", fp="Q3"))            # quarterly: ignored
    rows.append(fx._fact("2020-12-31", 99.0, "2020-10-01", form="10-K", fp="FY"))             # 3-month duration: ignored
    df = annual_table(f)
    assert df.loc["2020-12-31", "revenue"] == 1250.0
    assert annual_table({"facts": {}}).empty


# ------------------------------------------------------------------------------- report / html
class _FakeSec(SecClient):
    def __init__(self):
        super().__init__(user_agent="Tester test@example.com")

    def cik_for(self, t):
        return lookup_cik(fx.COMPANY_TICKERS, t)

    def filings(self, cik, include_older=True):
        return parse_filings(cik, fx.submissions()["filings"]["recent"])

    def company_facts(self, cik):
        return fx.company_facts()


def test_report_html_is_escaped_labelled_and_sourced():
    df, idx = _daily()
    ph = _hist(df)
    ph.name = "<script>alert(1)</script> Corp"
    rep = build_report(ph, None, _FakeSec())
    h = to_html(rep)
    assert "<script>alert(1)</script>" not in h and "&lt;script&gt;" in h
    for needle in ("ليست نصيحة مالية", "المصادر وتغطية البيانات", "آخر تحديث للتقرير", "SEC EDGAR", "dir='rtl'", "أول تداول", "غير متاح"):
        assert needle in h
    assert rep.profile["data_start"] == idx[0]                         # actual coverage is stated, not assumed
    assert rep.explanations and not rep.financials.empty


def test_report_without_sec_says_so_and_still_builds():
    df, _ = _daily()
    rep = build_report(_hist(df), None, SecClient(user_agent=""))
    assert any("SEC features disabled" in n for n in rep.notes)
    assert all(x.classification in ("unresolved", "coincident") for x in rep.explanations.values())


def test_focus_period_filters_events():
    df, idx = _daily()
    rep = build_report(_hist(df), None, None, focus=(str(idx[1400].date()), str(idx[1600].date())))
    assert not rep.events.empty and (rep.events["start"] <= idx[1600]).all() and (rep.events["end"] >= idx[1400]).all()


# ------------------------------------------------------------------------------- catalysts
def test_news_parsing_requires_a_publication_time_and_tags_are_hints():
    new = {"content": {"title": "XYZ wins contract", "pubDate": "2026-10-20T14:01:00Z", "provider": {"displayName": "Reuters"},
                       "canonicalUrl": {"url": "https://x.test/a"}}}
    assert _parse_news_item(new)["publisher"] == "Reuters"
    assert _parse_news_item({"title": "no time"}) is None
    assert classify_headline("XYZ beats earnings, raises guidance")[0]["category"] == "earnings"
    assert classify_headline("XYZ prices $40 million public offering")[0]["negative"] is True


# ------------------------------------------------------------------------------- scanner <-> history bridge
def _breakout_series(n_events):
    rng = np.random.default_rng(4)
    n = 60 + 40 * n_events
    c = 50 + np.cumsum(rng.normal(0, 0.15, n))
    v = np.full(n, 1e6)
    for k in range(n_events):
        i = 40 + 40 * k
        c[i:] += 3.0                     # a jump above the 20-day high ...
        v[i] = 4e6                       # ... on 4x volume
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"Open": c, "High": c + 0.2, "Low": c - 0.2, "Close": c, "Volume": v}, index=idx)


def test_breakout_events_use_only_past_data_for_detection():
    from history.link import breakout_events
    df = _breakout_series(6)
    full, cut = breakout_events(df), breakout_events(df.iloc[:120])
    assert len(full) >= 5
    assert set(cut["date"]) <= set(full["date"]) and set(cut["date"][cut["date"] < df.index[110]]) == set(full["date"][full["date"] < df.index[110]])


def test_similar_patterns_withhold_statistics_for_small_samples():
    from history.link import similar_patterns
    small = similar_patterns(_breakout_series(3))
    assert small["stats"] is None and any("no statistics" in n for n in small["notes"])
    big = similar_patterns(_breakout_series(14), min_n=10)
    assert big["stats"] is not None and "5d" in big["stats"] and any("does not mean" in n for n in big["notes"])


def test_candidate_context_lists_only_recent_risk_filings():
    from history.link import candidate_context
    df = _breakout_series(3)
    now = datetime(2026, 10, 20, tzinfo=timezone.utc)
    risky = Filing(1, "8-K", "2026-09-15", datetime(2026, 9, 15, 20, tzinfo=timezone.utc), "a-1", ("3.02",), "d.htm", "")
    old = Filing(1, "8-K", "2025-01-15", datetime(2025, 1, 15, 20, tzinfo=timezone.utc), "a-2", ("3.02",), "d.htm", "")
    ctx = candidate_context(df, df, [risky, old], None, now)
    assert len(ctx["recent_filing_risks"]) == 1 and "dilution" in ctx["recent_filing_risks"][0]["risk"]
