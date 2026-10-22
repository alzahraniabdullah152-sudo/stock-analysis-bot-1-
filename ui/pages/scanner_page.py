"""Live scanner page: pick tickers, run in the background, read ranked candidates and why."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from alerts.alert_manager import load_watchlist
from data.market_data import DataError
from scanner.catalysts import fetch_catalysts
from scanner.engine import Scanner, ScanReport
from scanner.loop import MIN_INTERVAL_S, ScanLoop
from scanner.prescreen import daily_prescreen
from history.link import candidate_context, similar_patterns
from history.loader import load_price_history
from scanner.universe import load_universe, parse_ticker_text
from ui import jobs, state
from ui.components import (candidate_card, empty_state, html_block, note, reasons_html, state_badge, status_strip)
from ui.i18n import tr

DEFAULT_LIST = "AAPL, AMD, F, PLTR, SOFI, RIVN, NIO, SNAP, T, BAC"


def _run_scan(job: jobs.Job, ctx: dict, tickers: list[str], universe: bool, max_n: int, send_alerts: bool) -> ScanReport:
    """Runs in a background thread: it must not touch st.session_state, so everything it needs is passed in `ctx`."""
    cfg = ctx["cfg"]
    names: dict[str, str] = {}
    if universe:
        job.message = "universe"
        uni = load_universe()
        names = dict(zip(uni["symbol"], uni["name"]))
        job.message = "prescreen"
        pre = daily_prescreen(list(uni["symbol"]), cfg)
        tickers = list(pre["symbol"].head(max_n))
        if not tickers:
            rep = ScanReport()
            rep.warnings.append("no_symbols: nothing passed the daily price/liquidity pre-screen")
            return rep
    job.message = "scanning"
    scanner = Scanner(ctx["provider"], cfg, ctx["journal"], ctx["notifier"], names=names)
    return scanner.scan(tickers, send_alerts=send_alerts)


def _detail(c, lang: str, sec) -> None:
    with st.expander(f"{c.ticker} - {c.label_ar if lang == 'ar' else c.label_en}  ({c.score:.0f}/100)", expanded=False):
        candidate_card(c, lang)
        st.markdown(f"**{tr('reasons', lang)}**")
        html_block(reasons_html(c.reasons, lang))
        cols = st.columns(3)
        cols[0].metric("RVOL", "n/a" if pd.isna(c.features.get("rvol")) else f"{c.features['rvol']:.1f}x")
        cols[1].metric("5m %", "n/a" if pd.isna(c.features.get("ret5")) else f"{c.features['ret5']:+.2f}")
        cols[2].metric("vs 15m high %", "n/a" if pd.isna(c.features.get("dist_pct")) else f"{c.features['dist_pct']:+.2f}")
        if c.invalidation:
            st.write(f"{tr('invalidation', lang)}: **{c.invalidation:.2f}**")
        if c.expires_at:
            st.write(f"{tr('expires', lang)}: {c.expires_at:%H:%M} UTC")
        comp = pd.DataFrame(c.components).T.rename(columns={"points": "points", "max": "max", "available_weight": "data-backed weight"})
        st.dataframe(comp, use_container_width=True)
        if st.button("سياق تاريخي للسهم" if lang == "ar" else "Historical context", key=f"hist_{c.ticker}"):
            _history_context(c.ticker, lang)
        if st.button("المحفزات والأخبار" if lang == "ar" else "Catalysts & news", key=f"cat_{c.ticker}"):
            cats = fetch_catalysts(c.ticker, sec)
            if not cats:
                st.info("المحفز غير معروف: لم يُعثر على خبر أو إفصاح خلال 48 ساعة." if lang == "ar"
                        else "Catalyst unknown: no headline or filing in the last 48 hours.")
            for x in cats:
                flag = "⚠️ " if x["negative"] else ""
                st.markdown(f"{flag}[{x['title']}]({x['url']}) - {x['publisher']}, {x['published']:%Y-%m-%d %H:%M} UTC" if x.get("url")
                            else f"{flag}{x['title']} - {x['publisher']}, {x['published']:%Y-%m-%d %H:%M} UTC")
            if cats:
                st.caption("التصنيف آلي بالكلمات المفتاحية، تحقق من المصدر قبل أي استنتاج." if lang == "ar"
                           else "Tags are keyword-based hints; verify at the source.")


def _history_context(ticker: str, lang: str) -> None:
    """Optional bridge to the history engine: similar past patterns, risks and relative performance (descriptive)."""
    with st.spinner(tr("loading", lang)):
        try:
            ph = load_price_history(ticker)
            try:
                bh = load_price_history(state.cfg().benchmark)
            except DataError:
                bh = None
            sec = state.sec_client()
            filings = []
            if sec.enabled:
                try:
                    filings = sec.filings(sec.cik_for(ticker)[0], include_older=False)
                except Exception:
                    filings = []
        except DataError as exc:
            st.warning(f"{tr('data_error', lang)}: {exc}")
            return
    ctx = candidate_context(ph.df, bh.df if bh else None, filings, ph.splits)
    cols = st.columns(3)
    cols[0].metric("From 52w high", f"{ctx['from_52w_high_%']:.1f}%")
    cols[1].metric("20d return", f"{ctx['return_20d_%']:.1f}%")
    cols[2].metric("vs benchmark (20d)", "n/a" if "vs_benchmark_20d_pp" not in ctx else f"{ctx['vs_benchmark_20d_pp']:+.1f} pp")
    for r in ctx["recent_filing_risks"]:
        st.warning(f"{r['date']} {r['form']}: {r['risk']} ({r['url']})")
    for sp in ctx["recent_splits"]:
        st.warning(f"Split {sp['date']:%Y-%m-%d} ({sp['ratio']:g}:1) in the last 90 days")
    pat = similar_patterns(ph.df, bh.df if bh else None)
    st.markdown(f"**{'أنماط يومية مشابهة سابقاً' if lang == 'ar' else 'Similar daily breakouts before'}: {pat['n']}**")
    if pat["stats"]:
        st.json(pat["stats"])
    elif pat["n"]:
        ev = pat["events"].copy()
        ev["date"] = ev["date"].dt.date
        st.dataframe(ev, use_container_width=True, hide_index=True)
    for n_ in pat["notes"]:
        st.caption(n_)
    st.caption(f"{'المصدر' if lang == 'ar' else 'Source'}: yfinance daily history ({ph.start:%Y-%m-%d} → {ph.end:%Y-%m-%d})"
               + (" + SEC" if filings else ""))


def _results(rep: ScanReport, lang: str) -> None:
    last_bar = max((c.bar_time for c in rep.candidates), default=None)
    status_strip(rep.market_status, rep.provider or "-", rep.delayed, last_bar, lang)
    for w in rep.warnings:
        code = w.split(":")[0]
        if code == "market_closed":
            note(tr("blocked_market_closed", lang))
        elif code == "delayed_data":
            continue   # already shown in the strip
        else:
            note(w)
    shown = [c for c in rep.candidates if c.state != "none"]
    if not shown:
        empty_state(tr("no_candidates", lang), tr("no_candidates_hint", lang))
    else:
        for c in shown[:10]:
            candidate_card(c, lang)
        st.markdown("---")
        for c in shown:
            _detail(c, lang, state.sec_client())
        with st.expander("الجدول الكامل" if lang == "ar" else "Full table"):
            st.dataframe(rep.table(), use_container_width=True)
    if rep.skipped:
        with st.expander(f"{tr('skipped', lang)} ({len(rep.skipped)})"):
            st.dataframe(pd.DataFrame({"ticker": list(rep.skipped), "reason": list(rep.skipped.values())}), use_container_width=True)
    st.caption(f"{rep.scanned_at:%Y-%m-%d %H:%M:%S} UTC")


def render(lang: str) -> None:
    cfg = state.cfg()
    st.title(tr("nav_scanner", lang))
    mode = st.radio("المصدر" if lang == "ar" else "Source",
                    ["list", "watchlist", "universe"], horizontal=True,
                    format_func=lambda x: {"list": "قائمتي" if lang == "ar" else "My list",
                                           "watchlist": "قائمة المتابعة" if lang == "ar" else "Watchlist",
                                           "universe": "كل أسهم NASDAQ/NYSE" if lang == "ar" else "All NASDAQ/NYSE"}[x])
    tickers: list[str] = []
    max_n = 60
    if mode == "list":
        tickers = parse_ticker_text(st.text_area(tr("scan_list", lang), DEFAULT_LIST, height=90))
    elif mode == "watchlist":
        tickers = load_watchlist()
        st.write(", ".join(tickers))
    else:
        max_n = st.slider("أقصى عدد أسهم بعد الفلترة الأولية" if lang == "ar" else "Max tickers after pre-screen", 10, 150, 60, 10)
        st.caption("يفلتر أولاً بالسعر والسيولة اليومية ثم يفحص الأعلى سيولة بالدقيقة. قد يستغرق دقائق." if lang == "ar"
                   else "Pre-filters by price/daily liquidity, then scans the most liquid on 1-minute bars. May take minutes.")
    notifier = state.notifier()
    send = st.checkbox(tr("send_alerts", lang), False, disabled=not notifier.enabled)
    st.caption(tr("telegram_on" if notifier.enabled else "telegram_off", lang))
    note(f"{cfg.min_price:g}-{cfg.max_price:g}$ | ≥ ${cfg.min_avg_dollar_volume:,.0f}/day | "
         f"breakout {cfg.breakout_lookback}m")

    c1, c2 = st.columns(2)
    run = c1.button(tr("run_scan", lang), type="primary", disabled=(mode != "universe" and not tickers))
    if run:
        ctx = {"cfg": cfg, "provider": state.provider(), "journal": state.journal(), "notifier": notifier}
        jobs.start("scan", lambda j: _run_scan(j, ctx, tickers, mode == "universe", max_n, send))
    loop = st.session_state.get("scan_loop")
    with c2:
        auto = st.checkbox("فحص تلقائي متكرر" if lang == "ar" else "Repeat scan automatically", bool(loop and loop.running))
        mins = st.number_input("كل (دقائق)" if lang == "ar" else "Every (minutes)", MIN_INTERVAL_S // 60, 60, 5)
    if auto and mode != "universe" and tickers and not (loop and loop.running):
        loop = ScanLoop(state.make_scanner(), lambda: tickers, int(mins) * 60, send)
        st.session_state["scan_loop"] = loop
        loop.start()
    if not auto and loop and loop.running:
        loop.stop()

    running = jobs.get("scan") is not None and jobs.get("scan").running

    @st.fragment(run_every=2 if (running or (loop and loop.running)) else None)
    def view():
        job = jobs.get("scan")
        lp = st.session_state.get("scan_loop")
        if job and job.running:
            st.progress(0.3, text=f"{tr('loading', lang)} ({job.message}, {job.seconds:.0f}s)")
        if job and job.status == "error":
            st.error(f"{tr('data_error', lang)}: {job.error}")
        rep = None
        if lp and lp.latest is not None:
            rep = lp.latest
            st.caption(f"auto-scan: {lp.runs} runs" + (f", last error: {lp.errors[-1]}" if lp.errors else ""))
        if job and job.status == "done" and job.result is not None:
            rep = job.result if rep is None or job.result.scanned_at >= rep.scanned_at else rep
        if rep is None:
            empty_state("لم تبدأ فحصاً بعد" if lang == "ar" else "No scan yet",
                        "اختر القائمة واضغط «ابدأ الفحص»." if lang == "ar" else "Pick a list and press Run scan.")
        else:
            st.session_state["last_report"] = rep
            _results(rep, lang)

    view()
