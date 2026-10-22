"""Single-stock technical analysis (the original app's analysis, risk calculator, backtest and alerts), unchanged logic."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from alerts.alert_manager import (RULE_KINDS, AlertManager, AlertRule, Scheduler, TelegramNotifier, load_watchlist,
                                  save_watchlist)
from analysis.report import analyze, scenario_text
from analysis.risk_management import RiskInputError, position_size
from analysis.trend_analysis import analyze_trend, mtf_agreement
from backtest.backtester import BacktestConfig, run_in_out_of_sample
from charts.interactive_chart import OVERLAYS, build_chart
from config import INTERVALS, INTRADAY, SETTINGS
from data.market_data import DataError, fetch
from indicators.technical_indicators import add_all
from ui.components import status_strip
from ui.i18n import tr


@st.cache_data(ttl=120, show_spinner=False)
def cached_fetch(ticker: str, tf: str, provider: str, provisional: bool):
    return fetch(ticker, tf, provider=provider, include_incomplete=provisional)


def render(lang: str) -> None:
    st.title(tr("nav_analysis", lang))
    with st.expander("الإعدادات" if lang == "ar" else "Settings", expanded=True):
        a, b, c = st.columns(3)
        ticker = a.text_input("Ticker", st.session_state.get("analysis_ticker", "AAPL")).strip().upper()
        st.session_state["analysis_ticker"] = ticker
        tf = b.selectbox("Timeframe", list(INTERVALS), index=4)
        provider = c.selectbox("Provider", ["yfinance", "alphavantage"], index=0 if SETTINGS.data_provider == "yfinance" else 1)
        provisional = st.checkbox("Include unfinished (provisional) candle", False,
                                  help="Off by default: only completed candles are analysed.")
        d1, d2 = st.columns(2)
        overlays = d1.multiselect("Overlays", list(OVERLAYS), ["SMA 50", "SMA 200", "EMA 21"])
        mtf_tfs = d2.multiselect("Multi-timeframe check", [k for k in INTERVALS if k != tf],
                                 [k for k in ["1d", "1wk"] if k != tf])
        e1, e2, e3, e4 = st.columns(4)
        show_bb = e1.checkbox("Bollinger Bands", False)
        show_vwap = e2.checkbox("VWAP (intraday)", tf in INTRADAY)
        show_sr = e3.checkbox("Support / resistance zones", True)
        show_markers = e4.checkbox("Historical setup markers", True)

    tab_an, tab_risk, tab_bt, tab_watch = st.tabs(["Analysis", "Risk calculator", "Backtest", "Watchlist & alerts"])
    report = None
    with tab_an:
        try:
            md = cached_fetch(ticker, tf, provider, provisional)
            mtf_states = {}
            for t in mtf_tfs + [tf]:
                try:
                    m = md if t == tf else cached_fetch(ticker, t, provider, False)
                    mtf_states[t] = analyze_trend(add_all(m.df, t in INTRADAY))
                except DataError as exc:
                    st.info(f"{t}: unavailable ({exc})")
            report = analyze(md, mtf_states)
        except DataError as exc:
            st.error(f"{tr('data_error', lang)}: {exc}")

        if report:
            md, d, tr_ = report.md, report.d, report.trend
            status_strip(md.market_status, md.provider, md.data_type != "historical", md.last_bar_time.to_pydatetime()
                         if hasattr(md.last_bar_time, "to_pydatetime") else None, lang)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Last close", f"{report.last_price:.2f}")
            c2.metric("Trend", tr_.direction.title())
            c3.metric("Momentum", tr_.momentum.title())
            sc = report.score["score"]
            c4.metric("Confluence (heuristic)", "n/a" if sc is None else f"{sc:.0f}/100")
            st.caption(f"Data: {md.data_type} via {md.provider} | last bar {md.last_bar_time} | fetched "
                       f"{md.fetched_at:%Y-%m-%d %H:%M UTC} | session: {md.market_status} | "
                       f"last candle {'COMPLETE' if md.last_bar_complete else 'UNFINISHED (provisional)'}. "
                       "Free providers are delayed, never real-time.")
            st.plotly_chart(build_chart(d, f"{md.ticker} {tf}", overlays, show_bb, show_vwap,
                                        report.zones if show_sr else None,
                                        report.markers if show_markers else None,
                                        report.setups[0] if report.setups else None, show_sr),
                            use_container_width=True)
            a_, b_ = st.columns(2)
            with a_:
                st.subheader("Trend & structure")
                st.write(f"Short / medium / long: **{report.horizons['short']} / {report.horizons['medium']} / "
                         f"{report.horizons['long']}**")
                st.write(f"Structure: **{tr_.structure}** | ADX: {'n/a' if tr_.adx is None else f'{tr_.adx:.1f}'} | "
                         f"Volume confirms: {tr_.volume_confirms}")
                st.write("; ".join(tr_.reasons) or "Insufficient history")
                if len(mtf_states) > 1:
                    txt, s = mtf_agreement(mtf_states)
                    st.write(f"Multi-timeframe: **{txt}** (weighted agreement {s:+.2f}). Higher timeframes carry more "
                             "weight: a setup against the weekly/daily trend is lower reliability.")
                st.write(f"Nearest support: {'none' if not report.support else f'{report.support.low:.2f}-{report.support.high:.2f}'}"
                         f" | resistance: {'none' if not report.resistance else f'{report.resistance.low:.2f}-{report.resistance.high:.2f}'}")
                st.write(f"Breakout level (20-bar high): {report.key_levels['breakout']:.2f} | "
                         f"breakdown level: {report.key_levels['breakdown']:.2f}")
            with b_:
                st.subheader("Confluence score (heuristic)")
                st.json({"score": report.score["score"], "components(-1..1)": report.score["components"],
                         "weights": report.score["weights"]})
                st.caption("score = 50 + 50 * sum(w*c)/sum(w) over available components. Missing components are "
                           "dropped and the rest re-normalised. NOT a probability of profit.")
            st.subheader(f"Setups on the latest {'completed' if md.last_bar_complete else 'provisional'} bar")
            if not report.setups:
                st.write("No rule-based setup triggered on the latest bar.")
            for s in report.setups:
                with st.expander(f"{s.label} - {s.bias.upper()}", expanded=True):
                    st.write("**Triggered by:** " + "; ".join(s.conditions))
                    st.write(f"Entry ref {s.entry:.2f} | invalidation {s.invalidation:.2f} | stop ref {s.stop:.2f} | "
                             f"targets {', '.join(f'{t:.2f}' for t in s.targets)} | "
                             f"R:R {'n/a' if s.rr is None else f'{s.rr:.2f}'}")
                    st.write("Supporting: " + (", ".join(s.supporting) or "none") +
                             " | Conflicting: " + (", ".join(s.conflicting) or "none"))
                    st.caption(scenario_text(s) + " " + s.freshness)

    with tab_risk:
        st.subheader("Position-size calculator (hypothetical)")
        default = report.setups[0] if report and report.setups else None
        c1, c2, c3 = st.columns(3)
        eq = c1.number_input("Account equity ($)", 100.0, 1e9, float(SETTINGS.default_equity))
        rp = c2.number_input("Risk per trade (%)", 0.05, 5.0, float(min(SETTINGS.default_risk_pct, 5.0)), 0.05)
        maxp = c3.number_input("Max position (% equity)", 1.0, 100.0, 100.0)
        e = c1.number_input("Entry", 0.01, 1e6, float(default.entry if default else (report.last_price if report else 100.0)))
        sl = c2.number_input("Stop", 0.01, 1e6, float(default.stop if default else (report.last_price * 0.97 if report else 97.0)))
        tg = c3.text_input("Targets (comma separated)", ", ".join(f"{t:.2f}" for t in default.targets) if default else "")
        com = c1.number_input("Commission / share ($)", 0.0, 10.0, float(SETTINGS.commission_per_share))
        slp = c2.number_input("Slippage (%)", 0.0, 5.0, float(SETTINGS.slippage_pct))
        frac = c3.checkbox("Allow fractional shares", False)
        try:
            targets = [float(x) for x in tg.split(",") if x.strip()]
            p = position_size(eq, rp, e, sl, targets, com, slp, frac, maxp)
            st.write(f"Side **{p.side}** | shares **{p.shares}** | modelled fill {p.entry:.2f} | modelled stop fill {p.stop:.2f}")
            st.write(f"Dollar risk on stop-out (incl. costs): **${p.dollar_risk:,.2f}** | position value ${p.position_value:,.2f}")
            for t in p.targets:
                st.write(f"Target {t['target']:.2f}: " + (f"reward ${t['reward']:,.2f}, R {t['rr']:.2f}" if t["valid"] else "invalid for this side"))
            for w in p.warnings:
                st.warning(w)
            st.caption("Hypothetical. Real fills, gaps and halts can produce losses larger than the stop distance.")
        except (RiskInputError, ValueError) as exc:
            st.error(str(exc))

    with tab_bt:
        st.subheader("Backtest (in-sample vs out-of-sample)")
        split = st.slider("In-sample fraction", 0.3, 0.9, 0.7, 0.05)
        if st.button("Run backtest") and report:
            res = run_in_out_of_sample(report.d, split, BacktestConfig(equity=SETTINGS.default_equity),
                                       intraday=tf in INTRADAY, ticker=ticker)
            for k, label in (("in_sample", "In-sample (development)"), ("out_of_sample", "Out-of-sample (judge here)")):
                st.markdown(f"**{label}**")
                st.dataframe(pd.DataFrame([res[k]["metrics"]]).T.rename(columns={0: "value"}))
                st.line_chart(res[k]["equity_curve"])
                if not res[k]["trades"].empty:
                    st.dataframe(res[k]["trades"].tail(50))
            st.caption("Stop assumed first when a bar touches both stop and target; entries at next open with slippage. "
                       "Backtested results do not guarantee future performance.")

    with tab_watch:
        st.subheader("Watchlist")
        wl = st.text_input("Tickers (comma separated)", ", ".join(load_watchlist()))
        tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]
        if st.button("Save watchlist"):
            save_watchlist(tickers)
            st.success(tr("saved", lang))
        kinds = st.multiselect("Alert rules", list(RULE_KINDS), ["ema_cross_up", "ema_cross_down", "breakout", "breakdown"])
        lvl = st.number_input("Level for price/RSI rules", 0.0, 1e6, 70.0)
        rules = [AlertRule(k, lvl if k.startswith(("price", "rsi")) else None) for k in kinds]
        scan_tf = st.selectbox("Scan timeframe", ["1d", "1h", "15m", "5m"], 0)
        prov_scan = st.checkbox("Scan unfinished candle (labelled provisional)", False)
        if "manager" not in st.session_state:
            st.session_state.manager = AlertManager()
        if st.button("Scan now"):
            with st.spinner("Scanning (rate-limited)..."):
                hits = st.session_state.manager.scan(tickers, rules, scan_tf, prov_scan)
            st.dataframe(pd.DataFrame(hits) if hits else pd.DataFrame({"result": ["No new alerts"]}))
        st.write(f"Telegram: {'configured' if TelegramNotifier().enabled else 'not configured (set env vars)'}")
        mins = st.number_input("Scheduled scan every (minutes, min 5)", 5, 1440, max(5, SETTINGS.scan_interval_minutes))
        if "sched" not in st.session_state:
            st.session_state.sched = None
        c1, c2 = st.columns(2)
        if c1.button("Start scheduled scan"):
            st.session_state.sched = Scheduler(st.session_state.manager, lambda: tickers, lambda: rules, scan_tf, int(mins))
            st.session_state.sched.start()
        if c2.button("Stop") and st.session_state.sched:
            st.session_state.sched.stop()
        if st.session_state.sched:
            sch = st.session_state.sched
            st.write(f"Scheduler running: {sch.running}")
            if sch.results:
                st.dataframe(pd.DataFrame(sch.results))
