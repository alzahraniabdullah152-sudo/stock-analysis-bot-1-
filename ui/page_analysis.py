"""Technical analysis page (the original dashboard: chart, risk calculator, backtest, watchlist alerts).

Logic is unchanged from the first version; only the layout moved from the sidebar into this page.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from alerts.alert_manager import (RULE_KINDS, AlertManager, AlertRule, Scheduler, TelegramNotifier,
                                  load_watchlist, save_watchlist)
from analysis.report import analyze, scenario_text
from analysis.risk_management import RiskInputError, position_size
from analysis.trend_analysis import analyze_trend, mtf_agreement
from backtest.backtester import BacktestConfig, run_in_out_of_sample
from charts.interactive_chart import OVERLAYS, build_chart
from config import INTERVALS, INTRADAY, SETTINGS
from data.market_data import DataError, fetch
from indicators.technical_indicators import add_all
from ui import theme


@st.cache_data(ttl=120, show_spinner=False)
def cached_fetch(ticker: str, tf: str, provider: str, provisional: bool):
    """Cached for 2 minutes only; the data timestamp is always shown next to the chart."""
    return fetch(ticker, tf, provider=provider, include_incomplete=provisional)


def render() -> None:
    theme.hero("التحليل الفني", "شارت تفاعلي ومؤشرات واتجاه وإشارات مبنية على قواعد، مع حاسبة مخاطر واختبار تاريخي.")
    with st.expander("إعدادات التحليل", expanded=True):
        a1, a2, a3, a4 = st.columns([1, 1, 1, 1])
        ticker = a1.text_input("رمز السهم", "AAPL").strip().upper()
        tf = a2.selectbox("الإطار الزمني", list(INTERVALS), index=4)
        provider = a3.selectbox("مزود البيانات", ["yfinance", "alphavantage"],
                                index=0 if SETTINGS.data_provider == "yfinance" else 1)
        provisional = a4.checkbox("تضمين الشمعة غير المكتملة", False,
                                  help="الأصل: تُحلَّل الشموع المكتملة فقط.")
        b1, b2 = st.columns(2)
        overlays = b1.multiselect("المتوسطات على الشارت", list(OVERLAYS), ["SMA 50", "SMA 200", "EMA 21"])
        mtf_tfs = b2.multiselect("فحص أطر زمنية متعددة", [k for k in INTERVALS if k != tf],
                                 [k for k in ["1d", "1wk"] if k != tf])
        c1, c2, c3, c4 = st.columns(4)
        show_bb = c1.checkbox("Bollinger Bands", False)
        show_vwap = c2.checkbox("VWAP (لحظي)", tf in INTRADAY)
        show_sr = c3.checkbox("مناطق الدعم والمقاومة", True)
        show_markers = c4.checkbox("علامات الإشارات السابقة", True)

    tab_an, tab_risk, tab_bt, tab_watch = st.tabs(["التحليل", "حاسبة المخاطر", "الاختبار التاريخي", "المراقبة والتنبيهات"])
    report = None
    with tab_an:
        try:
            with st.spinner("جارٍ تحميل البيانات..."):
                md = cached_fetch(ticker, tf, provider, provisional)
                mtf_states = {}
                for t in mtf_tfs + [tf]:
                    try:
                        m = md if t == tf else cached_fetch(ticker, t, provider, False)
                        mtf_states[t] = analyze_trend(add_all(m.df, t in INTRADAY))
                    except DataError as exc:
                        st.info(f"{t}: غير متاح ({exc})")
                report = analyze(md, mtf_states)
        except DataError as exc:
            st.error(f"تعذر تحميل البيانات: {exc}")

        if report:
            md, d, tr = report.md, report.d, report.trend
            theme.status_bar(md.market_status, md.provider, True, md.fetched_at,   # free providers are never real-time
                             f" · آخر شمعة {md.last_bar_time} · {'مكتملة' if md.last_bar_complete else 'غير مكتملة (مؤقتة)'}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("آخر إغلاق", f"{report.last_price:.2f}")
            c2.metric("الاتجاه", tr.direction.title())
            c3.metric("الزخم", tr.momentum.title())
            sc = report.score["score"]
            c4.metric("درجة التوافق (توجيهية)", "n/a" if sc is None else f"{sc:.0f}/100")
            st.plotly_chart(build_chart(d, f"{md.ticker} {tf}", overlays, show_bb, show_vwap,
                                        report.zones if show_sr else None,
                                        report.markers if show_markers else None,
                                        report.setups[0] if report.setups else None, show_sr),
                            use_container_width=True)
            a, b = st.columns(2)
            with a:
                st.subheader("الاتجاه والهيكل")
                st.write(f"قصير / متوسط / طويل: **{report.horizons['short']} / {report.horizons['medium']} / "
                         f"{report.horizons['long']}**")
                st.write(f"Structure: **{tr.structure}** | ADX: {'n/a' if tr.adx is None else f'{tr.adx:.1f}'} | "
                         f"Volume confirms: {tr.volume_confirms}")
                st.write("; ".join(tr.reasons) or "تاريخ غير كافٍ")
                if len(mtf_states) > 1:
                    txt, s = mtf_agreement(mtf_states)
                    st.write(f"Multi-timeframe: **{txt}** (weighted agreement {s:+.2f}). الأطر الأعلى لها وزن أكبر.")
                st.write(f"أقرب دعم: {'none' if not report.support else f'{report.support.low:.2f}-{report.support.high:.2f}'}"
                         f" | مقاومة: {'none' if not report.resistance else f'{report.resistance.low:.2f}-{report.resistance.high:.2f}'}")
                st.write(f"مستوى الاختراق (أعلى 20 شمعة): {report.key_levels['breakout']:.2f} | "
                         f"الكسر: {report.key_levels['breakdown']:.2f}")
            with b:
                st.subheader("درجة التوافق (توجيهية)")
                st.json({"score": report.score["score"], "components(-1..1)": report.score["components"],
                         "weights": report.score["weights"]})
                st.caption("score = 50 + 50 * sum(w*c)/sum(w). المكونات الناقصة تُحذف. ليست احتمال ربح.")
            st.subheader(f"الإعدادات على آخر شمعة {'مكتملة' if md.last_bar_complete else 'مؤقتة'}")
            if not report.setups:
                st.write("لا توجد إشارة مبنية على القواعد في آخر شمعة.")
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
        st.subheader("حاسبة حجم المركز (افتراضية)")
        default = report.setups[0] if report and report.setups else None
        c1, c2, c3 = st.columns(3)
        eq = c1.number_input("رأس المال ($)", 100.0, 1e9, float(SETTINGS.default_equity))
        rp = c2.number_input("المخاطرة لكل صفقة (%)", 0.05, 5.0, float(min(SETTINGS.default_risk_pct, 5.0)), 0.05)
        maxp = c3.number_input("أقصى حجم مركز (% من رأس المال)", 1.0, 100.0, 100.0)
        e = c1.number_input("الدخول", 0.01, 1e6, float(default.entry if default else (report.last_price if report else 100.0)))
        sl = c2.number_input("وقف الخسارة", 0.01, 1e6, float(default.stop if default else (report.last_price * 0.97 if report else 97.0)))
        tg = c3.text_input("الأهداف (مفصولة بفواصل)", ", ".join(f"{t:.2f}" for t in default.targets) if default else "")
        com = c1.number_input("عمولة / سهم ($)", 0.0, 10.0, float(SETTINGS.commission_per_share))
        slp = c2.number_input("الانزلاق (%)", 0.0, 5.0, float(SETTINGS.slippage_pct))
        frac = c3.checkbox("السماح بأسهم كسرية", False)
        try:
            targets = [float(x) for x in tg.split(",") if x.strip()]
            p = position_size(eq, rp, e, sl, targets, com, slp, frac, maxp)
            st.write(f"الاتجاه **{p.side}** | الأسهم **{p.shares}** | سعر الدخول المقدّر {p.entry:.2f} | سعر الوقف المقدّر {p.stop:.2f}")
            st.write(f"المخاطرة بالدولار عند الوقف (مع التكاليف): **${p.dollar_risk:,.2f}** | قيمة المركز ${p.position_value:,.2f}")
            for t in p.targets:
                st.write(f"Target {t['target']:.2f}: " + (f"reward ${t['reward']:,.2f}, R {t['rr']:.2f}" if t["valid"] else "غير صالح لهذا الاتجاه"))
            for w in p.warnings:
                st.warning(w)
            st.caption("افتراضي. الفجوات وإيقاف التداول قد تسبب خسائر أكبر من مسافة الوقف.")
        except (RiskInputError, ValueError) as exc:
            st.error(str(exc))

    with tab_bt:
        st.subheader("اختبار تاريخي (داخل العينة مقابل خارجها)")
        split = st.slider("نسبة داخل العينة", 0.3, 0.9, 0.7, 0.05)
        if st.button("شغّل الاختبار") and report:
            res = run_in_out_of_sample(report.d, split, BacktestConfig(equity=SETTINGS.default_equity),
                                       intraday=tf in INTRADAY, ticker=ticker)
            for k, label in (("in_sample", "داخل العينة (للتطوير)"), ("out_of_sample", "خارج العينة (احكم من هنا)")):
                st.markdown(f"**{label}**")
                st.dataframe(pd.DataFrame([res[k]["metrics"]]).T.rename(columns={0: "value"}))
                st.line_chart(res[k]["equity_curve"])
                if not res[k]["trades"].empty:
                    st.dataframe(res[k]["trades"].tail(50))
            st.caption("عند لمس الشمعة للوقف والهدف يُفترض الوقف أولاً؛ الدخول على افتتاح الشمعة التالية مع انزلاق. "
                       "النتائج التاريخية لا تضمن الأداء المستقبلي.")

    with tab_watch:
        st.subheader("قائمة المراقبة")
        wl = st.text_input("الرموز (مفصولة بفواصل)", ", ".join(load_watchlist()))
        tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]
        if st.button("حفظ القائمة"):
            save_watchlist(tickers)
            st.success("تم الحفظ")
        kinds = st.multiselect("قواعد التنبيه", list(RULE_KINDS), ["ema_cross_up", "ema_cross_down", "breakout", "breakdown"])
        lvl = st.number_input("المستوى لقواعد السعر/RSI", 0.0, 1e6, 70.0)
        rules = [AlertRule(k, lvl if k.startswith(("price", "rsi")) else None) for k in kinds]
        scan_tf = st.selectbox("إطار الفحص", ["1d", "1h", "15m", "5m"], 0)
        prov_scan = st.checkbox("فحص الشمعة غير المكتملة (مؤقتة)", False)
        if "manager" not in st.session_state:
            st.session_state.manager = AlertManager()
        if st.button("افحص الآن"):
            with st.spinner("جارٍ الفحص..."):
                hits = st.session_state.manager.scan(tickers, rules, scan_tf, prov_scan)
            st.dataframe(pd.DataFrame(hits) if hits else pd.DataFrame({"result": ["لا تنبيهات جديدة"]}))
        st.write(f"Telegram: {'مفعّل' if TelegramNotifier().enabled else 'غير مضبوط (راجع صفحة الإعدادات)'}")
        mins = st.number_input("فحص مجدول كل (دقائق، الحد الأدنى 5)", 5, 1440, max(5, SETTINGS.scan_interval_minutes))
        if "sched" not in st.session_state:
            st.session_state.sched = None
        c1, c2 = st.columns(2)
        if c1.button("ابدأ الفحص المجدول"):
            st.session_state.sched = Scheduler(st.session_state.manager, lambda: tickers, lambda: rules, scan_tf, int(mins))
            st.session_state.sched.start()
        if c2.button("إيقاف") and st.session_state.sched:
            st.session_state.sched.stop()
        if st.session_state.sched:
            sch = st.session_state.sched
            st.write(f"الفحص المجدول يعمل: {sch.running}")
            if sch.results:
                st.dataframe(pd.DataFrame(sch.results))
