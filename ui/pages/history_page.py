"""Company history page: coverage, performance, events with evidence classes, financials, risks, export."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data.market_data import DataError
from history.events import EventConfig
from history.explain import LABELS_AR
from history.loader import load_price_history
from history.report import CompanyReport, build_report, history_figure, to_html
from ui import jobs, state
from ui.components import empty_state, html_block, note
from ui.i18n import tr

CLASS_COLORS = {"direct_evidence": ("#14653F", "#D6F0E2"), "probable": ("#0F6B6B", "#D8ECEA"),
                "coincident": ("#8A5A00", "#FBEFD5"), "unresolved": ("#5B6B78", "#EEF1F3")}
CLASS_EN = {"direct_evidence": "Direct evidence", "probable": "Probable", "coincident": "Coincident only", "unresolved": "Unresolved"}


def _build(job: jobs.Job, sec, ticker: str, bench: str, compare: str, focus, cfg: EventConfig) -> CompanyReport:
    """Background thread: `sec` is passed in; no st.* access here."""
    job.message = "prices"
    ph = load_price_history(ticker)
    bh = None
    if bench:
        job.message = "benchmark"
        try:
            bh = load_price_history(bench)
        except DataError:
            bh = None
    ch = None
    if compare and compare != ticker:
        job.message = "comparison"
        try:
            ch = load_price_history(compare)
        except DataError:
            ch = None
    job.message = "events and sources"
    rep = build_report(ph, bh, sec, cfg, focus, ch)
    if bench and bh is None:
        rep.notes.append(f"Benchmark {bench} could not be loaded: market comparison is missing.")
    return rep


def _badge(cls: str, lang: str) -> str:
    fg, bg = CLASS_COLORS[cls]
    return f"<span class='badge' style='color:{fg};background:{bg}'>{LABELS_AR[cls] if lang == 'ar' else CLASS_EN[cls]}</span>"


def _show(rep: CompanyReport, lang: str) -> None:
    p = rep.profile
    st.header(f"{rep.ticker} {('- ' + p['name']) if p['name'] else ''}")
    st.caption(f"{'التغطية الفعلية' if lang == 'ar' else 'Actual coverage'}: {p['data_start']:%Y-%m-%d} → {p['data_end']:%Y-%m-%d} "
               f"({p['bars']:,} {'يوم تداول' if lang == 'ar' else 'trading days'}) | "
               f"{'تاريخ التأسيس غير متاح من المصادر وهو غير تاريخ الإدراج' if lang == 'ar' else 'Founding date is not provided and differs from the listing date'}")
    for n in rep.notes:
        note(n)
    r = rep.risk_stats
    c = st.columns(4)
    c[0].metric("Max drawdown", f"{r['max_drawdown_%']:.1f}%")
    c[1].metric("From all-time high", f"{r['current_drawdown_from_high_%']:.1f}%")
    c[2].metric("Annualised vol", f"{r['annualised_volatility_%']:.1f}%")
    c[3].metric("CAGR", "n/a" if pd.isna(r["cagr_%"]) else f"{r['cagr_%']:.1f}%")
    log_scale = st.checkbox("مقياس لوغاريتمي" if lang == "ar" else "Log scale", True)
    try:
        st.plotly_chart(history_figure(rep, log_scale), use_container_width=True)
    except Exception as exc:
        st.warning(f"Chart unavailable: {type(exc).__name__}")
    st.subheader("الأداء" if lang == "ar" else "Performance")
    st.dataframe(rep.performance, use_container_width=True, hide_index=True)

    st.subheader("الأحداث السعرية" if lang == "ar" else "Price events")
    if rep.events.empty:
        empty_state("لا أحداث ضمن الحدود الحالية" if lang == "ar" else "No events under the current thresholds",
                    "خفّف حدود الكشف من «إعدادات الكشف» أعلاه." if lang == "ar" else "Relax the detection thresholds above.")
    else:
        kinds = sorted(rep.events["type"].unique())
        sel = st.multiselect("النوع" if lang == "ar" else "Type", kinds, [k for k in kinds if k in ("surge", "plunge", "drawdown", "gap_up", "gap_down")] or kinds)
        ev = rep.events[rep.events["type"].isin(sel)].copy()
        ev["start"], ev["end"] = ev["start"].dt.date, ev["end"].dt.date
        cols = [c for c in ["id", "type", "start", "end", "trading_days", "change_pct", "volume_mult", "volume_peak_mult", "market_ret_pct", "excess_pct",
                            "market_driven", "split_artifact", "max_drawdown_pct", "days_to_recover"] if c in ev.columns]
        st.dataframe(ev[cols], use_container_width=True, hide_index=True)

        st.subheader("أسباب الأحداث ومصادرها" if lang == "ar" else "Why events happened (with sources)")
        if not rep.explanations:
            st.caption("لا توجد أحداث مفسَّرة.")
        for i, x in rep.explanations.items():
            row = rep.events[rep.events["id"] == i]
            if row.empty or row.iloc[0]["type"] not in sel:
                continue
            row = row.iloc[0]
            with st.expander(f"#{i} {row['type']}  {row['start']:%Y-%m-%d} → {row['end']:%Y-%m-%d}  ({row['change_pct']:+.1f}%)"):
                html_block(_badge(x.classification, lang) + f" &nbsp; confidence: <b>{x.confidence}</b>")
                st.write(x.summary)
                for e in x.evidence[:8]:
                    when = f"{e.accepted_utc:%Y-%m-%d %H:%M} UTC" if e.accepted_utc else "time unknown"
                    st.markdown(f"- **{e.form}** {', '.join(e.items)} {'; '.join(e.meaning)} · {when} · timing vs move: `{e.timing}` · [source]({e.url})")
                for cl in x.clues:
                    st.caption("• " + cl)
                for a in x.alternatives:
                    st.caption("↔ " + a)
                for cv in x.caveats:
                    st.caption("ⓘ " + cv)

    if not rep.financials.empty:
        st.subheader("التحليل المالي (SEC، كما أُبلغ)" if lang == "ar" else "Financials (SEC, as reported)")
        keep = ["revenue", "net_income", "gross_margin_%", "operating_margin_%", "net_margin_%", "free_cash_flow", "debt", "cash",
                "current_ratio", "shares_wtd", "revenue_growth_%", "share_count_change_%"]
        show = rep.financials[[c for c in keep if c in rep.financials.columns]].tail(10).copy()
        show.index = show.index.date
        st.dataframe(show, use_container_width=True)
        for f in rep.financial_flags:
            st.caption("• " + f)
        st.caption("USD، فترات سنوية بتاريخ نهاية السنة المالية. لا قيمة عادلة مستنتجة." if lang == "ar"
                   else "USD, annual periods labelled by fiscal year end. No fair value is inferred.")
    if rep.filing_risks:
        st.subheader("مخاطر من إفصاحات SEC" if lang == "ar" else "Risk items in SEC filings")
        st.dataframe(pd.DataFrame(rep.filing_risks), use_container_width=True, hide_index=True,
                     column_config={"url": st.column_config.LinkColumn("filing")})
    ca = rep.corporate_actions
    st.subheader("الإجراءات المؤسسية" if lang == "ar" else "Corporate actions")
    st.write(", ".join(f"{s['date']:%Y-%m-%d} ({s['ratio']:g}:1)" for s in ca["splits"]) or ("لا تجزئات في البيانات" if lang == "ar" else "No splits in the data"))
    st.caption(f"{'دفعات التوزيعات' if lang == 'ar' else 'Dividend payments'}: {ca['dividend_payments']}")
    st.subheader("أنماط تاريخية (وصفية)" if lang == "ar" else "Historical patterns (descriptive)")
    st.json({k: v for k, v in rep.patterns.items() if k != "notes"})
    for n in rep.patterns.get("notes", []):
        st.caption(n)
    st.subheader("المصادر" if lang == "ar" else "Sources")
    for s in rep.sources:
        st.markdown(f"- [{s['name']}]({s['url']}) · {s['retrieved']:%Y-%m-%d %H:%M} UTC")
    st.caption(f"{'آخر تحديث للتقرير' if lang == 'ar' else 'Report updated'}: {rep.generated_at:%Y-%m-%d %H:%M} UTC")
    st.download_button(tr("download_html", lang), to_html(rep), file_name=f"{rep.ticker}_history.html", mime="text/html")
    st.caption("للحصول على PDF: افتح الملف في المتصفح ثم اطبع واختر «حفظ كـ PDF»." if lang == "ar" else "For PDF: open the file in a browser and print to PDF.")


def render(lang: str) -> None:
    st.title(tr("nav_history", lang))
    if not state.sec_client().enabled:
        note(tr("sec_off", lang))
    a, b = st.columns([2, 1])
    ticker = a.text_input(tr("ticker", lang), st.session_state.get("hist_ticker", "AAPL")).strip().upper()
    bench = b.text_input("المؤشر" if lang == "ar" else "Benchmark", "SPY").strip().upper()
    compare = st.text_input(tr("compare_with", lang), "").strip().upper()
    with st.expander(tr("focus_period", lang)):
        use_focus = st.checkbox("فعّل" if lang == "ar" else "Enable", False)
        f1, f2 = st.columns(2)
        d0 = f1.date_input("من" if lang == "ar" else "From", pd.Timestamp("2020-01-01").date())
        d1 = f2.date_input("إلى" if lang == "ar" else "To", pd.Timestamp.today().date())
    with st.expander("إعدادات الكشف" if lang == "ar" else "Detection settings"):
        x1, x2, x3 = st.columns(3)
        surge = x1.number_input("حركة يومية ≥ %" if lang == "ar" else "1-day move ≥ %", 3.0, 50.0, 8.0)
        gap = x2.number_input("فجوة ≥ %" if lang == "ar" else "Gap ≥ %", 2.0, 30.0, 5.0)
        dd = x3.number_input("تراجع ≥ %" if lang == "ar" else "Drawdown ≥ %", 10.0, 80.0, 20.0)
    if st.button(tr("build_report", lang), type="primary") and ticker:
        st.session_state["hist_ticker"] = ticker
        cfg = EventConfig(surge_1d_pct=surge, gap_pct=gap, drawdown_pct=dd)
        focus = (str(d0), str(d1)) if use_focus else None
        sec = state.sec_client()
        jobs.start("history", lambda j: _build(j, sec, ticker, bench, compare, focus, cfg))

    @st.fragment(run_every=2 if (jobs.get("history") and jobs.get("history").running) else None)
    def view():
        job = jobs.get("history")
        if job is None:
            empty_state("لم يُنشأ تقرير بعد" if lang == "ar" else "No report yet",
                        "اكتب رمز السهم واضغط «أنشئ التقرير». أول مرة قد تستغرق نصف دقيقة." if lang == "ar"
                        else "Enter a ticker and press Build report. The first run can take ~30 seconds.")
        elif job.running:
            st.progress(0.4, text=f"{tr('loading', lang)} ({job.message}, {job.seconds:.0f}s)")
        elif job.status == "error":
            st.error(f"{tr('data_error', lang)}: {job.error}")
        elif job.result is not None:
            _show(job.result, lang)

    view()
