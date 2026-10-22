"""Company historical analyser page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data.market_data import DataError
from history.events import EventConfig
from history.explain import LABELS_AR
from history.loader import load_price_history
from history.report import build_report, history_figure, to_html
from history.sec import SecClient
from ui import theme


@st.cache_data(ttl=3600, show_spinner=False)
def _load(ticker: str):
    return load_price_history(ticker)


def render() -> None:
    theme.hero("المحلل التاريخي للشركات", "تاريخ السهم كاملاً، الأحداث الكبرى، وأسباب محتملة مبنية على إفصاحات SEC الرسمية.")
    sec = SecClient()
    if not sec.enabled:
        st.info("لتفعيل إفصاحات SEC والقوائم المالية: أضف في ملف .env السطر  SEC_USER_AGENT=اسمك بريدك@example.com  "
                "(شرط من SEC). بدونه يعمل التحليل السعري فقط.")
    c1, c2, c3 = st.columns([1, 1, 2])
    ticker = c1.text_input("رمز السهم", "AAPL").strip().upper()
    compare = c2.text_input("مقارنة مع (اختياري)", "").strip().upper()
    use_focus = c3.checkbox("حصر الفترة الزمنية")
    focus = None
    if use_focus:
        d1, d2 = c3.columns(2)
        focus = (str(d1.date_input("من", pd.Timestamp("2020-01-01"))), str(d2.date_input("إلى", pd.Timestamp.today())))
    with st.expander("تعريفات الأحداث (قابلة للتعديل)"):
        e1, e2, e3, e4 = st.columns(4)
        cfg = EventConfig(surge_1d_pct=e1.number_input("حركة يومية ≥ %", 3.0, 50.0, 8.0),
                          move_5d_pct=e2.number_input("حركة 5 أيام ≥ %", 5.0, 80.0, 20.0),
                          gap_pct=e3.number_input("فجوة ≥ %", 2.0, 30.0, 5.0),
                          drawdown_pct=e4.number_input("تراجع من القمة ≥ %", 10.0, 80.0, 20.0))
    if not st.button("📊 حلّل", type="primary"):
        st.caption("أدخل الرمز واضغط «حلّل». تحميل التاريخ الكامل قد يستغرق ثوانٍ.")
        theme.disclaimer()
        return
    try:
        with st.spinner("جارٍ تحميل التاريخ الكامل والإفصاحات..."):
            ph, bench = _load(ticker), _load("SPY")
            cmp_ph = _load(compare) if compare else None
            rep = build_report(ph, bench, sec, cfg, focus, cmp_ph)
    except DataError as exc:
        st.error(f"تعذر إكمال التحليل: {exc}")
        return
    p = rep.profile
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("بداية البيانات", f"{p['data_start']:%Y-%m-%d}")
    m2.metric("أيام التداول", f"{p['bars']:,}")
    m3.metric("أقصى تراجع", f"{rep.risk_stats['max_drawdown_%']:.0f}%")
    m4.metric("التراجع الحالي", f"{rep.risk_stats['current_drawdown_from_high_%']:.0f}%")
    st.caption(f"{p['name'] or ''} · {p['exchange'] or ''} · {p['currency'] or ''} · تاريخ التأسيس: غير متاح من المصدر (ولا يُخلط مع تاريخ الإدراج). "
               "الأسعار معدّلة للتجزئات (Close) وللتوزيعات (AdjClose).")
    for n in rep.notes:
        st.warning(n) if "unavailable" in n or "disabled" in n or "not covered" in n else st.caption(n)
    st.plotly_chart(history_figure(rep), use_container_width=True)
    t1, t2, t3, t4, t5 = st.tabs(["الأداء", "الأحداث وأسبابها", "التحليل المالي", "المخاطر والأنماط", "المصادر والتصدير"])
    with t1:
        st.dataframe(rep.performance)
        st.write(rep.risk_stats)
    with t2:
        ev = rep.events[~rep.events["split_artifact"]] if not rep.events.empty else rep.events
        kinds = st.multiselect("نوع الحدث", sorted(ev["type"].unique()) if not ev.empty else [],
                               [k for k in ["surge", "plunge", "drawdown"] if not ev.empty and k in set(ev["type"])])
        shown = ev[ev["type"].isin(kinds)] if kinds else ev
        st.dataframe(shown)
        for i, x in rep.explanations.items():
            row = rep.events[rep.events["id"] == i].iloc[0]
            if kinds and row["type"] not in kinds:
                continue
            with st.expander(f"#{i} {row['type']} · {row['start']:%Y-%m-%d} → {row['end']:%Y-%m-%d} · {row['change_pct']:+.1f}% · {LABELS_AR[x.classification]}"):
                st.write(x.summary)
                for e in x.evidence:
                    st.markdown(f"- **{e.form}** {', '.join(e.items)} · {'; '.join(e.meaning)} · قُبل {e.accepted_utc:%Y-%m-%d %H:%M} UTC · "
                                f"التوقيت: `{e.timing}` · [المصدر]({e.url})")
                for c in x.clues:
                    st.caption("قرينة: " + c)
                for a in x.alternatives:
                    st.caption("تفسير بديل: " + a)
                st.caption(" · ".join(x.caveats))
    with t3:
        if rep.financials.empty:
            st.info("لا توجد بيانات مالية (تحتاج SEC_USER_AGENT، والشركة يجب أن تكون مسجّلة لدى SEC).")
        else:
            st.dataframe(rep.financials.tail(10).T)
            for f in rep.financial_flags:
                st.write("• " + f)
            st.caption("القيم بالدولار كما أُبلغت للـ SEC. السنة المالية تُسمّى بتاريخ نهايتها. لا تُحسب قيمة عادلة.")
    with t4:
        if rep.filing_risks:
            st.dataframe(pd.DataFrame(rep.filing_risks))
        else:
            st.write("لا توجد إفصاحات مخاطر مستخرجة (أو SEC غير مفعّل).")
        st.write(rep.patterns)
        st.write({"إجراءات مؤسسية": rep.corporate_actions})
    with t5:
        for s in rep.sources:
            st.markdown(f"- [{s['name']}]({s['url']}) · جُلب {s['retrieved']:%Y-%m-%d %H:%M} UTC")
        st.caption(f"آخر تحديث للتقرير: {rep.generated_at:%Y-%m-%d %H:%M} UTC")
        st.download_button("⬇️ تنزيل التقرير HTML", to_html(rep).encode("utf-8"), file_name=f"{ticker}_report.html",
                           mime="text/html")
        st.caption("لـ PDF: افتح ملف HTML في المتصفح ثم اطبع واختر «حفظ كـ PDF» (المخطط يحتاج اتصالاً بالإنترنت).")
    theme.disclaimer()
