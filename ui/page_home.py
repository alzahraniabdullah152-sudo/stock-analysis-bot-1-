"""Home dashboard."""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from data.us_calendar import NY, market_status, next_trading_day, session_hours
from scanner.states import LABELS
from ui import common, theme


def render(go_to) -> None:
    theme.hero("بوت تحليل الأسهم الأمريكية", "كشف مبكر للأسهم النشطة، وتحليل تاريخي مع مصادر، واختبار صارم. تعليمي وليس نصيحة مالية.")
    now = datetime.now(timezone.utc)
    rep = st.session_state.get("last_scan")
    theme.status_bar(market_status(now), common.provider().name, True, rep.scanned_at if rep else None)
    s = market_status(now)
    if s != "open":
        nxt = session_hours(next_trading_day(now.astimezone(NY).date()))
        st.caption(f"افتتاح الجلسة القادمة: {nxt[0]:%Y-%m-%d %H:%M} ET")
    st.write("")
    c1, c2, c3 = st.columns(3)
    for col, title, text, page in ((c1, "🔎 الماسح اللحظي", "ابحث عن أسهم بحجم غير معتاد واقتراب من اختراق.", "scanner"),
                                   (c2, "📚 تاريخ الشركة", "أحداث الصعود والهبوط وأسبابها بمصادر رسمية.", "company"),
                                   (c3, "📈 التحليل الفني", "شارت ومؤشرات وحاسبة مخاطر لسهم واحد.", "analysis")):
        with col:
            st.markdown(f'<div class="card"><h4>{title}</h4><div class="muted">{text}</div></div>', unsafe_allow_html=True)
            st.button("فتح", key=f"open_{page}", on_click=go_to, args=(page,))
    st.subheader("آخر الإشارات المسجّلة")
    sig = common.journal().signals(8, ("early", "near_breakout", "confirmed", "failed"))
    if sig.empty:
        st.info("لا توجد إشارات بعد. افتح الماسح واضغط «افحص الآن» أثناء ساعات السوق.")
    else:
        sig["الحالة"] = sig["state"].map(lambda x: LABELS[x][1])
        st.dataframe(sig[["ticker", "الحالة", "score", "ref_price", "signal_time"]])
    if rep and rep.candidates:
        st.subheader("أفضل نتائج آخر فحص")
        for c in rep.candidates[:3]:
            st.markdown(theme.candidate_card(c), unsafe_allow_html=True)
    theme.disclaimer()
