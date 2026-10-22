"""Home: what is happening now (session, last scan, journal) and where to go next."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from data.us_calendar import NY, is_early_close, is_trading_day, market_status, next_trading_day, session_hours
from ui import state
from ui.components import candidate_card, empty_state, note, status_strip
from ui.i18n import tr


def _go(page: str) -> None:
    st.session_state["nav"] = page


def render(lang: str) -> None:
    now = datetime.now(timezone.utc)
    ms = market_status(now)
    rep = st.session_state.get("last_report")
    status_strip(ms, rep.provider if rep else state.provider().name, True,
                 max((c.bar_time for c in rep.candidates), default=None) if rep else None, lang)
    st.title(tr("app_title", lang))
    st.write("اكتشف الأسهم المرشحة لحركة سريعة، واقرأ لماذا تحرّك أي سهم عبر التاريخ، وتحقق من القواعد قبل أن تثق بها."
             if lang == "ar" else "Find stocks setting up for fast moves, read why any stock moved in the past, and test the rules before trusting them.")
    d = now.astimezone(NY).date()
    if is_trading_day(d):
        o, c = session_hours(d)
        st.caption(f"{'جلسة اليوم' if lang == 'ar' else 'Today'}: {o:%H:%M}-{c:%H:%M} ET" + (" (إغلاق مبكر)" if is_early_close(d) and lang == "ar" else (" (early close)" if is_early_close(d) else "")))
    else:
        nd = next_trading_day(d)
        st.caption(f"{'السوق مغلق اليوم. الجلسة القادمة' if lang == 'ar' else 'Closed today. Next session'}: {nd:%A %Y-%m-%d}")

    c1, c2, c3 = st.columns(3)
    c1.button("📡 " + tr("nav_scanner", lang), on_click=_go, args=("scanner",), use_container_width=True)
    c2.button("🏛 " + tr("nav_history", lang), on_click=_go, args=("history",), use_container_width=True)
    c3.button("📈 " + tr("nav_analysis", lang), on_click=_go, args=("analysis",), use_container_width=True)

    st.subheader("آخر فحص" if lang == "ar" else "Latest scan")
    shown = [c for c in (rep.candidates if rep else []) if c.state != "none"][:5]
    if shown:
        for c in shown:
            candidate_card(c, lang)
    else:
        empty_state("لا نتائج فحص في هذه الجلسة" if lang == "ar" else "No scan results this session",
                    "افتح «الماسح اللحظي» وابدأ فحصاً." if lang == "ar" else "Open the live scanner and run a scan.")
    st.subheader("آخر الإشارات المسجّلة" if lang == "ar" else "Latest logged signals")
    sig = state.journal().signals(limit=8)
    if sig.empty:
        st.caption("السجل فارغ." if lang == "ar" else "The journal is empty.")
    else:
        st.dataframe(sig[["ticker", "signal_time", "state", "score", "ref_price"]], use_container_width=True, hide_index=True)
    note(tr("disclaimer", lang))
