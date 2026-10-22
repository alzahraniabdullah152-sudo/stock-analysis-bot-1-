"""Settings, data-source status, Telegram test and run log. Secrets are never displayed."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from config import LOG_FILE
from data.market_data import DataError
from history.sec import SecClient
from scanner.config import DEFAULT_WEIGHTS, ScannerConfig
from ui import common, theme

WEIGHT_AR = {"rvol": "الحجم النسبي", "momentum": "الزخم والتسارع", "breakout": "الاختراق", "liquidity": "السيولة والسبريد",
             "relstrength": "القوة النسبية", "continuation": "الاستمرارية"}


def _flag(ok: bool) -> str:
    return "✅ مضبوط" if ok else "⚠️ غير مضبوط"


def render() -> None:
    theme.hero("الإعدادات والحالة", "حدود الفحص والأوزان قابلة للتعديل. هي نقطة بداية وليست قيماً مثبتة علمياً.")
    t1, t2, t3 = st.tabs(["حدود الفحص والأوزان", "المصادر والاتصال", "سجل التشغيل"])
    cfg = common.cfg()
    with t1:
        c1, c2, c3 = st.columns(3)
        cfg.min_price = c1.number_input("أدنى سعر ($)", 0.5, 1000.0, cfg.min_price)
        cfg.max_price = c2.number_input("أعلى سعر ($)", 1.0, 5000.0, cfg.max_price)
        cfg.min_avg_dollar_volume = c3.number_input("أدنى متوسط قيمة تداول يومي ($)", 0.0, 1e10, cfg.min_avg_dollar_volume, 500_000.0)
        c1, c2, c3 = st.columns(3)
        cfg.max_spread_pct = c1.number_input("أقصى سبريد (%)", 0.01, 5.0, cfg.max_spread_pct)
        cfg.breakout_lookback = int(c2.selectbox("نافذة الاختراق (دقائق)", [15, 30, 45, 60], [15, 30, 45, 60].index(cfg.breakout_lookback) if cfg.breakout_lookback in [15, 30, 45, 60] else 0))
        cfg.min_prior_sessions = int(c3.number_input("أقل عدد جلسات سابقة", 2, 7, cfg.min_prior_sessions))
        c1, c2, c3 = st.columns(3)
        cfg.confirm_min_rvol = c1.number_input("أدنى حجم نسبي للاختراق المؤكد", 1.0, 20.0, cfg.confirm_min_rvol)
        cfg.confirm_min_score = c2.number_input("أدنى درجة للاختراق المؤكد", 0.0, 100.0, cfg.confirm_min_score)
        cfg.expiry_minutes = int(c3.number_input("صلاحية الإشارة (دقائق)", 5, 240, cfg.expiry_minutes))
        st.markdown("**أوزان التقييم** (المجموع المرجعي 100)")
        cols = st.columns(3)
        for i, k in enumerate(DEFAULT_WEIGHTS):
            cfg.weights[k] = cols[i % 3].slider(WEIGHT_AR[k], 0, 50, min(50, int(cfg.weights.get(k, DEFAULT_WEIGHTS[k]))))
        total = sum(cfg.weights.values())
        st.caption(f"مجموع الأوزان الحالي: {total}. الدرجة تُحسب دائماً من 100 مهما كان المجموع. "
                   "لا تضبط الأوزان على نتائج الاختبار النهائي؛ استخدم الجلسات الأقدم للتطوير فقط.")
        b1, b2 = st.columns(2)
        if b1.button("💾 حفظ الإعدادات", type="primary"):
            try:
                cfg.validate().save()
                st.success("تم الحفظ في scanner_settings.json")
            except ValueError as exc:
                st.error(str(exc))
        if b2.button("↩️ استعادة الافتراضي"):
            st.session_state.scan_cfg = ScannerConfig()
            st.rerun()
    with t2:
        notif = common.notifier()
        sec = SecClient()
        rows = {"Telegram": _flag(notif.enabled), "SEC (SEC_USER_AGENT)": _flag(sec.enabled),
                "Alpha Vantage": _flag(bool(os.getenv("ALPHAVANTAGE_API_KEY")))}
        for k, v in rows.items():
            st.write(f"**{k}**: {v}")
        st.caption("القيم السرّية لا تُعرض. تُضبط في ملف .env فقط ولا تُرفع إلى Git.")
        c1, c2 = st.columns(2)
        if c1.button("اختبر مصدر البيانات (SPY)"):
            try:
                with st.spinner("..."):
                    r = common.provider().bars(["SPY"], 2)["SPY"]
                if isinstance(r, Exception):
                    st.error(f"فشل: {r}")
                else:
                    st.success(f"نجح: {len(r)} شمعة، آخر شمعة {r.index[-1]}. المصدر: {common.provider().description}")
            except DataError as exc:
                st.error(str(exc))
        if c2.button("أرسل رسالة اختبار Telegram", disabled=not notif.enabled):
            st.success("أُرسلت") if notif.send("اختبار: بوت تحليل الأسهم يعمل. أداة تعليمية وليست نصيحة مالية.") else st.error("فشل الإرسال (راجع السجل).")
        st.info("المزود الحالي (yfinance) غير رسمي ومتأخر. للبيانات اللحظية الحقيقية والبث المباشر يلزم مزود مدفوع، "
                "ويُضاف عبر واجهة MinuteProvider دون تغيير بقية النظام.")
    with t3:
        p = Path(LOG_FILE)
        if p.exists():
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
            st.code("\n".join(lines) or "(فارغ)", language="text")
        else:
            st.info("لا يوجد سجل بعد.")
    theme.disclaimer()
