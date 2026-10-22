"""Settings (thresholds, weights), data-source status, Telegram test, connectivity and recent log."""
from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from data.market_data import DataError
from data.us_calendar import is_early_close, market_status
from scanner.config import DEFAULT_WEIGHTS, ScannerConfig
from scanner.universe import CACHE
from ui import jobs, logbuffer, state
from ui.components import note
from ui.i18n import tr


def _connectivity(job: jobs.Job, provider, sec) -> dict:
    """Background thread: no st.* access, objects are passed in."""
    out = {}
    t0 = time.time()
    try:
        res = provider.bars(["SPY"], 2).get("SPY")
        if isinstance(res, Exception):
            raise res
        last = res.index[-1]
        out["yfinance 1m"] = f"OK: last bar {last}, {len(res)} rows, {time.time() - t0:.1f}s"
    except Exception as exc:
        out["yfinance 1m"] = f"FAILED: {type(exc).__name__}: {exc}"
    if not sec.enabled:
        out["SEC EDGAR"] = "disabled (set SEC_USER_AGENT in .env)"
    else:
        try:
            cik, name = sec.cik_for("AAPL")
            out["SEC EDGAR"] = f"OK: AAPL -> CIK {cik} ({name})"
        except Exception as exc:
            out["SEC EDGAR"] = f"FAILED: {type(exc).__name__}: {exc}"
    return out


def render(lang: str) -> None:
    st.title(tr("nav_settings", lang))
    cfg = state.cfg()
    t1, t2 = st.tabs(["حدود الماسح والأوزان" if lang == "ar" else "Scanner thresholds & weights",
                      "الحالة والاتصال والسجل" if lang == "ar" else "Status, connections & log"])
    with t1:
        with st.form("cfg_form"):
            st.subheader("الفلترة الأولية" if lang == "ar" else "Pre-filters")
            a, b, c, d = st.columns(4)
            min_p = a.number_input("Min price $", 0.1, 1000.0, float(cfg.min_price))
            max_p = b.number_input("Max price $", 0.5, 5000.0, float(cfg.max_price))
            dv = c.number_input("Min avg $ volume / session", 0.0, 1e10, float(cfg.min_avg_dollar_volume), 500_000.0)
            sp = d.number_input("Max spread %", 0.01, 5.0, float(cfg.max_spread_pct))
            st.subheader("الإشارات" if lang == "ar" else "Signals")
            e, f, g, h = st.columns(4)
            look = e.selectbox("Breakout lookback (min)", [15, 30], index=0 if cfg.breakout_lookback <= 15 else 1)
            early_rv = f.number_input("Early RVOL ≥", 1.0, 20.0, float(cfg.early_min_rvol), 0.5)
            conf_rv = g.number_input("Confirmed RVOL ≥", 1.0, 20.0, float(cfg.confirm_min_rvol), 0.5)
            ext = h.number_input("Extended beyond level %", 0.5, 20.0, float(cfg.max_extension_pct), 0.5)
            i, j, k, l = st.columns(4)
            conf_sc = i.number_input("Confirmed score ≥", 10.0, 100.0, float(cfg.confirm_min_score), 5.0)
            early_sc = j.number_input("Early score ≥", 10.0, 100.0, float(cfg.early_min_score), 5.0)
            expiry = k.number_input("Signal expiry (min)", 5, 240, int(cfg.expiry_minutes))
            cool = l.number_input("Alert cooldown (min)", 1, 240, int(cfg.cooldown_minutes))
            st.subheader("أوزان التقييم (نقطة بداية وليست نموذجاً مثبتاً)" if lang == "ar" else "Score weights (a starting point, not a proven model)")
            wcols = st.columns(6)
            weights = {k_: wcols[n].number_input(k_, 0, 100, int(cfg.weights.get(k_, DEFAULT_WEIGHTS[k_]))) for n, k_ in enumerate(DEFAULT_WEIGHTS)}
            st.caption(f"Σ = {sum(weights.values())}")
            ok = st.form_submit_button(tr("save", lang), type="primary")
        if ok:
            new = ScannerConfig(**{**asdict(cfg), "min_price": min_p, "max_price": max_p, "min_avg_dollar_volume": dv,
                                   "max_spread_pct": sp, "breakout_lookback": int(look), "early_min_rvol": early_rv,
                                   "confirm_min_rvol": conf_rv, "max_extension_pct": ext, "confirm_min_score": conf_sc,
                                   "early_min_score": early_sc, "expiry_minutes": int(expiry), "cooldown_minutes": int(cool),
                                   "weights": weights})
            try:
                new.validate().save()
                st.session_state["scanner_cfg"] = new
                st.success(tr("saved", lang))
            except ValueError as exc:
                st.error(str(exc))
        if st.button(tr("reset_defaults", lang)):
            st.session_state["scanner_cfg"] = ScannerConfig()
            ScannerConfig().save()
            st.rerun()
        st.caption("لا تضبط الأوزان على فترة الاختبار النهائية: عدّل على جلسات التطوير فقط." if lang == "ar"
                   else "Do not tune weights on the hold-out period: tune on development sessions only.")
    with t2:
        now = datetime.now(timezone.utc)
        st.write(f"{'حالة السوق الآن' if lang == 'ar' else 'Market now'}: **{market_status(now)}**"
                 + (" (إغلاق مبكر اليوم)" if is_early_close(now.date()) else ""))
        n = state.notifier()
        st.write("Telegram: " + tr("telegram_on" if n.enabled else "telegram_off", lang))
        if n.enabled and st.button("أرسل رسالة اختبار" if lang == "ar" else "Send test message"):
            ok = n.send("اختبار: Telegram يعمل. أداة تعليمية، ليست نصيحة مالية." if lang == "ar" else "Test message from the stock analyst.")
            st.success("تم الإرسال" if ok and lang == "ar" else ("Sent" if ok else "Failed (see log)"))
        sec = state.sec_client()
        st.write("SEC: " + (("مفعّل" if lang == "ar" else "enabled") if sec.enabled else tr("sec_off", lang)))
        st.write(f"{'قائمة الأسهم الرسمية' if lang == 'ar' else 'Official symbol list'}: "
                 + (f"cached ({Path(CACHE).stat().st_size // 1024} KB)" if Path(CACHE).exists() else ("غير محمّلة بعد" if lang == "ar" else "not downloaded yet")))
        st.caption(state.provider().description)
        if st.button("اختبر الاتصال" if lang == "ar" else "Test connections"):
            prov_, sec_ = state.provider(), state.sec_client()
            jobs.start("connectivity", lambda j: _connectivity(j, prov_, sec_))

        @st.fragment(run_every=2 if (jobs.get("connectivity") and jobs.get("connectivity").running) else None)
        def conn():
            job = jobs.get("connectivity")
            if job is None:
                return
            if job.running:
                st.progress(0.5, text=tr("loading", lang))
            elif job.result:
                for k_, v in job.result.items():
                    (st.success if v.startswith("OK") else st.warning)(f"{k_}: {v}")

        conn()
        st.subheader("سجل التشغيل" if lang == "ar" else "Recent log")
        lvl = st.selectbox("Level", ["INFO", "WARNING", "ERROR"], 1)
        recs = logbuffer.records(lvl)
        if recs:
            df = pd.DataFrame(recs)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            st.dataframe(df[["time", "level", "logger", "message"]].head(100), use_container_width=True, hide_index=True)
        else:
            st.caption("لا سجلات عند هذا المستوى." if lang == "ar" else "No records at this level.")
