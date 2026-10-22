"""Signal journal, outcomes, calibration and rule replay."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from scanner.backtest import ReplayConfig, run_replay
from scanner.calibration import calibration_report
from scanner.config import ScannerConfig
from scanner.providers import FrameProvider
from scanner.universe import parse_ticker_text
from ui import jobs, state
from ui.components import empty_state, note
from ui.i18n import tr


def _outcome_frame(journal) -> pd.DataFrame:
    """signals joined with outcomes, with the columns calibration_report needs."""
    sig, out = journal.signals(limit=5000), journal.outcomes()
    if sig.empty or out.empty:
        return pd.DataFrame()
    df = out.merge(sig[["id", "ticker", "bar_time", "score", "state"]], left_on="signal_id", right_on="id")
    df["session"] = pd.to_datetime(df["bar_time"], utc=True).dt.tz_convert("America/New_York").dt.date
    return df


def _replay_job(job: jobs.Job, cfg: ScannerConfig, provider, tickers: list[str], days: int, trigger: tuple[str, ...],
                slip: float) -> dict:
    """Background thread: no st.* access, everything is passed in."""
    job.message = "download"
    data = provider.bars(tickers + [cfg.benchmark], days)
    bench = data.get(cfg.benchmark)
    bench = None if isinstance(bench, Exception) else bench
    frames = {t: d for t, d in data.items() if t != cfg.benchmark and not isinstance(d, Exception)}
    errs = {t: str(d) for t, d in data.items() if isinstance(d, Exception) and t != cfg.benchmark}
    job.message = "replay"
    res = run_replay(frames, bench, cfg, ReplayConfig(slippage_pct=slip, trigger_states=trigger))
    res["download_errors"] = errs
    return res


def _show_replay(res: dict, lang: str) -> None:
    n = len(res["signals"])
    st.metric("Signals", n)
    if n < 30:
        note("عدد الإشارات أقل من 30: النتائج أدناه اختبار تشغيل وليست دليلاً إحصائياً." if lang == "ar"
             else "Fewer than 30 signals: results below are a smoke test, not statistical evidence.")
    for k, lbl in (("overall", "الكل" if lang == "ar" else "Overall"), ("development", "تطوير" if lang == "ar" else "Development"),
                   ("holdout", "اختبار مستقل (الحكم هنا)" if lang == "ar" else "Hold-out (judge here)"),
                   ("momentum_baseline", "خط أساس: زخم بسيط" if lang == "ar" else "Baseline: simple momentum"),
                   ("random_baseline", "خط أساس: اختيار عشوائي" if lang == "ar" else "Baseline: random bars")):
        df = res.get(k)
        st.markdown(f"**{lbl}**")
        if df is None or df.empty:
            st.caption("-")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
    if res.get("split"):
        st.caption(str(res["split"]))
    for t, m in {**res.get("skipped", {}), **res.get("download_errors", {})}.items():
        st.caption(f"{t}: {m}")
    for nline in res["notes"]:
        st.caption("ⓘ " + nline)


def render(lang: str) -> None:
    st.title(tr("nav_journal", lang))
    journal = state.journal()
    t1, t2, t3, t4 = st.tabs(["سجل الإشارات" if lang == "ar" else "Signal log", "النتائج" if lang == "ar" else "Outcomes",
                              "المعايرة" if lang == "ar" else "Calibration", "اختبار القواعد" if lang == "ar" else "Rule replay"])
    with t1:
        sig = journal.signals(limit=1000)
        if sig.empty:
            empty_state("السجل فارغ" if lang == "ar" else "The journal is empty",
                        "تُسجَّل الإشارات تلقائياً عند تغيّر حالة السهم في الماسح." if lang == "ar" else "Signals are logged automatically when a ticker's state changes during a scan.")
        else:
            states = st.multiselect("الحالة" if lang == "ar" else "State", sorted(sig["state"].unique()), sorted(sig["state"].unique()))
            view = sig[sig["state"].isin(states)]
            st.dataframe(view[["id", "ticker", "signal_time", "state", "score", "coverage", "ref_price", "level", "invalidation",
                               "provider", "alerted"]], use_container_width=True, hide_index=True)
            pick = st.number_input("افتح إشارة (id)" if lang == "ar" else "Open signal (id)", 0, int(sig["id"].max()), 0)
            row = sig[sig["id"] == pick]
            if not row.empty:
                for r in json.loads(row.iloc[0]["reasons"] or "[]"):
                    st.write(("• " + r["ar"]) if lang == "ar" else ("• " + r["en"]))
    with t2:
        if st.button("حدّث النتائج" if lang == "ar" else "Update outcomes"):
            sig = journal.signals(limit=2000, states=("early", "near_breakout", "confirmed"))
            tickers = sorted(sig["ticker"].unique())[:60] if not sig.empty else []
            if not tickers:
                st.info("لا إشارات لتقييمها." if lang == "ar" else "No signals to evaluate.")
            else:
                with st.spinner(tr("loading", lang)):
                    data = state.provider().bars(tickers, 7)
                frames = {t: d for t, d in data.items() if not isinstance(d, Exception)}
                n = journal.evaluate_outcomes(frames, state.cfg())
                st.success(f"{n} {'نتيجة حُدّثت' if lang == 'ar' else 'outcomes updated'}")
        df = _outcome_frame(journal)
        if df.empty:
            empty_state("لا نتائج بعد" if lang == "ar" else "No outcomes yet", "")
        else:
            summ = df.groupby(["target_pct", "horizon_min", "status"]).size().unstack(fill_value=0)
            st.dataframe(summ, use_container_width=True)
            st.caption("«غامض» = الهدف والوقف في نفس الدقيقة؛ يُحسب غير ناجح. «معلّق» = لم تكتمل المدة بعد." if lang == "ar"
                       else "'ambiguous' = target and stop inside the same minute (counted as not a win). 'pending' = horizon not finished.")
    with t3:
        df = _outcome_frame(journal)
        tgt = st.selectbox("الهدف" if lang == "ar" else "Target", [tuple(t) for t in state.cfg().targets],
                           format_func=lambda x: f"+{x[0]}% / {x[1]}min")
        if df.empty:
            empty_state("لا بيانات كافية" if lang == "ar" else "Not enough data", "")
        else:
            rep = calibration_report(df, tgt[0], tgt[1])
            if rep["ok"]:
                st.success(rep["reason"])
                st.metric("Brier (model)", f"{rep['brier_model']:.4f}", delta=f"{rep['brier_model'] - rep['brier_base_rate']:+.4f} vs base rate", delta_color="inverse")
                st.dataframe(rep["table"], use_container_width=True, hide_index=True)
            else:
                note(("لا يُعرض أي احتمال: " if lang == "ar" else "No probability is shown: ") + rep["reason"])
            st.caption("الدرجة لا تعني احتمالاً حتى تثبت المعايرة على جلسات لاحقة مستقلة." if lang == "ar"
                       else "The score is not a probability until calibration is validated on later, independent sessions.")
    with t4:
        note("بيانات الدقيقة المجانية لا تغطي سوى أيام قليلة؛ هذا الاختبار للتحقق من سلامة القواعد وليس دليلاً على الربحية." if lang == "ar"
             else "Free 1-minute data covers only a few days: this checks the rules work, it is not evidence of profitability.")
        tickers = parse_ticker_text(st.text_area("الأسهم" if lang == "ar" else "Tickers", "AAPL, AMD, F, PLTR, SOFI, SNAP, BAC", height=70))
        c1, c2, c3 = st.columns(3)
        days = c1.slider("أيام" if lang == "ar" else "Days", 3, 7, 7)
        trig = c2.multiselect("الحالات المُطلِقة" if lang == "ar" else "Trigger states", ["early", "near_breakout", "confirmed"], ["confirmed"])
        slip = c3.number_input("انزلاق لكل جهة %" if lang == "ar" else "Slippage per side %", 0.0, 2.0, 0.10, 0.01)
        if st.button("شغّل الاختبار" if lang == "ar" else "Run replay", type="primary") and tickers and trig:
            cfg_, prov_ = state.cfg(), state.provider()
            jobs.start("replay", lambda j: _replay_job(j, cfg_, prov_, tickers, days, tuple(trig), slip))

        @st.fragment(run_every=2 if (jobs.get("replay") and jobs.get("replay").running) else None)
        def view():
            job = jobs.get("replay")
            if job is None:
                return
            if job.running:
                st.progress(0.5, text=f"{tr('loading', lang)} ({job.message}, {job.seconds:.0f}s)")
            elif job.status == "error":
                st.error(f"{tr('data_error', lang)}: {job.error}")
            elif job.result:
                _show_replay(job.result, lang)

        view()
