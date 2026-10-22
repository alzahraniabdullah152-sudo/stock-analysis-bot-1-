"""Signal journal, outcome tracking, strict replay backtest and calibration."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from alerts.alert_manager import load_watchlist
from data.market_data import DataError
from scanner.backtest import ReplayConfig, run_replay, wilson_interval
from scanner.calibration import calibration_report
from scanner.states import LABELS
from scanner.universe import parse_ticker_text
from ui import common, theme


def _outcome_summary(out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tgt, h), g in out.groupby(["target_pct", "horizon_min"]):
        g = g[g["status"].isin(["hit", "miss", "ambiguous"])]
        n, k = len(g), int((g["status"] == "hit").sum())
        lo, hi = wilson_interval(k, n)
        rows.append({"الهدف": f"+{tgt}% / {h}د", "إشارات محسومة": n, "حققت الهدف": k, "غامضة": int((g['status'] == 'ambiguous').sum()),
                     "نسبة النجاح": k / n if n else float("nan"), "نطاق 95%": f"{lo:.0%}–{hi:.0%}" if n else "–",
                     "ملاحظة": "عينة صغيرة" if n < 30 else ""})
    return pd.DataFrame(rows)


def render() -> None:
    theme.hero("السجل والاختبار والمعايرة", "كل إشارة تُسجَّل مع نتيجتها، حتى الفاشلة. الاختبار التاريخي صارم ولا يعرض احتمالات غير مدعومة.")
    t1, t2, t3 = st.tabs(["سجل الإشارات", "اختبار تاريخي للماسح", "معايرة النموذج"])
    j = common.journal()
    with t1:
        sig = j.signals(500)
        if sig.empty:
            st.info("لا توجد إشارات مسجّلة بعد. شغّل الماسح في أوقات السوق وستظهر هنا.")
        else:
            states = st.multiselect("الحالة", list(LABELS), ["early", "near_breakout", "confirmed", "failed"])
            view = sig[sig["state"].isin(states)] if states else sig
            st.dataframe(view[["id", "ticker", "signal_time", "state", "score", "ref_price", "level", "invalidation", "alerted", "provider"]],
                         )
            if st.button("🔄 حدّث نتائج الإشارات"):
                tick = sig[sig["state"].isin(["early", "near_breakout", "confirmed"])]["ticker"].unique().tolist()
                try:
                    with st.spinner("جارٍ تحميل الشموع اللاحقة وتقييم النتائج..."):
                        bars = {t: b for t, b in common.provider().bars(tick, 7).items() if not isinstance(b, Exception)}
                        n = j.evaluate_outcomes(bars, common.cfg())
                    st.success(f"تم تحديث {n} نتيجة.")
                except DataError as exc:
                    st.error(str(exc))
            out = j.outcomes()
            if not out.empty:
                st.subheader("نتائج الإشارات (افتراضية، بسعر الإشارة المرجعي دون انزلاق)")
                st.dataframe(_outcome_summary(out))
                st.caption("الإشارات الغامضة (الهدف والوقف في نفس الدقيقة) لا تُحسب نجاحاً. الشمعة بدقة الدقيقة لا تكشف الترتيب.")
    with t2:
        st.write("إعادة تشغيل القواعد على شموع الدقيقة المتاحة. التنفيذ على افتتاح الشمعة التالية مع انزلاق وعمولة، والتعارض يُحسب غامضاً.")
        txt = st.text_area("الأسهم", ", ".join(load_watchlist()), height=60, key="rp_t")
        c1, c2, c3 = st.columns(3)
        slip = c1.number_input("الانزلاق لكل جهة (%)", 0.0, 2.0, 0.10, 0.05)
        comm = c2.number_input("عمولة / سهم ($)", 0.0, 0.1, 0.0, 0.005)
        trig = c3.multiselect("حالات تُحتسب إشارة", ["early", "near_breakout", "confirmed"], ["confirmed"])
        if st.button("▶️ شغّل الاختبار", type="primary"):
            tickers = parse_ticker_text(txt)
            cfg = common.cfg()
            try:
                with st.spinner("جارٍ تحميل البيانات وإعادة التشغيل..."):
                    data = common.provider().bars(list(dict.fromkeys(tickers + [cfg.benchmark])), 7)
                    bench = data.get(cfg.benchmark)
                    bench = None if isinstance(bench, Exception) else bench
                    use = {t: b for t, b in data.items() if t in tickers and not isinstance(b, Exception)}
                    res = run_replay(use, bench, cfg, ReplayConfig(slippage_pct=slip, commission_per_share=comm,
                                                                    trigger_states=tuple(trig) or ("confirmed",), random_trials=100))
                st.session_state.replay = res
            except DataError as exc:
                st.error(str(exc))
        res = st.session_state.get("replay")
        if res:
            st.write(f"عدد الإشارات: **{len(res['signals'])}** · {res['split']}")
            if res["skipped"]:
                st.caption(f"مستبعد: {res['skipped']}")
            for key, title in (("overall", "الكل"), ("development", "التطوير (الجلسات الأقدم)"), ("holdout", "الاختبار النهائي (الجلسات الأحدث، لا تضبط عليه)"),
                               ("random_baseline", "خط أساس: اختيار عشوائي"), ("momentum_baseline", "خط أساس: زخم بسيط")):
                if isinstance(res[key], pd.DataFrame) and not res[key].empty:
                    st.markdown(f"**{title}**")
                    st.dataframe(res[key])
            for n in res["notes"]:
                st.caption("• " + n)
            if len(res["signals"]) < 30:
                st.warning("عدد الإشارات أقل من 30: النتائج مجرد تجربة تشغيل وليست دليلاً على جودة القواعد.")
    with t3:
        res = st.session_state.get("replay")
        if not res or res["outcomes"].empty:
            st.info("شغّل الاختبار التاريخي أولاً. لا تُعرض أي احتمالات قبل وجود عينة كافية ومعايرة ناجحة على فترة مستقلة.")
        else:
            cfg = common.cfg()
            for tgt, h in cfg.targets:
                rep = calibration_report(res["outcomes"], tgt, h)
                st.markdown(f"**+{tgt}% خلال {h} دقيقة**")
                if rep["ok"]:
                    st.success(rep["reason"])
                    st.dataframe(rep["table"])
                    st.caption(f"Brier: النموذج {rep['brier_model']:.3f} مقابل المعدل الثابت {rep['brier_base_rate']:.3f}")
                else:
                    st.warning("لا يُعرض احتمال: " + rep["reason"])
    theme.disclaimer()
