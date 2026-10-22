"""Live scanner page: universe -> two-stage filter -> score -> cards -> details, catalysts, history link."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alerts.alert_manager import load_watchlist
from data.market_data import DataError
from data.us_calendar import NY
from history.link import similar_history
from history.loader import load_price_history
from scanner.catalysts import fetch_news
from scanner.engine import Scanner
from scanner.features import build_features
from scanner.states import LABELS, RANK
from scanner.universe import clean_universe, load_universe, parse_ticker_text
from ui import common, theme


@st.cache_data(ttl=3600, show_spinner=False)
def _history(ticker: str):
    return load_price_history(ticker)


def _stage1(tickers: list[str], cfg, max_n: int) -> tuple[list[str], dict[str, str]]:
    """Cheap daily screen (price range, 20-day dollar volume) before any 1-minute download."""
    prov = common.provider()
    if not hasattr(prov, "daily_summary"):
        return tickers[:max_n], {}
    df = prov.daily_summary(tickers)
    dropped = {}
    keep = df[(df["close"].between(cfg.min_price, cfg.max_price)) & (df["avg_dollar_volume_20d"] >= cfg.min_avg_dollar_volume)]
    for t in set(tickers) - set(keep["ticker"]):
        dropped[t] = "stage-1: price/liquidity filter or no daily data"
    keep = keep.sort_values("avg_dollar_volume_20d", ascending=False).head(max_n)
    return keep["ticker"].tolist(), dropped


def run_scan(tickers: list[str], send_alerts: bool, two_stage: bool, max_n: int):
    cfg = common.cfg()
    bar = st.progress(0.0, text="بدء الفحص...")
    dropped: dict[str, str] = {}
    if two_stage:
        bar.progress(0.02, text="المرحلة 1: فلترة يومية للسعر والسيولة...")
        tickers, dropped = _stage1(tickers, cfg, max_n)
    sc = Scanner(common.provider(), cfg, common.journal(), common.notifier() if send_alerts else None,
                 names=st.session_state.get("names", {}))
    rep = sc.scan(tickers, send_alerts=send_alerts, progress=lambda f, t: bar.progress(min(1.0, f), text=t))
    rep.skipped.update(dropped)
    bar.empty()
    st.session_state.last_scan = rep
    return rep


def _chart(ticker: str, cand):
    bars = common.provider().bars([ticker], common.cfg().history_days).get(ticker)
    if bars is None or isinstance(bars, Exception):
        st.info(f"تعذر تحميل شارت الدقيقة: {bars}")
        return
    f = build_features(bars, None, common.cfg(), drop_incomplete=True)
    today = f[f["session"] == f["session"].iloc[-1]]
    fig = go.Figure(go.Candlestick(x=today.index, open=today["Open"], high=today["High"], low=today["Low"],
                                   close=today["Close"], name=ticker, increasing_line_color="#16a34a",
                                   decreasing_line_color="#dc2626"))
    fig.add_trace(go.Scatter(x=today.index, y=today["vwap"], name="VWAP", line=dict(color="#7c3aed", width=1.4)))
    if cand.level:
        fig.add_hline(y=cand.level, line_dash="dash", line_color="#2563eb", annotation_text="مستوى الاختراق")
    if cand.invalidation:
        fig.add_hline(y=cand.invalidation, line_dash="dot", line_color="#dc2626", annotation_text="إبطال")
    fig.update_layout(height=430, margin=dict(l=8, r=8, t=24, b=8), template="plotly_white",
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def _detail(rep):
    names = [c.ticker for c in rep.candidates]
    if not names:
        return
    pick = st.selectbox("تفاصيل الإشارة", names, key="scan_pick")
    c = next(x for x in rep.candidates if x.ticker == pick)
    a, b = st.columns([3, 2])
    with a:
        _chart(pick, c)
    with b:
        st.markdown(f"**{pick}** · {LABELS[c.state][1]} · تقييم {c.score:.0f}/100")
        comp = pd.DataFrame([{"المكوّن": k, "النقاط": v["points"], "الحد الأعلى": v["max"]} for k, v in c.components.items()])
        fig = go.Figure(go.Bar(x=comp["النقاط"], y=comp["المكوّن"], orientation="h", marker_color="#2563eb"))
        fig.update_layout(height=250, margin=dict(l=8, r=8, t=8, b=8), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        for r in c.reasons:
            st.markdown(f"{ {'+': '✅', '-': '➖', '!': '⚠️'}[r.kind] } {r.ar}")
        if c.invalidation:
            st.caption(f"إبطال السيناريو: إغلاق دقيقة تحت {c.invalidation:.2f} · تنتهي الإشارة: "
                       f"{c.expires_at.astimezone(NY):%H:%M} ET" if c.expires_at else "")
        st.caption(f"آخر شمعة مكتملة قبل {c.data_age_s / 60:.0f} دقيقة · المصدر {c.provider} (متأخر)")
    n1, n2 = st.columns(2)
    if n1.button("🗞 المحفزات والأخبار", key="btn_news"):
        with st.spinner("جارٍ البحث..."):
            news = fetch_news(pick)
        if not news:
            st.info("المحفز غير معروف: لم يُعثر على أخبار حديثة من المصدر المتاح. لا تُفترض أسباب غير موثقة.")
        for n in news[:6]:
            tag = "🔻" if n["negative"] else "•"
            st.markdown(f"{tag} [{n['title']}]({n['url']}) · {n['source']} · نُشر {n['published']:%m-%d %H:%M} UTC · {n['category']}")
    if n2.button("📚 هل تكرر نمط مشابه تاريخياً؟", key="btn_hist"):
        with st.spinner("جارٍ تحليل التاريخ..."):
            try:
                ph = _history(pick)
                bench = _history("SPY")
                res = similar_history(ph.df, bench.df)
            except DataError as exc:
                st.error(f"تعذر تحميل التاريخ: {exc}")
                return
        st.write(f"عدد الحالات السابقة المشابهة (اختراق فوق أعلى 20 يوماً بحجم ≥ 2x): **{res['n_cases']}**")
        if res["stats"]:
            st.json(res["stats"])
        elif res["n_cases"]:
            st.dataframe(res["cases"].tail(10))
        for n in res["notes"]:
            st.caption(n)


def _show(rep):
    for w in rep.warnings:
        st.warning(common.warn_text(w))
    cnt = pd.Series([c.state for c in rep.candidates]).value_counts().to_dict()
    cols = st.columns(5)
    for col, s in zip(cols, ["confirmed", "near_breakout", "early", "watch", "failed"]):
        col.metric(LABELS[s][1], cnt.get(s, 0))
    shown = [c for c in rep.candidates if c.state != "none"][:9]
    if not shown:
        st.info("لا توجد فرص حسب القواعد الحالية. هذا طبيعي في أغلب الأوقات: الماسح لا يخترع إشارات.")
    grid = st.columns(3)
    for i, c in enumerate(shown):
        with grid[i % 3]:
            st.markdown(theme.candidate_card(c), unsafe_allow_html=True)
            if c.alert_sent:
                st.caption("📨 أُرسل تنبيه Telegram")
    with st.expander("الجدول الكامل"):
        t = rep.table()
        st.dataframe(t if not t.empty else pd.DataFrame({"result": ["لا توجد نتائج"]}))
    if rep.skipped:
        with st.expander(f"أسهم مستبعدة أو بلا بيانات ({len(rep.skipped)})"):
            st.dataframe(pd.DataFrame({"ticker": list(rep.skipped), "reason": list(rep.skipped.values())}))
    _detail(rep)


def render() -> None:
    theme.hero("الماسح اللحظي", "يرصد الأسهم التي يظهر عليها حجم غير معتاد وزخم واقتراب من اختراق، ويشرح سبب كل درجة.")
    cfg = common.cfg()
    rep = st.session_state.get("last_scan")
    theme.status_bar(rep.market_status if rep else "closed", common.provider().name, True,
                     rep.scanned_at if rep else None)
    src = st.radio("مصدر الأسهم", ["قائمتي", "سوق ناسداك/NYSE (فلترة على مرحلتين)"], horizontal=True)
    two_stage, max_n = False, 60
    if src == "قائمتي":
        txt = st.text_area("الرموز (مفصولة بفواصل أو أسطر)", ", ".join(load_watchlist()), height=80)
        tickers = parse_ticker_text(txt)
    else:
        c1, c2 = st.columns([1, 1])
        max_n = c2.slider("أقصى عدد أسهم للفحص الدقيق", 20, 150, 60, 10,
                          help="Yahoo يحدّ الطلبات. كلما زاد العدد زاد الوقت واحتمال الحظر المؤقت.")
        if c1.button("تحميل قائمة السوق"):
            try:
                with st.spinner("جارٍ تحميل قائمة الرموز من Nasdaq Trader..."):
                    uni = load_universe(refresh=False)
                st.session_state.universe = uni
                st.session_state.names = dict(zip(uni["symbol"], uni["name"]))
            except DataError as exc:
                st.error(f"{exc}")
        uni = st.session_state.get("universe")
        tickers = uni["symbol"].tolist() if uni is not None else []
        st.caption(f"{len(tickers):,} رمزاً محمّلاً. القائمة تضم الأسهم المدرجة حالياً فقط (المشطوبة غير موجودة)." if tickers
                   else "اضغط «تحميل قائمة السوق» أولاً.")
        two_stage = True
    notif = common.notifier()
    send = st.checkbox("إرسال تنبيهات Telegram للإشارات الجديدة", False, disabled=not notif.enabled,
                       help=None if notif.enabled else "Telegram غير مضبوط: راجع صفحة الإعدادات.")
    c1, c2 = st.columns([1, 2])
    auto = c2.checkbox("تحديث تلقائي", False, help="يعيد الفحص تلقائياً طالما الصفحة مفتوحة.")
    mins = c2.slider("كل (دقائق)", 2, 30, 5) if auto else 5
    go_btn = c1.button("🔎 افحص الآن", type="primary", disabled=not tickers)
    if go_btn or (auto and tickers):
        def job():
            try:
                _show(run_scan(tickers, send, two_stage, max_n))
            except DataError as exc:
                st.error(f"خطأ في البيانات: {exc}")
        if auto:
            st.fragment(run_every=f"{mins}m")(job)()
        else:
            job()
    elif rep:
        st.caption("آخر نتيجة محفوظة في هذه الجلسة:")
        _show(rep)
    theme.disclaimer()
