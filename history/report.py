"""Company history report: coverage, performance, events with evidence classes, financials, risks, HTML export."""
from __future__ import annotations

import html
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data.market_data import DataError
from history.events import EventConfig, detect_events
from history.explain import LABELS_AR, Explanation, explain_event
from history.financials import annual_table, flags
from history.loader import PriceHistory
from history.sec import RISK_ITEMS, Filing, SecClient, SecError

log = logging.getLogger(__name__)
HORIZONS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "10Y": 2520}
EXPLAIN_TYPES = ("surge", "plunge", "gap_up", "gap_down", "drawdown", "reversal_up", "ath_break")


@dataclass
class CompanyReport:
    ticker: str
    profile: dict
    performance: pd.DataFrame
    risk_stats: dict
    events: pd.DataFrame
    explanations: dict[int, Explanation]
    corporate_actions: dict
    financials: pd.DataFrame
    financial_flags: list[str]
    filing_risks: list[dict]
    patterns: dict
    sources: list[dict]
    notes: list[str]
    price_df: pd.DataFrame
    compare: pd.DataFrame | None = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def performance_table(df: pd.DataFrame, bench: pd.DataFrame | None, since: str | None = None) -> pd.DataFrame:
    """Total return (dividend-adjusted close) over standard horizons; horizons longer than the data are omitted."""
    px = df["AdjClose"]
    rows = []
    for name, n in HORIZONS.items():
        if len(px) <= n:
            continue
        r = (px.iloc[-1] / px.iloc[-1 - n] - 1) * 100
        b = np.nan
        if bench is not None and len(bench) > n:
            bp = bench["AdjClose"].reindex(px.index, method="ffill")
            if bp.iloc[-1 - n] == bp.iloc[-1 - n]:
                b = (bp.iloc[-1] / bp.iloc[-1 - n] - 1) * 100
        rows.append({"horizon": name, "return_%": r, "benchmark_%": b, "excess_%": r - b if b == b else np.nan})
    rows.append({"horizon": f"since {px.index[0]:%Y-%m-%d}", "return_%": (px.iloc[-1] / px.iloc[0] - 1) * 100,
                 "benchmark_%": np.nan, "excess_%": np.nan})
    return pd.DataFrame(rows)


def risk_stats(df: pd.DataFrame) -> dict:
    c = df["Close"]
    ret = c.pct_change().dropna()
    peak = c.cummax()
    dd = c / peak - 1
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
    adj = df["AdjClose"]
    return {"annualised_volatility_%": float(ret.std() * np.sqrt(252) * 100),
            "max_drawdown_%": float(dd.min() * 100), "current_drawdown_from_high_%": float(dd.iloc[-1] * 100),
            "all_time_high_close": float(c.max()), "all_time_high_date": c.idxmax(),
            "best_day_%": float(ret.max() * 100), "best_day": ret.idxmax(),
            "worst_day_%": float(ret.min() * 100), "worst_day": ret.idxmin(),
            "cagr_%": float(((adj.iloc[-1] / adj.iloc[0]) ** (1 / years) - 1) * 100) if years >= 1 else np.nan,
            "years_covered": float(years)}


def pattern_summary(events: pd.DataFrame, min_n: int = 10) -> dict:
    """Descriptive counts and medians. Statistics are withheld for small samples."""
    out: dict = {"by_type": events["type"].value_counts().to_dict() if not events.empty else {}, "notes": []}
    if events.empty:
        return out
    dd = events[(events["type"] == "drawdown") & ~events["split_artifact"]]
    if len(dd) >= 3:
        out["drawdown_median_pct"] = float(dd["max_drawdown_pct"].median())
        out["drawdown_median_days_to_trough"] = float(dd["days_to_trough"].median())
        rec = dd["days_to_recover"].dropna()
        out["drawdown_recovered"] = f"{len(rec)}/{len(dd)}"
        if len(rec) >= 3:
            out["drawdown_median_days_to_recover"] = float(rec.median())
    for kind in ("surge", "plunge"):
        g = events[(events["type"] == kind) & ~events["split_artifact"]]
        if 0 < len(g) < min_n:
            out["notes"].append(f"Only {len(g)} '{kind}' events: too few for statistics, listed individually.")
        elif len(g) >= min_n:
            out[f"{kind}_median_change_pct"] = float(g["change_pct"].median())
            out[f"{kind}_market_driven_share"] = float(g["market_driven"].mean())
    out["notes"].append("Descriptive history only. A repeated past pattern does not imply it will repeat.")
    return out


def filing_risks(filings: list[Filing], years: int = 5) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=365 * years)
    out = []
    for f in filings:
        hits = [i for i in f.items if i in RISK_ITEMS]
        if hits and f.accepted_utc and f.accepted_utc >= cutoff:
            out.append({"date": f.filed, "form": f.form, "items": ", ".join(hits),
                        "risk": "; ".join(RISK_ITEMS[i] for i in hits), "url": f.url})
    return out[::-1]


def build_report(ph: PriceHistory, bench: PriceHistory | None = None, sec: SecClient | None = None,
                 cfg: EventConfig | None = None, focus: tuple[str, str] | None = None,
                 compare: PriceHistory | None = None, max_explain: int = 25) -> CompanyReport:
    df = ph.df
    notes = list(ph.notes)
    events = detect_events(df, bench.df if bench else None, cfg)
    if focus:
        lo, hi = pd.Timestamp(focus[0]), pd.Timestamp(focus[1])
        events = events[(events["end"] >= lo) & (events["start"] <= hi)].reset_index(drop=True)
        notes.append(f"Events filtered to {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}.")
    sec_ok, filings, cik, sec_name = False, [], None, None
    sources = [{"name": ph.source, "url": "https://finance.yahoo.com/quote/" + ph.ticker, "retrieved": ph.fetched_at}]
    if sec is not None and sec.enabled:
        try:
            cik, sec_name = sec.cik_for(ph.ticker)
            filings = sec.filings(cik)
            sec_ok = True
            sources.append({"name": f"SEC EDGAR filings (CIK {cik})", "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}",
                            "retrieved": datetime.now(timezone.utc)})
        except SecError as exc:
            notes.append(f"SEC data unavailable: {exc}")
    elif sec is not None:
        notes.append("SEC features disabled: set SEC_USER_AGENT in .env (your name and email) to enable filings and financials.")
    fin, fin_flags = pd.DataFrame(), []
    if sec_ok and cik is not None:
        try:
            fin = annual_table(sec.company_facts(cik))
            fin_flags = flags(fin)
            sources.append({"name": "SEC XBRL company facts", "url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                            "retrieved": datetime.now(timezone.utc)})
        except SecError as exc:
            notes.append(f"Financial statements unavailable: {exc}")
    expl: dict[int, Explanation] = {}
    cand = events[events["type"].isin(EXPLAIN_TYPES)].copy()
    cand["abs"] = cand["change_pct"].abs()
    for _, ev in cand.sort_values("abs", ascending=False).head(max_explain).iterrows():
        expl[int(ev["id"])] = explain_event(ev, filings, sec_available=sec_ok)
    splits = ph.splits
    actions = {"splits": [{"date": d, "ratio": float(r)} for d, r in splits.items()],
               "dividend_payments": int(len(ph.dividends)),
               "dividends_total_per_share": float(ph.dividends.sum()) if len(ph.dividends) else 0.0}
    profile = {"ticker": ph.ticker, "name": ph.name or sec_name, "exchange": ph.exchange, "currency": ph.currency,
               "data_start": df.index[0], "data_end": df.index[-1], "bars": len(df),
               "provider_first_trade_date": ph.first_trade_date,
               "company_founding_date": "not provided by the sources (and distinct from the listing date)",
               "sec_registrant": sec_name, "sec_cik": cik}
    cmp_df = None
    if compare is not None:
        a = df["AdjClose"]
        b = compare.df["AdjClose"].reindex(a.index, method="ffill")
        common = pd.concat([a, b], axis=1, keys=[ph.ticker, compare.ticker]).dropna()
        if len(common) > 2:
            cmp_df = common / common.iloc[0] * 100
            notes.append(f"Comparison rebased to 100 at {common.index[0]:%Y-%m-%d} (common coverage with {compare.ticker}).")
    return CompanyReport(ph.ticker, profile, performance_table(df, bench.df if bench else None), risk_stats(df), events,
                         expl, actions, fin, fin_flags, filing_risks(filings) if sec_ok else [],
                         pattern_summary(events), sources, notes, df, cmp_df)


# ------------------------------------------------------------------------------------ chart + HTML
def history_figure(rep: CompanyReport, log_scale: bool = True):
    import plotly.graph_objects as go

    df = rep.price_df
    fig = go.Figure(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=rep.ticker, line=dict(width=1.6, color="#2563eb")))
    if rep.compare is not None:
        for col in rep.compare.columns[1:]:
            fig.add_trace(go.Scatter(x=rep.compare.index, y=rep.compare[col] / 100 * float(df.loc[rep.compare.index[0], "Close"]),
                                     mode="lines", name=col + " (rebased)", line=dict(width=1.2, dash="dot")))
    ev = rep.events[rep.events["type"].isin(["surge", "plunge", "gap_up", "gap_down", "ath_break"]) & ~rep.events["split_artifact"]]
    colors = {"surge": "#16a34a", "gap_up": "#16a34a", "ath_break": "#0ea5e9", "plunge": "#dc2626", "gap_down": "#dc2626"}
    for kind, g in ev.groupby("type"):
        fig.add_trace(go.Scatter(x=g["end"], y=df["Close"].reindex(g["end"]).to_numpy(), mode="markers", name=kind,
                                 marker=dict(size=9, color=colors.get(kind, "#6b7280"), line=dict(width=1, color="white")),
                                 text=[f"#{i}: {c:+.1f}%" for i, c in zip(g["id"], g["change_pct"])], hoverinfo="text+x"))
    dd = rep.events[(rep.events["type"] == "drawdown") & ~rep.events["split_artifact"]]
    for r in dd.itertuples():
        fig.add_vrect(x0=r.peak_date, x1=r.trough_date, fillcolor="#dc2626", opacity=0.08, line_width=0)
    for s in rep.corporate_actions["splits"]:
        fig.add_vline(x=s["date"], line_dash="dot", line_color="#6b7280")
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10), yaxis_type="log" if log_scale else "linear",
                      legend=dict(orientation="h"), template="plotly_white", title=f"{rep.ticker}: split-adjusted close")
    return fig


def _tbl(df: pd.DataFrame, **kw) -> str:
    return df.to_html(index=False, escape=True, border=0, classes="t", na_rep="–", float_format=lambda x: f"{x:,.2f}", **kw)


def to_html(rep: CompanyReport, include_plotlyjs: str = "cdn") -> str:
    """Self-contained RTL HTML (Arabic headings, English technical terms). 'cdn' needs internet to draw the chart."""
    e = html.escape
    p = rep.profile
    parts = [f"<!doctype html><html lang='ar' dir='rtl'><head><meta charset='utf-8'><title>{e(rep.ticker)} - تقرير تاريخي</title>"
             "<style>body{font-family:Segoe UI,Tahoma,Arial,sans-serif;max-width:1050px;margin:24px auto;padding:0 16px;color:#0f172a;line-height:1.7}"
             "h1,h2{color:#1e3a8a}table.t{border-collapse:collapse;width:100%;font-size:13px;direction:ltr;margin:8px 0}"
             "table.t td,table.t th{border-bottom:1px solid #e2e8f0;padding:5px 8px;text-align:right}table.t th{background:#f1f5f9}"
             ".box{background:#fff7ed;border:1px solid #fed7aa;padding:10px 14px;border-radius:8px}.small{color:#64748b;font-size:12px}"
             ".tag{display:inline-block;padding:1px 8px;border-radius:10px;background:#e0e7ff;font-size:12px}</style></head><body>"]
    parts.append(f"<h1>{e(rep.ticker)} {('- ' + e(str(p['name']))) if p['name'] else ''}</h1>")
    parts.append("<p class='box'>أداة تعليمية وبحثية، وليست نصيحة مالية. التفسيرات السببية مبنية على وثائق رسمية وتوقيتها فقط، "
                 "وليست إثباتاً للسببية.</p>")
    parts.append("<h2>1. ملخص تنفيذي</h2><ul>")
    r = rep.risk_stats
    parts.append(f"<li>التغطية الفعلية للبيانات: {p['data_start']:%Y-%m-%d} إلى {p['data_end']:%Y-%m-%d} ({p['bars']:,} يوم تداول، {r['years_covered']:.1f} سنة).</li>")
    parts.append(f"<li>أقصى تراجع من القمة: {r['max_drawdown_%']:.1f}% | التراجع الحالي: {r['current_drawdown_from_high_%']:.1f}% | التذبذب السنوي: {r['annualised_volatility_%']:.1f}%.</li>")
    n_ev = 0 if rep.events.empty else int((~rep.events['split_artifact']).sum())
    parts.append(f"<li>عدد الأحداث السعرية المكتشفة: {n_ev}. الأحداث المفسَّرة بدليل مباشر: "
                 f"{sum(1 for x in rep.explanations.values() if x.classification == 'direct_evidence')}.</li></ul>")
    parts.append("<h2>2. تعريف الشركة والإدراج</h2>")
    parts.append(f"<p>البورصة: {e(str(p['exchange'] or 'غير متاح'))} | العملة: {e(str(p['currency'] or 'غير متاح'))} | "
                 f"أول تداول حسب المزود: {e(str(p['provider_first_trade_date'].date() if p['provider_first_trade_date'] is not None else 'غير متاح'))}<br>"
                 f"تاريخ التأسيس: {e(p['company_founding_date'])}<br>مسجَّلة لدى SEC: {e(str(p['sec_registrant'] or 'غير متاح'))}</p>")
    try:
        parts.append(str(history_figure(rep).to_html(full_html=False, include_plotlyjs=include_plotlyjs)))
    except Exception as exc:  # plotly missing or figure error: report keeps working
        parts.append(f"<p class='small'>تعذر رسم المخطط ({e(type(exc).__name__)}).</p>")
    parts.append("<h2>3. الأداء</h2>" + _tbl(rep.performance))
    if not rep.events.empty:
        show = rep.events[~rep.events["split_artifact"]][["id", "type", "start", "end", "change_pct", "volume_mult", "volume_peak_mult", "market_ret_pct", "excess_pct", "market_driven"]]
        parts.append("<h2>4. جدول الأحداث السعرية</h2>" + _tbl(show.assign(start=show["start"].dt.date, end=show["end"].dt.date)))
    parts.append("<h2>5. تحليل الأسباب ومصادرها</h2>")
    for i, x in rep.explanations.items():
        row = rep.events[rep.events["id"] == i].iloc[0]
        parts.append(f"<h3>#{i} {e(row['type'])} {row['start']:%Y-%m-%d} → {row['end']:%Y-%m-%d} ({row['change_pct']:+.1f}%)</h3>"
                     f"<p><span class='tag'>{e(LABELS_AR[x.classification])}</span> ثقة: {e(x.confidence)}<br>{e(x.summary)}</p>")
        if x.evidence:
            parts.append("<ul>" + "".join(
                f"<li>{e(f.form)} {e(', '.join(f.items))} ({e('; '.join(f.meaning))}) - قُبل {f.accepted_utc:%Y-%m-%d %H:%M} UTC - "
                f"التوقيت: {e(f.timing)} - <a href='{e(f.url)}'>المصدر</a></li>" for f in x.evidence[:6]) + "</ul>")
        for c in x.clues:
            parts.append(f"<p class='small'>قرينة: {e(c)}</p>")
        for a in x.alternatives:
            parts.append(f"<p class='small'>تفسير بديل: {e(a)}</p>")
    if not rep.financials.empty:
        keep = ["revenue", "net_income", "gross_margin_%", "operating_margin_%", "net_margin_%", "free_cash_flow", "debt",
                "cash", "current_ratio", "shares_wtd", "revenue_growth_%"]
        fin = rep.financials[[c for c in keep if c in rep.financials.columns]].reset_index()
        fin["fiscal_year_end"] = fin["fiscal_year_end"].dt.date
        parts.append("<h2>6. التحليل المالي (SEC XBRL، بالدولار كما أُبلغ، الفترات السنوية بتاريخ نهاية السنة المالية)</h2>" + _tbl(fin.tail(10)))
        parts.append("<ul>" + "".join(f"<li>{e(x)}</li>" for x in rep.financial_flags) + "</ul>")
    parts.append("<h2>7. الإجراءات المؤسسية</h2>")
    ca = rep.corporate_actions
    parts.append("<p>تجزئات الأسهم: " + (", ".join(f"{s['date']:%Y-%m-%d} ({s['ratio']:g}:1)" for s in ca["splits"]) or "لا يوجد في البيانات")
                 + f"<br>دفعات التوزيعات: {ca['dividend_payments']}.</p>")
    if rep.filing_risks:
        parts.append("<h2>8. مخاطر مستخرجة من إفصاحات SEC (آخر 5 سنوات)</h2>" + _tbl(pd.DataFrame(rep.filing_risks).drop(columns="url")))
    parts.append("<h2>9. أنماط تاريخية (وصفية فقط)</h2><ul>" + "".join(f"<li>{e(str(k))}: {e(str(v))}</li>" for k, v in rep.patterns.items() if k != "notes")
                 + "".join(f"<li>{e(n)}</li>" for n in rep.patterns.get("notes", [])) + "</ul>")
    parts.append("<h2>10. المصادر وتغطية البيانات</h2><ul>" + "".join(
        f"<li><a href='{e(s['url'])}'>{e(s['name'])}</a> - جُلب {s['retrieved']:%Y-%m-%d %H:%M} UTC</li>" for s in rep.sources) + "</ul>")
    parts.append("<ul>" + "".join(f"<li class='small'>{e(n)}</li>" for n in rep.notes) + "</ul>")
    parts.append(f"<p class='small'>آخر تحديث للتقرير: {rep.generated_at:%Y-%m-%d %H:%M} UTC. للحصول على PDF: افتح الملف في المتصفح ثم اطبع واختر «حفظ كـ PDF».</p></body></html>")
    return "".join(parts)
