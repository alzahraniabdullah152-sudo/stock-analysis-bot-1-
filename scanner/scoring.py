"""Explainable 0-100 score. A heuristic ranking aid, NOT a probability of success.

score = 100 * sum(points) / sum(all weights). A component whose data is missing contributes 0 points but its
weight stays in the denominator, so missing data can only LOWER a score. `coverage` reports how much of the
weight is backed by real data; candidates below `cfg.min_coverage` are never actionable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scanner.config import ScannerConfig

# code -> (kind, English, Arabic). kind: "+" supports, "-" weakens, "!" data/risk warning
TEMPLATES: dict[str, tuple[str, str, str]] = {
    "rvol_high": ("+", "Volume {rvol:.1f}x higher than usual for this time of day (last few minutes)",
                  "حجم التداول {rvol:.1f} ضعف المعتاد في نفس الوقت (آخر دقائق)"),
    "rvol_normal": ("-", "Volume only {rvol:.1f}x of normal", "حجم التداول {rvol:.1f} من المعتاد فقط"),
    "rvol_missing": ("!", "Relative volume unavailable (needs enough prior sessions)",
                     "الحجم النسبي غير متاح (يحتاج جلسات سابقة كافية)"),
    "momentum_up": ("+", "Momentum: {ret5:+.2f}% in 5 min, accelerating {accel:+.2f}pp",
                    "زخم: {ret5:+.2f}% خلال 5 دقائق، والتسارع {accel:+.2f} نقطة"),
    "momentum_weak": ("-", "Weak momentum ({ret5:+.2f}% in 5 min)", "زخم ضعيف ({ret5:+.2f}% خلال 5 دقائق)"),
    "breakout_confirmed": ("+", "Closed {dist:.2f}% above the {look}-minute high", "إغلاق {dist:.2f}% فوق أعلى سعر لآخر {look} دقيقة"),
    "near_breakout": ("+", "{dist:.2f}% below the {look}-minute high (approaching)", "على بعد {dist:.2f}% تحت أعلى سعر لآخر {look} دقيقة (يقترب)"),
    "no_breakout": ("-", "Not near the {look}-minute high ({dist:.2f}%)", "بعيد عن أعلى سعر لآخر {look} دقيقة ({dist:.2f}%)"),
    "extended": ("!", "Move extended: {dist:.2f}% above breakout level, late-entry risk",
                 "الحركة ممتدة: {dist:.2f}% فوق مستوى الاختراق، خطر دخول متأخر"),
    "vwap_extended": ("!", "Price {x:.1f}% above VWAP (stretched)", "السعر {x:.1f}% فوق VWAP (مبالغ فيه)"),
    "liquidity_ok": ("+", "Avg dollar volume ${dv:,.0f}/session", "متوسط قيمة التداول ${dv:,.0f} للجلسة"),
    "liquidity_low": ("-", "Low liquidity: ${dv:,.0f}/session is under the minimum", "سيولة ضعيفة: ${dv:,.0f} للجلسة أقل من الحد"),
    "liquidity_missing": ("!", "Liquidity baseline unavailable", "خط الأساس للسيولة غير متاح"),
    "spread_ok": ("+", "Spread {sp:.2f}%", "السبريد {sp:.2f}%"),
    "spread_wide": ("!", "Wide spread {sp:.2f}%", "سبريد مرتفع {sp:.2f}%"),
    "spread_missing": ("!", "Bid/ask spread unavailable from this data source", "السبريد غير متاح من مصدر البيانات هذا"),
    "rs_strong": ("+", "Outperforming the benchmark by {rs:+.2f}pp over 15 min", "يتفوق على المؤشر بـ {rs:+.2f} نقطة خلال 15 دقيقة"),
    "rs_weak": ("-", "Not outperforming the benchmark ({rs:+.2f}pp)", "لا يتفوق على المؤشر ({rs:+.2f} نقطة)"),
    "rs_missing": ("!", "Relative strength vs benchmark unavailable", "القوة النسبية مقابل المؤشر غير متاحة"),
    "hold": ("+", "Holding above breakout level for {n} bars ({hold:.0%} of closes)", "يثبت فوق مستوى الاختراق منذ {n} شمعة ({hold:.0%} من الإغلاقات)"),
    "trend_aligned": ("+", "Above VWAP and EMA20 > EMA50", "فوق VWAP و EMA20 أعلى من EMA50"),
    "trend_partial": ("-", "Only partly aligned with VWAP/EMAs", "توافق جزئي فقط مع VWAP والمتوسطات"),
    "continuation_missing": ("!", "Continuation signals unavailable", "مؤشرات الاستمرارية غير متاحة"),
    "failed_breakout": ("!", "Breakout failed: closed back below the level", "فشل الاختراق: إغلاق عاد تحت المستوى"),
    "coverage_low": ("!", "Only {cov:.0%} of the score is backed by data", "{cov:.0%} فقط من التقييم مدعوم ببيانات"),
}


@dataclass(frozen=True)
class Reason:
    code: str
    kind: str
    en: str
    ar: str


def make_reason(code: str, **kw) -> Reason:
    kind, en, ar = TEMPLATES[code]
    return Reason(code, kind, en.format(**kw), ar.format(**kw))


@dataclass
class ScoreResult:
    score: float
    coverage: float
    components: dict[str, dict] = field(default_factory=dict)
    reasons: list[Reason] = field(default_factory=list)
    extended: bool = False
    note: str = "Heuristic ranking score from rules. Not a probability of success."


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and not pd.isna(x)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def score_row(row: pd.Series, cfg: ScannerConfig) -> ScoreResult:
    w = cfg.weights
    total_w = float(sum(w.values()))
    comps: dict[str, dict] = {}
    reasons: list[Reason] = []
    look = cfg.breakout_lookback

    def put(name: str, norm_parts: list[tuple[float, float | None]]):
        """norm_parts: [(share_of_component_weight, norm in [0,1] or None if unavailable)]."""
        maxw = float(w.get(name, 0))
        pts = avail = 0.0
        for share, norm in norm_parts:
            avail += maxw * share if norm is not None else 0
            pts += maxw * share * norm if norm is not None else 0
        comps[name] = {"points": round(pts, 2), "max": maxw, "available_weight": round(avail, 2)}

    # 1) relative volume ------------------------------------------------------------------------
    rv = row.get("rvol")
    if _ok(rv):
        put("rvol", [(1.0, _clip01((rv - 1) / 4))])
        reasons.append(make_reason("rvol_high" if rv >= cfg.early_min_rvol else "rvol_normal", rvol=float(rv)))
    else:
        put("rvol", [(1.0, None)])
        reasons.append(make_reason("rvol_missing"))

    # 2) momentum -------------------------------------------------------------------------------
    r5, ac = row.get("ret5"), row.get("accel")
    if _ok(r5):
        n_ret = _clip01(r5 / 2.0)
        n_acc = _clip01(ac / 1.0) if _ok(ac) else 0.0
        put("momentum", [(1.0, 0.6 * n_ret + 0.4 * n_acc)])
        if r5 > 0.3 and _ok(ac) and ac > 0:
            reasons.append(make_reason("momentum_up", ret5=float(r5), accel=float(ac)))
        elif r5 <= 0.3:
            reasons.append(make_reason("momentum_weak", ret5=float(r5)))
    else:
        put("momentum", [(1.0, None)])

    # 3) breakout -------------------------------------------------------------------------------
    dist = row.get("dist_pct")
    extended = False
    if _ok(dist):
        near = cfg.near_breakout_pct
        if dist > 0:
            if dist <= cfg.max_extension_pct * 0.25:
                n_b = 1.0
            elif dist <= cfg.max_extension_pct:
                n_b = 1.0 - 0.7 * (dist - cfg.max_extension_pct * 0.25) / (cfg.max_extension_pct * 0.75)
            else:
                n_b = 0.2
                extended = True
            reasons.append(make_reason("breakout_confirmed", dist=float(dist), look=look))
        elif dist >= -near:
            n_b = 0.5 + 0.4 * (1 - abs(dist) / near)
            reasons.append(make_reason("near_breakout", dist=abs(float(dist)), look=look))
        else:
            n_b = max(0.0, 0.3 * (1 - abs(dist) / 1.5))
            reasons.append(make_reason("no_breakout", dist=float(dist), look=look))
        put("breakout", [(1.0, n_b)])
        if extended:
            reasons.append(make_reason("extended", dist=float(dist)))
    else:
        put("breakout", [(1.0, None)])
    vx = row.get("ext_vwap_pct")
    if _ok(vx) and vx > cfg.max_vwap_extension_pct:
        extended = True
        reasons.append(make_reason("vwap_extended", x=float(vx)))

    # 4) liquidity: 60% dollar volume, 40% spread -------------------------------------------------
    dv, sp = row.get("avg_dollar_vol"), row.get("spread_pct")
    if _ok(dv):
        n_dv = _clip01(math.log10(dv / cfg.min_avg_dollar_volume)) if dv >= cfg.min_avg_dollar_volume else 0.0
        reasons.append(make_reason("liquidity_ok" if dv >= cfg.min_avg_dollar_volume else "liquidity_low", dv=float(dv)))
    else:
        n_dv = None
        reasons.append(make_reason("liquidity_missing"))
    if _ok(sp):
        n_sp = _clip01(1 - sp / cfg.max_spread_pct)
        reasons.append(make_reason("spread_ok" if sp <= cfg.max_spread_pct else "spread_wide", sp=float(sp)))
    else:
        n_sp = None
        reasons.append(make_reason("spread_missing"))
    put("liquidity", [(0.6, n_dv), (0.4, n_sp)])

    # 5) relative strength vs benchmark ----------------------------------------------------------
    rs = row.get("rs15")
    if _ok(rs):
        put("relstrength", [(1.0, _clip01(rs / 1.5))])
        reasons.append(make_reason("rs_strong" if rs > 0.3 else "rs_weak", rs=float(rs)))
    else:
        put("relstrength", [(1.0, None)])
        reasons.append(make_reason("rs_missing"))

    # 6) continuation / confirmation -------------------------------------------------------------
    since, hold = row.get("bars_since_breakout"), row.get("hold_ratio")
    if bool(row.get("failed_breakout", False)):
        put("continuation", [(1.0, 0.0)])
        reasons.append(make_reason("failed_breakout"))
    elif _ok(since) and _ok(hold):
        put("continuation", [(1.0, _clip01(hold * min(1.0, (since + 1) / 5)))])
        reasons.append(make_reason("hold", n=int(since) + 1, hold=float(hold)))
    else:
        c, v, e20, e50 = row.get("Close"), row.get("vwap"), row.get("ema20"), row.get("ema50")
        parts = []
        if _ok(c) and _ok(v):
            parts.append(1.0 if c > v else 0.0)
        if _ok(e20) and _ok(e50):
            parts.append(1.0 if e20 > e50 else 0.0)
        if parts:
            frac = sum(parts) / len(parts)
            put("continuation", [(1.0, 0.5 * frac)])  # no breakout yet: at most half credit
            reasons.append(make_reason("trend_aligned" if frac == 1.0 else "trend_partial"))
        else:
            put("continuation", [(1.0, None)])
            reasons.append(make_reason("continuation_missing"))

    points = sum(c["points"] for c in comps.values())
    avail = sum(c["available_weight"] for c in comps.values())
    score = 100.0 * points / total_w if total_w > 0 else 0.0
    coverage = avail / total_w if total_w > 0 else 0.0
    if not np.isfinite(score):
        score = 0.0
    score = float(min(100.0, max(0.0, score)))
    if coverage < cfg.min_coverage:
        reasons.append(make_reason("coverage_low", cov=coverage))
    return ScoreResult(round(score, 1), round(float(coverage), 3), comps, reasons, extended)
