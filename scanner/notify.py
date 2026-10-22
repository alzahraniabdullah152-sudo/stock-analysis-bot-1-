"""Telegram alert text for scanner signals (plain text, no markup, <= 4096 chars)."""
from __future__ import annotations

from datetime import datetime

from data.us_calendar import NY
from scanner.states import LABELS

MAX_LEN = 4000


def _pct(x, d=2) -> str:
    return "n/a" if x is None or x != x else f"{x:+.{d}f}%"


def _num(x, d=2) -> str:
    return "n/a" if x is None or x != x else f"{x:.{d}f}"


def format_alert(c, now: datetime) -> str:
    """c: scanner.engine.Candidate. Bilingual (Arabic first), every claim carries its source and delay status."""
    f = c.features
    t_ny = now.astimezone(NY)
    ar, en = LABELS[c.state][1], LABELS[c.state][0]
    name = f" - {c.company}" if getattr(c, "company", None) else ""
    status = "بيانات متأخرة وغير لحظية" if c.delayed else "بيانات مباشرة"
    lines = [
        f"{'⚠️' if c.state == 'failed' else '📈'} {c.ticker}{name}",
        f"الحالة: {ar} ({en})",
        f"وقت الإشارة: {t_ny:%Y-%m-%d %H:%M} ET ({now:%H:%M} UTC)",
        f"السعر المرجعي: {c.ref_price:.2f}",
        f"درجة التقييم: {c.score:.0f}/100 (ترتيب توجيهي، وليست احتمال نجاح) | تغطية البيانات {c.coverage:.0%}",
        f"الحجم النسبي: {_num(f.get('rvol'), 1)}x | تغير 1د {_pct(f.get('ret1'))} | 5د {_pct(f.get('ret5'))} | 15د {_pct(f.get('ret15'))}",
        f"الاختراق: {'فوق' if (f.get('dist_pct') or -1) > 0 else 'تحت'} أعلى سعر قصير الأجل بـ {_num(abs(f['dist_pct']) if f.get('dist_pct') == f.get('dist_pct') and f.get('dist_pct') is not None else None)}%"
        + (f" (المستوى {c.level:.2f})" if c.level else ""),
        f"السبريد: {_num(f.get('spread_pct')) + '%' if _num(f.get('spread_pct')) != 'n/a' else 'غير متاح'} | متوسط قيمة التداول: "
        + ("n/a" if f.get("avg_dollar_vol") != f.get("avg_dollar_vol") or f.get("avg_dollar_vol") is None else f"${f['avg_dollar_vol']:,.0f}"),
        "",
        "الأسباب:",
    ]
    for r in c.reasons:
        mark = {"+": "✅", "-": "➖", "!": "⚠️"}[r.kind]
        lines.append(f" {mark} {r.ar}")
    lines.append("")
    lines.append("المخاطر: حركة قصيرة الأجل قد تنعكس بسرعة؛ السبريد والانزلاق قد يلغيان أي ميزة.")
    if c.invalidation:
        lines.append(f"إبطال السيناريو: إغلاق دقيقة تحت {c.invalidation:.2f}")
    if c.expires_at:
        lines.append(f"تنتهي الإشارة: {c.expires_at.astimezone(NY):%H:%M} ET ما لم تتجدد")
    cats = getattr(c, "catalysts", None)
    if cats:
        lines.append("المحفزات المعروفة: " + "; ".join(f"{x['category']} ({x['published']:%m-%d %H:%M} UTC)" for x in cats[:3]))
    else:
        lines.append("المحفز: غير معروف (لم يُعثر على مصدر موثوق)")
    lines.append(f"المصدر: {c.provider} - {status} | آخر شمعة مكتملة قبل {c.data_age_s / 60:.0f} دقيقة")
    lines.append("أداة تعليمية وليست نصيحة مالية. لا يوجد تنفيذ تلقائي للصفقات.")
    return "\n".join(lines)[:MAX_LEN]
