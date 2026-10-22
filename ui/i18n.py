"""Tiny translation helper. Arabic is the default interface language; technical rule texts stay in English."""
from __future__ import annotations

T: dict[str, dict[str, str]] = {
    "app_title": {"ar": "محلل الأسهم الأمريكية", "en": "US Stock Analyst"},
    "disclaimer": {"ar": "أداة تعليمية وبحثية. ليست نصيحة مالية، والإشارات فرضيات مبنية على قواعد وليست تنبؤات.",
                   "en": "Educational research tool. Not financial advice; signals are rule-based hypotheses, not predictions."},
    "nav_home": {"ar": "الرئيسية", "en": "Home"}, "nav_scanner": {"ar": "الماسح اللحظي", "en": "Live scanner"},
    "nav_analysis": {"ar": "تحليل سهم", "en": "Stock analysis"}, "nav_history": {"ar": "تاريخ الشركة", "en": "Company history"},
    "nav_journal": {"ar": "السجل والأداء", "en": "Journal & performance"}, "nav_settings": {"ar": "الإعدادات والحالة", "en": "Settings & status"},
    "language": {"ar": "اللغة", "en": "Language"},
    "market_open": {"ar": "السوق مفتوح", "en": "Market open"}, "market_closed": {"ar": "السوق مغلق", "en": "Market closed"},
    "market_pre": {"ar": "ما قبل/بعد الجلسة", "en": "Pre/post session"},
    "delayed": {"ar": "بيانات متأخرة وغير لحظية", "en": "Delayed data, not real-time"},
    "realtime": {"ar": "بث مباشر", "en": "Live feed"},
    "last_bar": {"ar": "آخر شمعة مكتملة", "en": "Last completed bar"}, "source": {"ar": "المصدر", "en": "Source"},
    "score_note": {"ar": "ترتيب توجيهي وليس احتمال نجاح", "en": "Ranking aid, not a probability"},
    "coverage": {"ar": "تغطية البيانات", "en": "Data coverage"},
    "ticker": {"ar": "رمز السهم", "en": "Ticker"}, "run_scan": {"ar": "ابدأ الفحص", "en": "Run scan"},
    "scan_list": {"ar": "قائمة الأسهم (افصل بينها بفاصلة أو سطر جديد)", "en": "Tickers (comma or newline separated)"},
    "no_candidates": {"ar": "لا توجد فرص تطابق القواعد الآن", "en": "No candidates match the rules right now"},
    "no_candidates_hint": {"ar": "هذا طبيعي في أغلب الأوقات. خفّف الحدود من صفحة الإعدادات أو جرّب قائمة أوسع.",
                           "en": "This is normal most of the time. Relax thresholds in Settings or try a wider list."},
    "skipped": {"ar": "أسهم مستبعدة وأسبابها", "en": "Excluded tickers and why"},
    "reasons": {"ar": "أسباب التقييم", "en": "Why this score"},
    "invalidation": {"ar": "إبطال السيناريو", "en": "Invalidation"}, "expires": {"ar": "تنتهي الإشارة", "en": "Signal expires"},
    "state_none": {"ar": "لا توجد فرصة", "en": "No setup"}, "actionable": {"ar": "قابلة للمتابعة", "en": "Actionable"},
    "not_actionable": {"ar": "غير قابلة للمتابعة", "en": "Not actionable"},
    "blocked_market_closed": {"ar": "السوق مغلق: عرض للمراجعة فقط", "en": "Market closed: review only"},
    "blocked_data_stale": {"ar": "البيانات قديمة", "en": "Data is stale"},
    "blocked_insufficient_data": {"ar": "بيانات غير كافية", "en": "Insufficient data"},
    "blocked_extended": {"ar": "الحركة ممتدة (خطر دخول متأخر)", "en": "Move extended (late-entry risk)"},
    "send_alerts": {"ar": "أرسل تنبيهات Telegram", "en": "Send Telegram alerts"},
    "telegram_on": {"ar": "Telegram مفعّل", "en": "Telegram configured"}, "telegram_off": {"ar": "Telegram غير مفعّل (أضف المفاتيح في .env)", "en": "Telegram not configured (add keys to .env)"},
    "company": {"ar": "الشركة", "en": "Company"}, "build_report": {"ar": "أنشئ التقرير", "en": "Build report"},
    "download_html": {"ar": "نزّل التقرير (HTML)", "en": "Download report (HTML)"},
    "compare_with": {"ar": "قارن مع سهم أو مؤشر (اختياري)", "en": "Compare with a ticker or index (optional)"},
    "focus_period": {"ar": "حصر الأحداث في فترة (اختياري)", "en": "Limit events to a period (optional)"},
    "sec_off": {"ar": "إفصاحات SEC غير مفعّلة. أضف SEC_USER_AGENT في ملف .env (اسمك وإيميلك) لتفعيل التفسير بالمصادر والتحليل المالي.",
                "en": "SEC data is off. Add SEC_USER_AGENT to .env (your name and email) to enable sourced explanations and financials."},
    "save": {"ar": "احفظ", "en": "Save"}, "saved": {"ar": "تم الحفظ", "en": "Saved"},
    "reset_defaults": {"ar": "استرجع القيم الافتراضية", "en": "Restore defaults"},
    "loading": {"ar": "جارٍ التحميل...", "en": "Loading..."},
    "data_error": {"ar": "تعذر جلب البيانات", "en": "Could not load data"},
}


def tr(key: str, lang: str = "ar") -> str:
    entry = T.get(key)
    if not entry:
        return key
    return entry.get(lang) or entry.get("en") or key
