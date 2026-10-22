"""Shared helpers for pages: session-scoped config, journal and provider."""
from __future__ import annotations

import streamlit as st

from alerts.alert_manager import TelegramNotifier
from scanner.config import ScannerConfig
from scanner.providers import YFinanceMinuteProvider
from scanner.signal_log import SignalLog

WARNING_AR = {
    "market_closed": "السوق مغلق: تُعرض نتائج آخر جلسة للمراجعة فقط، ولا توجد إشارات قابلة للتنفيذ.",
    "delayed_data": "البيانات متأخرة وغير رسمية: لا تتعامل مع الأسعار على أنها لحظية.",
}


def cfg() -> ScannerConfig:
    if "scan_cfg" not in st.session_state:
        st.session_state.scan_cfg = ScannerConfig.load()
    return st.session_state.scan_cfg


def journal() -> SignalLog:
    if "journal" not in st.session_state:
        st.session_state.journal = SignalLog()
    return st.session_state.journal


def provider():
    if "provider" not in st.session_state:
        st.session_state.provider = YFinanceMinuteProvider()
    return st.session_state.provider


def notifier() -> TelegramNotifier:
    return TelegramNotifier()


def warn_text(code: str) -> str:
    key = code.split(":")[0]
    return WARNING_AR.get(key, code)
