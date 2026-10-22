"""Shared, cached objects for the UI (one config, one journal, one scanner per session)."""
from __future__ import annotations

import streamlit as st

from alerts.alert_manager import TelegramNotifier
from history.sec import SecClient
from scanner.config import ScannerConfig
from scanner.engine import Scanner
from scanner.providers import YFinanceMinuteProvider
from scanner.signal_log import SignalLog


@st.cache_resource(show_spinner=False)
def journal() -> SignalLog:
    return SignalLog()


@st.cache_resource(show_spinner=False)
def sec_client() -> SecClient:
    return SecClient()


@st.cache_resource(show_spinner=False)
def provider() -> YFinanceMinuteProvider:
    return YFinanceMinuteProvider()


def cfg() -> ScannerConfig:
    if "scanner_cfg" not in st.session_state:
        st.session_state["scanner_cfg"] = ScannerConfig.load()
    return st.session_state["scanner_cfg"]


def notifier() -> TelegramNotifier:
    return TelegramNotifier()


def make_scanner(names: dict | None = None) -> Scanner:
    return Scanner(provider(), cfg(), journal(), notifier(), names=names)
