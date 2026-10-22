"""Watchlist, alert rules, cooldown/dedup, Telegram delivery and scanning."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from analysis.signal_engine import compute_signals
from config import INTRADAY, SETTINGS
from data.market_data import DataError, fetch
from indicators.technical_indicators import add_all

log = logging.getLogger(__name__)
STATE_FILE = Path(".alert_state.json")
WATCHLIST_FILE = Path("watchlist.json")
RULE_KINDS = ("price_above", "price_below", "ema_cross_up", "ema_cross_down",
              "rsi_above", "rsi_below", "breakout", "breakdown")


@dataclass(frozen=True)
class AlertRule:
    """kind is one of RULE_KINDS; value is the price/RSI level where relevant."""

    kind: str
    value: float | None = None

    def key(self, ticker: str, tf: str) -> str:
        return f"{ticker}|{tf}|{self.kind}|{self.value}"


def load_watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        try:
            return [str(x).upper() for x in json.loads(WATCHLIST_FILE.read_text())]
        except (json.JSONDecodeError, OSError):
            log.warning("watchlist file unreadable; using defaults")
    return ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]


def save_watchlist(tickers: list[str]) -> None:
    WATCHLIST_FILE.write_text(json.dumps(sorted(set(t.upper() for t in tickers))))


def evaluate_rule(rule: AlertRule, d: pd.DataFrame, sigs: pd.DataFrame) -> str | None:
    """Evaluate on the LAST row of `d` against the previous row (crossing semantics)."""
    r, p = d.iloc[-1], d.iloc[-2]
    k, v = rule.kind, rule.value
    if k == "price_above" and v is not None and r.Close > v >= p.Close:
        return f"Close {r.Close:.2f} crossed above {v}"
    if k == "price_below" and v is not None and r.Close < v <= p.Close:
        return f"Close {r.Close:.2f} crossed below {v}"
    if pd.notna(r.get("ema21")) and pd.notna(p.get("ema21")):
        if k == "ema_cross_up" and r.ema9 > r.ema21 and p.ema9 <= p.ema21:
            return "EMA9 crossed above EMA21"
        if k == "ema_cross_down" and r.ema9 < r.ema21 and p.ema9 >= p.ema21:
            return "EMA9 crossed below EMA21"
    if pd.notna(r.get("rsi14")) and pd.notna(p.get("rsi14")):
        if k == "rsi_above" and v is not None and r.rsi14 > v >= p.rsi14:
            return f"RSI {r.rsi14:.1f} crossed above {v}"
        if k == "rsi_below" and v is not None and r.rsi14 < v <= p.rsi14:
            return f"RSI {r.rsi14:.1f} crossed below {v}"
    if k == "breakout" and sigs["breakout"].iloc[-1] > 0:
        return "Breakout above prior 20-bar high with volume"
    if k == "breakdown" and sigs["breakout"].iloc[-1] < 0:
        return "Breakdown below prior 20-bar low with volume"
    return None


class TelegramNotifier:
    """Optional notifier; credentials only from environment variables."""

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = SETTINGS.telegram_token if token is None else token
        self.chat_id = SETTINGS.telegram_chat_id if chat_id is None else chat_id

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            r = requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                              data={"chat_id": self.chat_id, "text": text}, timeout=10)
            r.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.error("Telegram send failed: %s", type(exc).__name__)  # never log the token-bearing URL
            return False


class AlertManager:
    """Evaluates rules per watchlist ticker with per-candle dedup and cooldown."""

    def __init__(self, cooldown_min: int | None = None, notifier: TelegramNotifier | None = None):
        mins = SETTINGS.alert_cooldown_minutes if cooldown_min is None else cooldown_min
        self.cooldown = timedelta(minutes=mins)
        self.notifier = notifier or TelegramNotifier()
        self._last: dict[str, datetime] = {}
        self._last_bar: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if STATE_FILE.exists():
            try:
                raw = json.loads(STATE_FILE.read_text())
                self._last = {k: datetime.fromisoformat(v) for k, v in raw.get("last", {}).items()}
                self._last_bar = dict(raw.get("bars", {}))
            except (json.JSONDecodeError, ValueError, AttributeError, OSError):
                log.warning("alert state unreadable; starting fresh")

    def _save(self) -> None:
        try:
            STATE_FILE.write_text(json.dumps({"last": {k: v.isoformat() for k, v in self._last.items()},
                                              "bars": self._last_bar}))
        except OSError as exc:
            log.warning("could not save alert state: %s", exc)

    def should_fire(self, key: str, bar_id: str, now: datetime) -> bool:
        if self._last_bar.get(key) == bar_id:
            return False  # this candle already alerted
        last = self._last.get(key)
        return last is None or now - last >= self.cooldown

    def scan(self, tickers: list[str], rules: list[AlertRule], tf: str = "1d",
             provisional: bool = False) -> list[dict]:
        """One scan. provisional=True evaluates the unfinished candle and labels alerts accordingly."""
        out: list[dict] = []
        now = datetime.now(timezone.utc)
        for t in tickers:
            try:
                md = fetch(t, tf, include_incomplete=provisional)
            except DataError as exc:
                log.warning("scan skip %s: %s", t, exc)
                out.append({"ticker": t, "error": str(exc)})
                continue
            d = add_all(md.df, intraday=tf in INTRADAY)
            if len(d) < 3:
                continue
            sigs = compute_signals(d, tf in INTRADAY)
            for rule in rules:
                msg = evaluate_rule(rule, d, sigs)
                if not msg:
                    continue
                key, bar_id = rule.key(t, tf), str(d.index[-1])
                if not self.should_fire(key, bar_id, now):
                    continue
                status = "completed candle" if md.last_bar_complete else "UNFINISHED candle (may change)"
                text = f"[{t} {tf}] {msg} - {status}, bar {bar_id}. Educational signal, not advice."
                self._last[key], self._last_bar[key] = now, bar_id
                out.append({"ticker": t, "timeframe": tf, "rule": rule.kind, "message": text,
                            "completed": md.last_bar_complete, "telegram_sent": self.notifier.send(text)})
            time.sleep(0.3)  # stay polite to the data provider
        self._save()
        return out


class Scheduler:
    """In-process periodic scan; stops when the app process stops."""

    def __init__(self, manager: AlertManager, tickers_fn: Callable[[], list[str]],
                 rules_fn: Callable[[], list[AlertRule]], tf: str, minutes: int):
        self.manager, self.tickers_fn, self.rules_fn, self.tf = manager, tickers_fn, rules_fn, tf
        self.minutes = max(5, minutes)  # floor protects provider rate limits
        self._stop = threading.Event()
        self.results: list[dict] = []
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                new = self.manager.scan(self.tickers_fn(), self.rules_fn(), self.tf)
                self.results = (new + self.results)[:200]
            except Exception:  # keep the thread alive; traceback is logged, not hidden
                log.exception("scheduled scan failed")
            self._stop.wait(self.minutes * 60)
