"""Repeating scan in a background thread (in-process: stops when the app stops)."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Callable

from scanner.engine import ScanReport, Scanner

log = logging.getLogger(__name__)
MIN_INTERVAL_S = 120   # 1-minute bars; polling faster only hits provider rate limits


class ScanLoop:
    def __init__(self, scanner: Scanner, tickers_fn: Callable[[], list[str]], interval_s: int = 300,
                 send_alerts: bool = False):
        self.scanner, self.tickers_fn = scanner, tickers_fn
        self.interval_s = max(MIN_INTERVAL_S, int(interval_s))
        self.send_alerts = send_alerts
        self.latest: ScanReport | None = None
        self.runs = 0
        self.errors: list[str] = []
        self.last_run: datetime | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="scan-loop")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> ScanReport:
        rep = self.scanner.scan(self.tickers_fn(), send_alerts=self.send_alerts)
        self.latest, self.runs, self.last_run = rep, self.runs + 1, datetime.now(timezone.utc)
        return rep

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as exc:  # keep looping; the error is kept for the status page
                log.exception("scan loop iteration failed")
                self.errors = (self.errors + [f"{type(exc).__name__}: {exc}"])[-20:]
            self._stop.wait(self.interval_s)
