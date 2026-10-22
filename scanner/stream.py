"""Generic resilient streaming wrapper (reconnect with back-off, de-duplication, timestamp sanity checks).

No vendor is hard-wired: pass `connect`, a callable that opens a connection and yields message dicts such as
{"symbol": "AAPL", "price": 190.1, "size": 100, "ts": <aware datetime>}. A WebSocket client for a paid provider
(Polygon, Alpaca, Finnhub, ...) can be plugged in later by implementing that one callable. It is covered by unit
tests with fake connections only; it has NOT been run against a live vendor.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

log = logging.getLogger(__name__)


class Deduper:
    """Remembers the last `size` message keys; `seen()` is True for a repeat."""

    def __init__(self, size: int = 5000):
        self.size, self._d = size, OrderedDict()

    def seen(self, key) -> bool:
        if key in self._d:
            return True
        self._d[key] = None
        if len(self._d) > self.size:
            self._d.popitem(last=False)
        return False


def valid_timestamp(ts: datetime | None, now: datetime, max_future_s: float = 5, max_age_s: float = 3600) -> bool:
    if ts is None or ts.tzinfo is None:
        return False
    return -max_future_s <= (now - ts).total_seconds() <= max_age_s


class ReconnectingStream:
    def __init__(self, connect: Callable[[], Iterable[dict]], on_message: Callable[[dict], None], *,
                 base_delay: float = 1.0, max_delay: float = 60.0, max_age_s: float = 3600,
                 sleep: Callable[[float], None] | None = None, clock: Callable[[], datetime] | None = None):
        self.connect, self.on_message = connect, on_message
        self.base_delay, self.max_delay, self.max_age_s = base_delay, max_delay, max_age_s
        self._sleep = sleep or time.sleep
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._dedup = Deduper()
        self._stop = threading.Event()
        self.connected = False
        self.reconnects = 0
        self.dropped_duplicates = 0
        self.dropped_invalid = 0
        self.errors: list[str] = []
        self.last_message_at: datetime | None = None

    def stop(self) -> None:
        self._stop.set()

    def stale(self, after_s: float = 30) -> bool:
        return self.last_message_at is None or (self._clock() - self.last_message_at) > timedelta(seconds=after_s)

    def run(self, max_attempts: int | None = None) -> None:
        delay, attempts = self.base_delay, 0
        while not self._stop.is_set() and (max_attempts is None or attempts < max_attempts):
            attempts += 1
            try:
                self.connected = True
                for msg in self.connect():
                    if self._stop.is_set():
                        break
                    delay = self.base_delay            # healthy traffic resets the back-off
                    now = self._clock()
                    if not valid_timestamp(msg.get("ts"), now, max_age_s=self.max_age_s):
                        self.dropped_invalid += 1
                        continue
                    if self._dedup.seen((msg.get("symbol"), msg.get("ts"), msg.get("price"), msg.get("size"), msg.get("id"))):
                        self.dropped_duplicates += 1
                        continue
                    self.last_message_at = now
                    self.on_message(msg)
            except Exception as exc:  # connection dropped / vendor error: log type only, never credentials
                self.errors = (self.errors + [f"{type(exc).__name__}: {exc}"])[-20:]
                log.warning("stream error (%s); reconnecting", type(exc).__name__)
            finally:
                self.connected = False
            if self._stop.is_set():
                break
            self.reconnects += 1
            self._sleep(delay)
            delay = min(self.max_delay, delay * 2)
