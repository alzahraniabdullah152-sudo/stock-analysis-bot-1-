"""In-memory ring buffer of recent log records for the status page (never stores secrets: URLs with tokens are not logged)."""
from __future__ import annotations

import logging
from collections import deque

_BUF: deque = deque(maxlen=400)
_INSTALLED = False


class _Handler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _BUF.append({"time": record.created, "level": record.levelname, "logger": record.name,
                         "message": record.getMessage()[:500]})
        except Exception:
            pass


def install() -> None:
    global _INSTALLED
    if not _INSTALLED:
        logging.getLogger().addHandler(_Handler())
        _INSTALLED = True


def records(min_level: str = "INFO") -> list[dict]:
    lv = getattr(logging, min_level, logging.INFO)
    return [r for r in list(_BUF) if getattr(logging, r["level"], 0) >= lv][::-1]
