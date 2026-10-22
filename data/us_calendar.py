"""NYSE/NASDAQ regular-session calendar: holidays, early closes, session hours (America/New_York).

Rule-based (no external dependency). Covers the standard full-day holidays and the standard 13:00 early closes.
Special one-off closures (national days of mourning, weather events) are NOT known and are not modelled.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
OPEN = time(9, 30)
CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
PREMARKET_START = time(4, 0)
AFTERHOURS_END = time(20, 0)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) given weekday (Mon=0) of a month."""
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _observed(d: date) -> date | None:
    """Saturday holidays are observed Friday; Sunday holidays Monday.

    NYSE does not observe a New Year's Day that falls on Saturday (handled by the caller).
    """
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def holidays(year: int) -> dict[date, str]:
    out: dict[date, str] = {}
    ny = date(year, 1, 1)
    if ny.weekday() != 5:  # Saturday New Year is not moved to the previous Dec 31
        out[_observed(ny)] = "New Year's Day"
    out[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    out[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    out[_easter(year) - timedelta(days=2)] = "Good Friday"
    out[_last_weekday(year, 5, 0)] = "Memorial Day"
    if year >= 2022:
        out[_observed(date(year, 6, 19))] = "Juneteenth"
    out[_observed(date(year, 7, 4))] = "Independence Day"
    out[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    out[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"
    out[_observed(date(year, 12, 25))] = "Christmas Day"
    return out


def is_holiday(d: date) -> bool:
    # New Year's Day observed on Monday Jan 2 is covered; a Dec 31 observance never happens (see above).
    return d in holidays(d.year)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and not is_holiday(d)


def is_early_close(d: date) -> bool:
    if not is_trading_day(d):
        return False
    thanksgiving = _nth_weekday(d.year, 11, 3, 4)
    if d == thanksgiving + timedelta(days=1):
        return True
    if d.month == 7 and d.day == 3 and date(d.year, 7, 4).weekday() in (1, 2, 3, 4):
        return True
    if d.month == 12 and d.day == 24:
        return True
    return False


def session_hours(d: date) -> tuple[datetime, datetime] | None:
    """(open, close) as NY-aware datetimes for a trading day, else None."""
    if not is_trading_day(d):
        return None
    close = EARLY_CLOSE if is_early_close(d) else CLOSE
    return datetime.combine(d, OPEN, NY), datetime.combine(d, close, NY)


def market_status(now: datetime | None = None) -> str:
    """'open' | 'pre/post' | 'closed' using the holiday and early-close calendar."""
    now = (now or datetime.now(timezone.utc)).astimezone(NY)
    d = now.date()
    if not is_trading_day(d):
        return "closed"
    hrs = session_hours(d)
    assert hrs is not None
    if hrs[0] <= now < hrs[1]:
        return "open"
    ext_end = datetime.combine(d, AFTERHOURS_END if not is_early_close(d) else time(17, 0), NY)
    if datetime.combine(d, PREMARKET_START, NY) <= now < ext_end:
        return "pre/post"
    return "closed"


def previous_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def next_trading_day(d: date) -> date:
    d += timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d
