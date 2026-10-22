from datetime import date, datetime, timezone

from data import us_calendar as cal
from data.market_data import market_session


def test_2026_full_day_holidays():
    got = sorted(cal.holidays(2026))
    assert got == [date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
                   date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25)]


def test_observed_rules_and_new_year_on_saturday():
    assert date(2022, 12, 31) not in cal.holidays(2022) and date(2021, 12, 31) not in cal.holidays(2021)
    assert date(2021, 7, 5) in cal.holidays(2021)      # July 4 on Sunday -> Monday
    assert date(2021, 12, 24) in cal.holidays(2021)    # Christmas on Saturday -> Friday


def test_early_closes():
    for d in (date(2025, 11, 28), date(2025, 12, 24), date(2025, 7, 3), date(2026, 11, 27)):
        assert cal.is_early_close(d)
    assert not cal.is_early_close(date(2026, 7, 2))    # July 4 is a Saturday: no early close
    assert cal.session_hours(date(2026, 11, 27))[1].hour == 13


def test_market_status_uses_calendar():
    assert cal.market_status(datetime(2026, 10, 21, 14, 0, tzinfo=timezone.utc)) == "open"
    assert cal.market_status(datetime(2026, 10, 24, 15, 0, tzinfo=timezone.utc)) == "closed"          # Saturday
    assert market_session(datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc)) == "closed"            # Thanksgiving
    assert cal.market_status(datetime(2026, 11, 27, 19, 0, tzinfo=timezone.utc)) != "open"           # after the 13:00 ET early close
