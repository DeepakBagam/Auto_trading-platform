from datetime import date, datetime, time, timedelta

import holidays

from utils.constants import IST_ZONE


def ist_now() -> datetime:
    return datetime.now(IST_ZONE)


def is_trading_day(day: date) -> bool:
    if day.weekday() >= 5:
        return False
    in_holidays = day in holidays.country_holidays("IN")
    return not in_holidays


def next_trading_day(day: date) -> date:
    probe = day + timedelta(days=1)
    while not is_trading_day(probe):
        probe += timedelta(days=1)
    return probe


def previous_trading_day(day: date) -> date:
    probe = day - timedelta(days=1)
    while not is_trading_day(probe):
        probe -= timedelta(days=1)
    return probe


def market_session_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time(9, 15), tzinfo=IST_ZONE)
    end = datetime.combine(day, time(15, 30), tzinfo=IST_ZONE)
    return start, end
