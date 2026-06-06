from __future__ import annotations

import re

from backend.utils.constants import HOUR_INTERVALS, MINUTE_INTERVALS, SUPPORTED_INTERVALS


INTERVAL_ALIASES = {
    "1d": "day",
    "1day": "day",
    "day": "day",
    "1w": "week",
    "1week": "week",
    "week": "week",
    "1mo": "month",
    "1mon": "month",
    "1month": "month",
    "month": "month",
}
for value in range(1, 301):
    INTERVAL_ALIASES[f"{value}m"] = f"{value}minute"
    INTERVAL_ALIASES[f"{value}min"] = f"{value}minute"
    INTERVAL_ALIASES[f"{value}mins"] = f"{value}minute"
    INTERVAL_ALIASES[f"{value}minute"] = f"{value}minute"
    INTERVAL_ALIASES[f"{value}minutes"] = f"{value}minute"
for value in range(1, 6):
    INTERVAL_ALIASES[f"{value}h"] = f"{value}hour"
    INTERVAL_ALIASES[f"{value}hr"] = f"{value}hour"
    INTERVAL_ALIASES[f"{value}hour"] = f"{value}hour"
    INTERVAL_ALIASES[f"{value}hours"] = f"{value}hour"

INTERVAL_QUERY_PATTERN = (
    r"^("
    r"(?:[1-9]|[1-9][0-9]|[12][0-9]{2}|300)(?:m|min|mins|minute|minutes)"
    r"|(?:[1-5])(?:h|hr|hour|hours)"
    r"|1?d|day|1?w|week|1mo|1mon|1month|month"
    r")$"
)


def normalize_interval(interval: str) -> str:
    key = re.sub(r"[\s_-]+", "", str(interval or "").strip().lower())
    try:
        normalized = INTERVAL_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported interval={interval}") from exc
    if normalized not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval={interval}")
    return normalized


def interval_to_upstox_v3_params(interval: str) -> tuple[str, str]:
    normalized = normalize_interval(interval)
    if normalized in MINUTE_INTERVALS:
        return "minutes", normalized.removesuffix("minute")
    if normalized in HOUR_INTERVALS:
        return "hours", normalized.removesuffix("hour")
    if normalized == "day":
        return "days", "1"
    if normalized == "week":
        return "weeks", "1"
    if normalized == "month":
        return "months", "1"
    raise ValueError(f"Unsupported interval={interval}")


def interval_backfill_start(interval: str):
    normalized = normalize_interval(interval)
    if normalized in MINUTE_INTERVALS or normalized in HOUR_INTERVALS:
        from datetime import date

        return date(2022, 1, 1)
    from datetime import date

    return date(2000, 1, 1)


def interval_chunk_days(interval: str) -> int:
    normalized = normalize_interval(interval)
    if normalized in MINUTE_INTERVALS:
        value = int(normalized.removesuffix("minute"))
        return 31 if value <= 15 else 92
    if normalized in HOUR_INTERVALS:
        return 92
    if normalized == "day":
        return 3652
    return 100_000
