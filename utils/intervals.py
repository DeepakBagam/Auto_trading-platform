from __future__ import annotations


INTERVAL_ALIASES = {
    "1m": "1minute",
    "1min": "1minute",
    "1minute": "1minute",
}

INTERVAL_QUERY_PATTERN = "^(1m|1min|1minute)$"


def normalize_interval(interval: str) -> str:
    key = str(interval or "").strip().lower()
    try:
        return INTERVAL_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported interval={interval}") from exc
