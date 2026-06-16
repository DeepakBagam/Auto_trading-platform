from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.engine import Engine

SYMBOL_PATTERNS = {
    "nifty50": ["NIFTY 50"],
    "banknifty": ["NIFTY BANK", "BANK NIFTY"],
    "indiavix": ["INDIA VIX"],
    "sensex": ["SENSEX"],
}

INTERVAL_ALIAS = {
    "1m": "1minute",
    "30m": "30minute",
    "1d": "day",
}


_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z0-9_ ]+$")
_ALLOWED_INTERVALS = {"1minute", "30minute", "day"}


def _safe_like_pattern(value: str) -> str:
    """Escape SQL LIKE special characters and strip anything non-alphanumeric/space."""
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_").replace("'", "''")
    return escaped.upper()


def _build_candle_view_sql(view_name: str, patterns: list[str], interval: str) -> str:
    if interval not in _ALLOWED_INTERVALS:
        raise ValueError(f"Unsupported interval for view: {interval!r}")
    like_clauses = " OR ".join(
        [f"UPPER(instrument_key) LIKE '%{_safe_like_pattern(p)}%' ESCAPE '\\'" for p in patterns]
    )
    return f"""
CREATE VIEW {view_name} AS
SELECT
  id,
  instrument_key,
  interval,
  ts,
  open,
  high,
  low,
  close,
  volume,
  oi,
  source,
  ingested_at
FROM raw_candles
WHERE ({like_clauses})
  AND interval = '{interval}'
"""


def create_symbol_interval_views(engine: Engine) -> None:
    with engine.begin() as conn:
        for slug, patterns in SYMBOL_PATTERNS.items():
            for short, interval in INTERVAL_ALIAS.items():
                candle_view = f"candles_{slug}_{short}"
                conn.execute(text(f"DROP VIEW IF EXISTS {candle_view}"))
                conn.execute(text(_build_candle_view_sql(candle_view, patterns, interval)))
