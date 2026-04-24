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


def _build_prediction_view_sql(view_name: str, symbol_label: str, interval: str) -> str:
    if interval == "day":
        return f"""
CREATE VIEW {view_name} AS
SELECT
  id,
  symbol,
  interval,
  target_session_date,
  pred_open,
  pred_high,
  pred_low,
  pred_close,
  direction,
  confidence,
  model_version,
  feature_cutoff_ist,
  generated_at,
  metadata_json
FROM predictions_daily
WHERE UPPER(symbol) = UPPER('{symbol_label}')
  AND interval = '{interval}'
"""
    return f"""
CREATE VIEW {view_name} AS
SELECT
  id,
  symbol,
  interval,
  DATE(target_ts) AS target_session_date,
  pred_open,
  pred_high,
  pred_low,
  pred_close,
  direction,
  confidence,
  model_version,
  feature_cutoff_ist,
  generated_at,
  metadata_json
FROM predictions_intraday
WHERE UPPER(symbol) = UPPER('{symbol_label}')
  AND interval = '{interval}'
"""


def create_symbol_interval_views(engine: Engine) -> None:
    with engine.begin() as conn:
        for slug, patterns in SYMBOL_PATTERNS.items():
            for short, interval in INTERVAL_ALIAS.items():
                candle_view = f"candles_{slug}_{short}"
                pred_view = f"predictions_{slug}_{short}"
                conn.execute(text(f"DROP VIEW IF EXISTS {candle_view}"))
                conn.execute(text(f"DROP VIEW IF EXISTS {pred_view}"))
                conn.execute(text(_build_candle_view_sql(candle_view, patterns, interval)))

                # Display symbol labels to match current model symbol values.
                symbol_label = {
                    "nifty50": "Nifty 50",
                    "banknifty": "Nifty Bank",
                    "indiavix": "India VIX",
                    "sensex": "SENSEX",
                }[slug]
                conn.execute(text(_build_prediction_view_sql(pred_view, symbol_label, interval)))
