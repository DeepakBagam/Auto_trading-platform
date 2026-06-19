from __future__ import annotations

import calendar as month_calendar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
from dateutil.relativedelta import relativedelta
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from backend.api.market_stream_runtime import get_market_stream_runtime_status
from backend.data_layer.collectors.upstox_option_chain import UpstoxOptionChainCollector
from backend.db.models import DataFreshness, ExecutionOrder, ExecutionPosition, OptionQuote, RawCandle, SignalLog
from backend.execution_engine.slippage_tracker import get_vix_context
from backend.execution_engine.strike_selector import get_atm_iv as _strike_get_atm_iv
from backend.feature_engine.price_features import build_price_features
from backend.execution_engine.options_engine import (
    OptionQuoteView,
    build_chain_rows,
    nearest_strike,
    next_weekly_expiries,
    strike_step_for_symbol,
    synthetic_option_chain,
)
from backend.utils.calendar_utils import is_trading_day, market_session_bounds, next_trading_day, previous_trading_day
from backend.utils.app_state import get_paper_reset_at, get_paper_starting_balance, get_runtime_trading_mode
from backend.utils.config import Settings, get_settings
from backend.utils.constants import IST_ZONE
from backend.utils.intervals import normalize_interval
from backend.utils.notifications import smtp_ready
from backend.utils.redis_cache import get_json as redis_get_json
from backend.utils.redis_cache import set_json as redis_set_json
from backend.utils.symbols import (
    canonical_symbol_name,
    display_symbol_from_instrument_key,
    instrument_key_filter,
    normalize_symbol_key,
    sort_display_symbols,
    symbol_aliases,
    symbol_value_filter,
)

LIVE_INTERVAL = "1minute"
SIGNAL_INTERVAL = "1minute"  # Changed from 5minute to 1minute for faster signals
DIRECTIONAL_SIGNALS_ENABLED = True
DEFAULT_SIGNAL_COOLDOWN_MINUTES = 12
DEFAULT_SIGNAL_MIN_SCORE = 63.0
VIX_MAX_THRESHOLD = 20.0   # Skip signals when VIX is too high (options too expensive)
VIX_MIN_THRESHOLD = 11.0   # Skip signals when VIX is too low (premiums too small)
DEFAULT_MAX_SIGNALS_PER_DAY = 2
DEFAULT_CHART_RANGE = "1d"
SIGNAL_ENTRY_START = time(9, 45)
SIGNAL_ENTRY_END = time(15, 0)
OPTION_STOP_LOSS_PCT = 0.35
OPTION_TARGET_PCT = 0.60
OPTION_TRAIL_TRIGGER_PCT = 0.50
OPTION_TRAIL_STOP_PCT = 0.20
CHART_RANGE_SPECS: dict[str, dict[str, Any]] = {
    "1d": {"label": "1D", "interval": "1minute", "days": 1, "supports_live": True},
    "5d": {"label": "5D", "interval": "5minute", "days": 5, "supports_live": False},
    "1m": {"label": "1M", "interval": "15minute", "days": 31, "supports_live": False},
    "6m": {"label": "6M", "interval": "1hour", "days": 183, "supports_live": False},
    "1y": {"label": "1Y", "interval": "day", "years": 1, "supports_live": False},
    "all": {"label": "ALL", "interval": "1minute", "all_history": True, "supports_live": True},
}
CHART_INTERVAL_OPTIONS: list[dict[str, str]] = [
    {"key": "1m", "label": "1m", "interval": "1minute"},
    {"key": "5m", "label": "5m", "interval": "5minute"},
    {"key": "15m", "label": "15m", "interval": "15minute"},
    {"key": "30m", "label": "30m", "interval": "30minute"},
    {"key": "1h", "label": "1h", "interval": "1hour"},
    {"key": "1d", "label": "1D", "interval": "day"},
]
CHART_CONFIRMATION_RULES: dict[str, tuple[str, str]] = {
    "1minute": ("3min", "5min"),
    "30minute": ("90min", "150min"),
    "1hour": ("3h", "5h"),
    "day": ("3D", "5D"),
    "week": ("3W", "5W"),
    "month": ("90D", "150D"),
}
CHART_MARKER_LIMITS: dict[str, int] = {
    "1d": 40,
    "5d": 80,
    "1m": 60,
    "6m": 40,
    "1y": 30,
    "2y": 30,
}
_CHART_PAYLOAD_CACHE: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}
_INSTRUMENT_RESOLVE_CACHE: dict[str, tuple[str, str]] = {}

# ---------------------------------------------------------------------------
# Market regime constants
# ---------------------------------------------------------------------------
_REGIME_TRENDING = "TRENDING"
_REGIME_RANGE = "RANGE_BOUND"
_REGIME_VOLATILE = "HIGH_VOLATILITY"

# Data staleness: if last candle is older than this, halt signal generation.
_DATA_STALE_SECONDS = 86400  # 24 hours - disabled for testing with historical data


def _detect_regime(
    adx: float,
    plus_di: float,
    minus_di: float,
    atr: float,
    atr_mean: float,
) -> str:
    """Classify current market into TRENDING / RANGE_BOUND / HIGH_VOLATILITY.

    Priority order: HIGH_VOLATILITY > TRENDING > RANGE_BOUND so we never
    mistake an ATR spike for normal trend continuation.
    """
    atr_ratio = atr / max(atr_mean, 1e-9)
    if atr_ratio > 1.6:
        return _REGIME_VOLATILE
    if adx >= 25.0 and abs(plus_di - minus_di) >= 10.0:
        return _REGIME_TRENDING
    if adx < 20.0:
        return _REGIME_RANGE
    return _REGIME_TRENDING  # transitional → treat as trending


def _dynamic_threshold(regime: str, vix_ratio: float) -> float:
    """Adaptive minimum score.  Higher = harder to fire a signal.

    Base shifts with VIX relative to its own MA: a spike raises the bar,
    a calm market lowers it slightly.  Regime then applies a flat offset.
    """
    base = 58.0 + (vix_ratio - 1.0) * 15.0
    if regime == _REGIME_TRENDING:
        base -= 6.0    # Trending: easier threshold, continuation expected
    elif regime == _REGIME_RANGE:
        base += 10.0   # Range-bound: much stricter, only clean breakouts
    elif regime == _REGIME_VOLATILE:
        base += 8.0    # Volatile: stricter, premium already expensive
    return max(48.0, min(80.0, base))


def _rsi_buy_sell_bands(regime: str) -> tuple[float, float, float, float]:
    """Return (buy_lo, buy_hi, sell_lo, sell_hi) for RSI momentum gate.

    In a trending market RSI can stay elevated for many bars; forcing it
    below 72 misses the meat of the move.  Range-bound markets need tighter
    bands to avoid buying near resistance.
    """
    if regime == _REGIME_TRENDING:
        return 50.0, 82.0, 18.0, 50.0
    if regime == _REGIME_RANGE:
        return 56.0, 72.0, 28.0, 44.0
    # HIGH_VOLATILITY
    return 52.0, 78.0, 22.0, 48.0


def _regime_exit_multipliers(regime: str) -> tuple[float, float, float]:
    """Return (sl_atr, t1_atr, t2_atr) multipliers for SL and target levels.

    Trending: ride the move with wider stops and ambitious targets.
    Range: quick scalp with tight stops.
    Volatile: medium — market can reverse fast.
    """
    if regime == _REGIME_TRENDING:
        return 1.5, 2.5, 4.0
    if regime == _REGIME_RANGE:
        return 0.8, 1.3, 2.0
    return 1.2, 2.0, 3.0


def _expiry_entry_cutoff(days_to_expiry: int) -> time | None:
    """On expiry day restrict entries to first 90 minutes only."""
    if days_to_expiry == 0:
        return time(11, 0)
    return None


@dataclass(slots=True)
class MarketContext:
    symbol: str
    instrument_key: str
    latest_price: float
    latest_candle_ts: datetime | None
    chart_rows: list[RawCandle]
    signal_rows: list[RawCandle]
    technical_context: dict[str, Any]
    current_bar: dict[str, float | str | None]


@dataclass(slots=True)
class TechnicalSignal:
    symbol: str
    interval: str
    timestamp: datetime
    action: str
    bias: str
    score: float
    confidence: float
    conviction: str
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    cooldown_seconds: int
    max_signals_reached: bool
    reasons: list[str]
    details: dict[str, Any]


@dataclass(slots=True)
class OptionSelection:
    expiry_date: date
    strike_step: int
    chain_source: str
    chain_generated_at: datetime | None
    available_expiries: list[date]
    chain_rows: list[dict[str, Any]]
    signal: dict[str, Any]


def _disabled_signal(
    context: MarketContext,
    *,
    now: datetime | None = None,
    reason: str | None = None,
) -> TechnicalSignal:
    timestamp = _ensure_ist(context.latest_candle_ts) or _ensure_ist(now) or datetime.now(IST_ZONE)
    latest_price = round(float(context.latest_price or 0.0), 2)
    return TechnicalSignal(
        symbol=context.symbol,
        interval=LIVE_INTERVAL,
        timestamp=timestamp,
        action="HOLD",
        bias="NEUTRAL",
        score=0.0,
        confidence=0.0,
        conviction="disabled",
        entry_price=latest_price,
        stop_loss=None,
        take_profit=None,
        cooldown_seconds=0,
        max_signals_reached=False,
        reasons=[reason or "Directional BUY and SELL signals are disabled. The app is running in data-only mode."],
        details={
            "signals_enabled": False,
            "signal_mode": "data_only",
            "latest_price": latest_price,
            "signal_candle_ts": _ensure_ist(context.latest_candle_ts).isoformat() if context.latest_candle_ts else None,
        },
    )


def _ensure_ist(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.astimezone(IST_ZONE) if dt.tzinfo is not None else dt.replace(tzinfo=IST_ZONE)


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _parse_time(value: str, fallback: time) -> time:
    try:
        hour, minute = str(value).split(":", 1)
        return time(int(hour), int(minute))
    except Exception:
        return fallback


def _parse_iso_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _ensure_ist(value)
    if not value:
        return None
    try:
        return _ensure_ist(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


def _ns_between(later: datetime | None, earlier: datetime | None) -> int | None:
    if later is None or earlier is None:
        return None
    return max(0, int((later - earlier).total_seconds() * 1_000_000_000))


def default_symbol(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    symbols = settings.execution_symbol_list
    return symbols[0] if symbols else "Nifty 50"


def list_symbols(db: Session, settings: Settings | None = None) -> list[str]:
    settings = settings or get_settings()
    instrument_keys = db.scalars(
        select(RawCandle.instrument_key).distinct().order_by(RawCandle.instrument_key.asc())
    ).all()
    configured = settings.instrument_keys
    symbols = [display_symbol_from_instrument_key(key) for key in [*instrument_keys, *configured]]
    ordered = sort_display_symbols(symbols)
    return ordered or [default_symbol(settings)]


def resolve_instrument_key(db: Session, symbol: str) -> tuple[str, str]:
    if "|" in symbol:
        return symbol, canonical_symbol_name(display_symbol_from_instrument_key(symbol))
    cache_key = normalize_symbol_key(symbol)
    cached = _INSTRUMENT_RESOLVE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    aliases = symbol_aliases(symbol)
    exact_candidates: list[str] = []
    for alias in aliases:
        exact_candidates.extend((alias, f"NSE_INDEX|{alias}", f"BSE_INDEX|{alias}"))
    key = db.scalar(
        select(RawCandle.instrument_key)
        .where(RawCandle.instrument_key.in_(exact_candidates))
        .order_by(RawCandle.instrument_key.asc())
        .limit(1)
    )
    if key is not None:
        resolved = (str(key), canonical_symbol_name(display_symbol_from_instrument_key(str(key))))
        _INSTRUMENT_RESOLVE_CACHE[cache_key] = resolved
        return resolved
    key = db.scalar(
        select(RawCandle.instrument_key)
        .where(instrument_key_filter(RawCandle.instrument_key, symbol))
        .order_by(RawCandle.instrument_key.asc())
        .limit(1)
    )
    if key is None:
        raise ValueError(f"Symbol not found in candles: {symbol}")
    display = canonical_symbol_name(display_symbol_from_instrument_key(str(key)))
    resolved = (str(key), display)
    _INSTRUMENT_RESOLVE_CACHE[cache_key] = resolved
    return resolved


def _load_recent_candles(
    db: Session,
    *,
    instrument_key: str,
    interval: str = LIVE_INTERVAL,
    limit: int = 240,
) -> list[RawCandle]:
    rows = (
        db.execute(
            select(RawCandle)
            .where(and_(RawCandle.instrument_key == instrument_key, RawCandle.interval == interval))
            .order_by(RawCandle.ts.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    rows.reverse()
    return rows


def _candles_to_frame(rows: list[RawCandle]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [_ensure_ist(row.ts) for row in rows],
            "open": [float(row.open) for row in rows],
            "high": [float(row.high) for row in rows],
            "low": [float(row.low) for row in rows],
            "close": [float(row.close) for row in rows],
            "volume": [float(row.volume or 0.0) for row in rows],
        }
    )


def _resample_frame(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    data["ts"] = pd.to_datetime(data["ts"])
    resampled = (
        data.set_index("ts")
        .resample(rule, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    return build_price_features(resampled) if not resampled.empty else resampled


def _resample_chart_frame(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    data["ts"] = pd.to_datetime(data["ts"])
    resampled = (
        data.set_index("ts")
        .resample(rule, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    return resampled


def _timeframe_confirmation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ts", "confirm_buy", "confirm_sell"])
    data = frame.copy()
    close = pd.to_numeric(data.get("close"), errors="coerce")
    ema_21 = pd.to_numeric(data.get("ema_21"), errors="coerce").fillna(close)
    ema_50 = pd.to_numeric(data.get("ema_50"), errors="coerce").fillna(close)
    vwap = pd.to_numeric(data.get("vwap"), errors="coerce").fillna(close)
    rsi = pd.to_numeric(data.get("rsi_14"), errors="coerce").fillna(50.0)
    macd_hist = pd.to_numeric(data.get("macd_hist"), errors="coerce").fillna(0.0)
    data["confirm_buy"] = (
        (close > ema_21)
        & (ema_21 > ema_50)
        & (close >= vwap)
        & (rsi >= 54.0)
        & (macd_hist >= 0.0)
    )
    data["confirm_sell"] = (
        (close < ema_21)
        & (ema_21 < ema_50)
        & (close <= vwap)
        & (rsi <= 46.0)
        & (macd_hist <= 0.0)
    )
    return data[["ts", "confirm_buy", "confirm_sell"]]


def _expected_chart_end_date(now: datetime | None = None, latest_available_ts: datetime | None = None) -> date:
    current = _ensure_ist(now) or datetime.now(IST_ZONE)
    today = current.date()
    session_start, _session_end = market_session_bounds(today)
    if not is_trading_day(today):
        expected = previous_trading_day(today)
    elif current < session_start:
        expected = previous_trading_day(today)
    else:
        expected = today
    latest_date = _ensure_ist(latest_available_ts).date() if latest_available_ts is not None else None
    if latest_date is not None and latest_date < expected:
        return latest_date
    return expected


def _chart_range_plan(
    range_key: str,
    now: datetime | None = None,
    latest_available_ts: datetime | None = None,
    earliest_available_ts: datetime | None = None,
    interval_override: str | None = None,
) -> dict[str, Any]:
    current = _ensure_ist(now) or datetime.now(IST_ZONE)
    key = str(range_key or DEFAULT_CHART_RANGE).strip().lower()
    if key not in CHART_RANGE_SPECS:
        try:
            interval = normalize_interval(key)
            key = str(range_key).strip().lower()
            CHART_RANGE_SPECS[key] = {
                "label": key.upper(),
                "interval": interval,
                "from_availability": True,
                "supports_live": interval == LIVE_INTERVAL,
            }
        except ValueError:
            key = DEFAULT_CHART_RANGE
    spec = dict(CHART_RANGE_SPECS[key])
    if interval_override:
        override = normalize_interval(interval_override)
        spec["interval"] = override
        spec["supports_live"] = override == LIVE_INTERVAL
    end_date = _expected_chart_end_date(current, latest_available_ts=latest_available_ts)
    earliest = _ensure_ist(earliest_available_ts)
    if bool(spec.get("all_history")) and earliest is not None:
        start_date = earliest.date()
    elif bool(spec.get("from_availability")):
        start_date = end_date - timedelta(days=_default_chart_days_for_interval(str(spec["interval"])))
    elif "years" in spec:
        start_date = end_date - relativedelta(years=int(spec["years"]))
    else:
        start_date = end_date - timedelta(days=max(1, int(spec.get("days", 1))) - 1)
    if not is_trading_day(start_date):
        start_date = next_trading_day(start_date)
    start_ts = datetime.combine(start_date, time.min, tzinfo=IST_ZONE)
    if str(spec["interval"]).endswith("minute") or str(spec["interval"]).endswith("hour"):
        start_ts = datetime.combine(start_date, time(9, 15), tzinfo=IST_ZONE)
    if bool(spec.get("all_history")) and earliest is not None:
        start_ts = earliest
    return {
        "key": key,
        "label": spec["label"],
        "interval": spec["interval"],
        "supports_live": bool(spec["supports_live"]),
        "start_date": start_date,
        "end_date": end_date,
        "start_ts": start_ts,
        "end_ts": min(current, datetime.combine(end_date, time.max, tzinfo=IST_ZONE)),
    }


def _chart_rows_from_range(
    db: Session,
    *,
    instrument_key: str,
    interval: str,
    start_ts: datetime,
    end_ts: datetime,
) -> list[dict[str, Any]]:
    target_rows = (
        db.execute(
            select(
                RawCandle.ts,
                RawCandle.open,
                RawCandle.high,
                RawCandle.low,
                RawCandle.close,
                RawCandle.volume,
            )
            .where(
                and_(
                    RawCandle.instrument_key == instrument_key,
                    RawCandle.interval == interval,
                    RawCandle.ts >= start_ts,
                    RawCandle.ts <= end_ts,
                )
            )
            .order_by(RawCandle.ts.asc())
        )
        .all()
    )
    if target_rows:
        return [
            {
                "ts": _ensure_ist(row.ts),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume or 0.0),
            }
            for row in target_rows
        ]

    base_rows = (
        db.execute(
            select(
                RawCandle.ts,
                RawCandle.open,
                RawCandle.high,
                RawCandle.low,
                RawCandle.close,
                RawCandle.volume,
            )
            .where(
                and_(
                    RawCandle.instrument_key == instrument_key,
                    RawCandle.interval == LIVE_INTERVAL,
                    RawCandle.ts >= start_ts,
                    RawCandle.ts <= end_ts,
                )
            )
            .order_by(RawCandle.ts.asc())
        )
        .all()
    )
    if interval == LIVE_INTERVAL:
        return [
            {
                "ts": _ensure_ist(row.ts),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume or 0.0),
            }
            for row in base_rows
        ]
    raw_frame = _candles_to_frame(base_rows)
    rule = _pandas_rule_for_interval(interval)
    resampled = _resample_chart_frame(raw_frame, rule)
    return [
        {
            "ts": _ensure_ist(row.ts.to_pydatetime() if hasattr(row.ts, "to_pydatetime") else row.ts),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume or 0.0),
        }
        for row in resampled.itertuples(index=False)
    ]


def _chart_range_options() -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "label": str(spec["label"]),
            "interval": str(spec["interval"]),
            "supports_live": bool(spec["supports_live"]),
        }
        for key, spec in CHART_RANGE_SPECS.items()
    ]


def _chart_interval_options() -> list[dict[str, str]]:
    return [dict(item) for item in CHART_INTERVAL_OPTIONS]


def _default_chart_days_for_interval(interval: str) -> int:
    if interval.endswith("minute"):
        value = int(interval.removesuffix("minute"))
        if value <= 5:
            return 7
        if value <= 15:
            return 31
        if value <= 75:
            return 92
        return 366
    if interval.endswith("hour"):
        return 183
    if interval == "day":
        return 366
    if interval == "week":
        return 366 * 2
    return 366 * 5


def _latest_complete_intraday_ts(
    db: Session,
    *,
    instrument_key: str,
    min_rows: int = 50,
    now: datetime | None = None,
) -> datetime | None:
    rows = db.scalars(
        select(RawCandle.ts)
        .where(
            and_(
                RawCandle.instrument_key == instrument_key,
                RawCandle.interval == LIVE_INTERVAL,
            )
        )
        .order_by(RawCandle.ts.desc())
        .limit(max(500, int(min_rows) * 20))
    ).all()
    latest_ts = _ensure_ist(rows[0]) if rows else None
    current = _ensure_ist(now) or datetime.now(IST_ZONE)
    session_start, session_end = market_session_bounds(current.date())
    if (
        latest_ts is not None
        and latest_ts.date() == current.date()
        and is_trading_day(current.date())
        and session_start <= current <= session_end
    ):
        return latest_ts

    sessions: dict[date, list[datetime]] = {}
    for raw_ts in rows:
        ts = _ensure_ist(raw_ts)
        if ts is None:
            continue
        session_rows = sessions.setdefault(ts.date(), [])
        session_rows.append(ts)
        if len(session_rows) >= max(1, int(min_rows)):
            return max(session_rows)
    return None


def _pandas_rule_for_interval(interval: str) -> str:
    normalized = normalize_interval(interval)
    if normalized.endswith("minute"):
        return f"{int(normalized.removesuffix('minute'))}min"
    if normalized.endswith("hour"):
        return f"{int(normalized.removesuffix('hour'))}h"
    if normalized == "day":
        return "1D"
    if normalized == "week":
        return "1W"
    if normalized == "month":
        return "1ME"
    raise ValueError(f"Unsupported interval={interval}")


def _interval_bucket_minutes(interval: str) -> int:
    normalized = normalize_interval(interval)
    if normalized.endswith("minute"):
        return int(normalized.removesuffix("minute"))
    if normalized.endswith("hour"):
        return int(normalized.removesuffix("hour")) * 60
    if normalized == "day":
        return 390
    if normalized == "week":
        return 390 * 5
    if normalized == "month":
        return 390 * 22
    return 1


def _serialize_candle_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "x": row["ts"].isoformat() if row.get("ts") is not None else None,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume") or 0.0),
        }
        for row in source_rows
    ]


def _parse_chart_boundary(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return _ensure_ist(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return _ensure_ist(datetime.fromisoformat(raw.replace("Z", "+00:00")))
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp={value}") from exc


def _chart_cache_symbol_key(symbol: str) -> str:
    return "".join(ch for ch in str(symbol or "").upper() if ch.isalnum()) or "SYMBOL"


def _chart_interval_key(interval: str) -> str:
    normalized = normalize_interval(interval)
    if normalized.endswith("minute"):
        return f"{normalized.removesuffix('minute')}m"
    if normalized.endswith("hour"):
        return f"{normalized.removesuffix('hour')}H"
    if normalized == "day":
        return "1D"
    if normalized == "week":
        return "1W"
    if normalized == "month":
        return "1ME"
    return normalized


def load_candles_payload(
    db: Session,
    *,
    symbol: str,
    interval: str = LIVE_INTERVAL,
    before: str | datetime | None = None,
    after: str | datetime | None = None,
    limit: int = 500,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    normalized_interval = normalize_interval(interval)
    max_candle_limit = max(5000, int(getattr(settings, "chart_max_candle_limit", 500_000)))
    limit_value = max(1, min(int(limit or 500), max_candle_limit))
    before_ts = _parse_chart_boundary(before)
    after_ts = _parse_chart_boundary(after)
    if before_ts is not None and after_ts is not None:
        raise ValueError("Use either before or after, not both")

    instrument_key, display_symbol = resolve_instrument_key(db, symbol)
    available_count: int | None = None
    direction = "history" if before_ts is not None else ("forward" if after_ts is not None else "latest")
    cache_symbol = _chart_cache_symbol_key(display_symbol)
    interval_key = _chart_interval_key(normalized_interval)
    simple_key = f"{cache_symbol}_{interval_key}"
    boundary_key = (before_ts or after_ts).isoformat() if (before_ts or after_ts) is not None else "latest"
    redis_key = f"{simple_key}:{direction}:{limit_value}:{boundary_key}"
    redis_cached = redis_get_json(redis_key)
    if redis_cached is not None:
        return redis_cached

    source_limit = limit_value
    query_interval = normalized_interval
    query = select(
        RawCandle.ts,
        RawCandle.open,
        RawCandle.high,
        RawCandle.low,
        RawCandle.close,
        RawCandle.volume,
    ).where(and_(RawCandle.instrument_key == instrument_key, RawCandle.interval == query_interval))
    if before_ts is not None:
        query = query.where(RawCandle.ts < before_ts).order_by(RawCandle.ts.desc()).limit(source_limit)
        raw_rows = list(reversed(db.execute(query).all()))
    elif after_ts is not None:
        query = query.where(RawCandle.ts > after_ts).order_by(RawCandle.ts.asc()).limit(source_limit)
        raw_rows = db.execute(query).all()
    else:
        query = query.order_by(RawCandle.ts.desc()).limit(source_limit)
        raw_rows = list(reversed(db.execute(query).all()))

    base_rows = [
        {
            "ts": _ensure_ist(row.ts),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume or 0.0),
        }
        for row in raw_rows
    ]
    if not base_rows and normalized_interval != LIVE_INTERVAL:
        base_query = select(
            RawCandle.ts,
            RawCandle.open,
            RawCandle.high,
            RawCandle.low,
            RawCandle.close,
            RawCandle.volume,
        ).where(and_(RawCandle.instrument_key == instrument_key, RawCandle.interval == LIVE_INTERVAL))
        if before_ts is not None:
            base_query = base_query.where(RawCandle.ts < before_ts).order_by(RawCandle.ts.desc()).limit(source_limit)
            raw_rows = list(reversed(db.execute(base_query).all()))
        elif after_ts is not None:
            base_query = base_query.where(RawCandle.ts > after_ts).order_by(RawCandle.ts.asc()).limit(source_limit)
            raw_rows = db.execute(base_query).all()
        else:
            base_query = base_query.order_by(RawCandle.ts.desc()).limit(source_limit)
            raw_rows = list(reversed(db.execute(base_query).all()))
        raw_frame = _candles_to_frame(raw_rows)
        resampled = _resample_chart_frame(raw_frame, _pandas_rule_for_interval(normalized_interval))
        base_rows = [
            {
                "ts": _ensure_ist(row.ts.to_pydatetime() if hasattr(row.ts, "to_pydatetime") else row.ts),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume or 0.0),
            }
            for row in resampled.itertuples(index=False)
        ]
    rows = base_rows

    if before_ts is not None:
        rows = [row for row in rows if row["ts"] is not None and row["ts"] < before_ts][-limit_value:]
    elif after_ts is not None:
        rows = [row for row in rows if row["ts"] is not None and row["ts"] > after_ts][:limit_value]
    else:
        rows = rows[-limit_value:]

    candles = _serialize_candle_rows(rows)
    payload = {
        "symbol": display_symbol,
        "instrument_key": instrument_key,
        "interval": normalized_interval,
        "cache_key": simple_key,
        "direction": direction,
        "limit": limit_value,
        "count": len(candles),
        "available_count": available_count,
        "oldest": candles[0]["x"] if candles else None,
        "latest": candles[-1]["x"] if candles else None,
        "candles": candles,
        "available_intervals": _chart_interval_options(),
    }
    redis_set_json(
        redis_key,
        payload,
        ttl_seconds=max(1, int(getattr(settings, "redis_chart_cache_ttl_seconds", 900))),
    )
    return payload


def _latest_chart_source_ts(
    db: Session,
    *,
    instrument_key: str,
    interval: str,
) -> datetime | None:
    latest = db.scalar(
        select(func.max(RawCandle.ts)).where(
            and_(
                RawCandle.instrument_key == instrument_key,
                RawCandle.interval == interval,
            )
        )
    )
    if latest is not None:
        return _ensure_ist(latest)
    if interval != LIVE_INTERVAL:
        fallback = db.scalar(
            select(func.max(RawCandle.ts)).where(
                and_(
                    RawCandle.instrument_key == instrument_key,
                    RawCandle.interval == LIVE_INTERVAL,
                )
            )
        )
        return _ensure_ist(fallback)
    return None


def _earliest_chart_source_ts(
    db: Session,
    *,
    instrument_key: str,
    interval: str,
) -> datetime | None:
    earliest = db.scalar(
        select(func.min(RawCandle.ts)).where(
            and_(
                RawCandle.instrument_key == instrument_key,
                RawCandle.interval == interval,
            )
        )
    )
    if earliest is not None:
        return _ensure_ist(earliest)
    if interval != LIVE_INTERVAL:
        fallback = db.scalar(
            select(func.min(RawCandle.ts)).where(
                and_(
                    RawCandle.instrument_key == instrument_key,
                    RawCandle.interval == LIVE_INTERVAL,
                )
            )
        )
        return _ensure_ist(fallback)
    return None


def _build_pine_chart_overlay(
    rows: list[dict[str, Any]],
    *,
    interval: str,
    settings: Settings,
    range_key: str,
) -> dict[str, Any]:
    if not DIRECTIONAL_SIGNALS_ENABLED:
        return {"markers": [], "levels": []}
    if len(rows) < 60:
        return {"markers": [], "levels": []}
    if len(rows) > 20_000:
        rows = rows[-20_000:]

    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"])
    if frame.empty or len(frame) < 60:
        return {"markers": [], "levels": []}

    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame.get("volume", 0.0), errors="coerce").fillna(0.0)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    def rma(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(alpha=1.0 / max(1, int(length)), adjust=False).mean()

    sensitivity = float(getattr(settings, "pine_signal_sensitivity", 1.0))
    atr_length = int(getattr(settings, "pine_signal_atr_length", 10))
    atr_multiplier = float(getattr(settings, "pine_signal_atr_multiplier", 7.0))
    use_trend_filter = bool(getattr(settings, "pine_signal_use_trend_filter", True))
    ma_length = int(getattr(settings, "pine_signal_ma_length", 20))
    use_volume_filter = bool(getattr(settings, "pine_signal_use_volume_filter", False))
    volume_threshold = float(getattr(settings, "pine_signal_volume_threshold", 1.1))
    show_signals = bool(getattr(settings, "pine_signal_show_signals", True))
    signal_cooldown = int(getattr(settings, "pine_signal_cooldown_bars", 2))
    atr_risk = int(getattr(settings, "pine_signal_atr_risk", 3))
    risk_atr_length = int(getattr(settings, "pine_signal_risk_atr_length", 14))
    percent_stop = float(getattr(settings, "pine_signal_percent_stop", 1.0))

    if not show_signals:
        return {"markers": [], "levels": []}

    atr = rma(true_range, atr_length)
    factor = sensitivity * atr_multiplier
    upper_band = close + (factor * atr)
    lower_band = close - (factor * atr)

    supertrend_line: list[float | None] = [None] * len(frame)
    direction: list[int | None] = [None] * len(frame)
    final_upper: list[float | None] = [None] * len(frame)
    final_lower: list[float | None] = [None] * len(frame)

    for index in range(len(frame)):
        if pd.isna(atr.iloc[index]) or pd.isna(close.iloc[index]):
            continue

        current_upper = float(upper_band.iloc[index])
        current_lower = float(lower_band.iloc[index])
        if index > 0:
            prev_lower = final_lower[index - 1]
            prev_upper = final_upper[index - 1]
            prev_close_value = close.iloc[index - 1]
            if prev_lower is not None and not pd.isna(prev_close_value):
                current_lower = current_lower if current_lower > prev_lower or float(prev_close_value) < prev_lower else prev_lower
            if prev_upper is not None and not pd.isna(prev_close_value):
                current_upper = current_upper if current_upper < prev_upper or float(prev_close_value) > prev_upper else prev_upper

        final_upper[index] = current_upper
        final_lower[index] = current_lower

        if index == 0 or pd.isna(atr.iloc[index - 1]) or supertrend_line[index - 1] is None:
            current_direction = 1
        elif supertrend_line[index - 1] == final_upper[index - 1]:
            current_direction = -1 if float(close.iloc[index]) > current_upper else 1
        else:
            current_direction = 1 if float(close.iloc[index]) < current_lower else -1

        direction[index] = current_direction
        supertrend_line[index] = current_lower if current_direction == -1 else current_upper

    ma = close.rolling(max(1, ma_length), min_periods=max(1, ma_length)).mean()
    volume_ma = volume.rolling(20, min_periods=20).mean()
    atr_14 = rma(true_range, 14)
    atr_band = rma(true_range, risk_atr_length) * atr_risk

    markers: list[dict[str, Any]] = []
    levels: list[dict[str, Any]] = []
    last_signal_bar = 0
    last_signal_type = ""

    for index in range(1, len(frame)):
        line = supertrend_line[index]
        prev_line = supertrend_line[index - 1]
        if line is None or prev_line is None:
            continue

        current_close = close.iloc[index]
        previous_close = close.iloc[index - 1]
        if pd.isna(current_close) or pd.isna(previous_close):
            continue

        raw_buy = float(previous_close) <= float(prev_line) and float(current_close) > float(line)
        raw_sell = float(previous_close) >= float(prev_line) and float(current_close) < float(line)

        trend_up = bool(not pd.isna(ma.iloc[index]) and float(current_close) > float(ma.iloc[index]))
        trend_down = bool(not pd.isna(ma.iloc[index]) and float(current_close) < float(ma.iloc[index]))
        trend_filter_buy = (not use_trend_filter) or trend_up
        trend_filter_sell = (not use_trend_filter) or trend_down

        volume_ok = True
        if use_volume_filter:
            volume_ok = bool(
                not pd.isna(volume_ma.iloc[index])
                and float(volume.iloc[index]) > (float(volume_ma.iloc[index]) * volume_threshold)
            )

        momentum_ok = bool(
            not pd.isna(atr_14.iloc[index])
            and abs(float(current_close) - float(previous_close)) > (float(atr_14.iloc[index]) * 0.1)
        )

        buy_signal = raw_buy and trend_filter_buy and volume_ok and momentum_ok
        sell_signal = raw_sell and trend_filter_sell and volume_ok and momentum_ok

        bars_since_last = index - last_signal_bar
        cooldown_ok = bars_since_last >= signal_cooldown
        final_buy = buy_signal and cooldown_ok and last_signal_type != "BUY"
        final_sell = sell_signal and cooldown_ok and last_signal_type != "SELL"
        if not final_buy and not final_sell:
            continue

        action = "BUY" if final_buy else "SELL"
        last_signal_bar = index
        last_signal_type = action
        ts = _ensure_ist(frame.iloc[index]["ts"].to_pydatetime() if hasattr(frame.iloc[index]["ts"], "to_pydatetime") else frame.iloc[index]["ts"])
        if ts is None:
            continue

        if percent_stop != 0 and not pd.isna(atr_band.iloc[index]):
            entry = float(current_close)
            stop = (
                float(low.iloc[index]) - float(atr_band.iloc[index])
                if action == "BUY"
                else float(high.iloc[index]) + float(atr_band.iloc[index])
            )
            risk_unit = entry - stop
            levels = [
                {"label": "ENTRY", "price": round(entry, 4), "color": "#f8fafc", "lineStyle": "solid"},
                {"label": "STOP LOSS", "price": round(stop, 4), "color": "#ef4444", "lineStyle": "solid"},
                {"label": "TP 1", "price": round(entry + (risk_unit * 1), 4), "color": "#22c55e", "lineStyle": "dotted"},
                {"label": "TP 2", "price": round(entry + (risk_unit * 2), 4), "color": "#22c55e", "lineStyle": "dotted"},
                {"label": "TP 3", "price": round(entry + (risk_unit * 3), 4), "color": "#22c55e", "lineStyle": "dotted"},
                {"label": "TP 4", "price": round(entry + (risk_unit * 4), 4), "color": "#22c55e", "lineStyle": "dotted"},
                {"label": "TP 5", "price": round(entry + (risk_unit * 5), 4), "color": "#22c55e", "lineStyle": "dotted"},
            ]

        markers.append(
            {
                "time": ts.isoformat(),
                "position": "belowBar" if action == "BUY" else "aboveBar",
                "color": "#16a34a" if action == "BUY" else "#dc2626",
                "shape": "arrowUp" if action == "BUY" else "arrowDown",
                "text": action,
            }
        )
    limit = max(1, CHART_MARKER_LIMITS.get(range_key, 40))
    return {"markers": markers[-limit:], "levels": levels}


def _build_chart_markers(
    rows: list[dict[str, Any]],
    *,
    interval: str,
    settings: Settings,
    range_key: str,
) -> list[dict[str, Any]]:
    return list(_build_pine_chart_overlay(rows, interval=interval, settings=settings, range_key=range_key)["markers"])


def _latest_fresh_pine_marker(
    rows: list[RawCandle],
    *,
    settings: Settings,
    candle_ts: datetime,
) -> dict[str, Any] | None:
    candle_ts = _ensure_ist(candle_ts)
    if candle_ts is None:
        return None
    session_start, session_end = market_session_bounds(candle_ts.date())
    session_rows = [
        row
        for row in rows
        if (
            _ensure_ist(row.ts) is not None
            and session_start <= _ensure_ist(row.ts) < session_end
        )
    ]
    overlay_rows = _candles_to_frame(session_rows).to_dict("records")
    overlay = _build_pine_chart_overlay(
        overlay_rows,
        interval=SIGNAL_INTERVAL,
        settings=settings,
        range_key="all",
    )
    markers = list(overlay.get("markers") or [])
    if not markers:
        return None
    latest_marker = markers[-1]
    marker_ts = _parse_iso_datetime(latest_marker.get("time"))
    if marker_ts is None:
        return None
    if marker_ts.replace(second=0, microsecond=0) != candle_ts.replace(second=0, microsecond=0):
        return None
    action = str(latest_marker.get("text") or "").upper()
    return latest_marker if action in {"BUY", "SELL"} else None


def _closed_signal_rows(rows: list[RawCandle], now: datetime) -> list[RawCandle]:
    if len(rows) <= 1:
        return rows
    latest_ts = _ensure_ist(rows[-1].ts)
    if latest_ts is None:
        return rows
    current_minute = now.replace(second=0, microsecond=0)
    if latest_ts >= current_minute:
        return rows[:-1]
    return rows


def load_market_context(
    db: Session,
    *,
    symbol: str,
    settings: Settings | None = None,
    chart_limit: int = 180,
    signal_limit: int = 240,
    now: datetime | None = None,
) -> MarketContext:
    settings = settings or get_settings()
    now = _ensure_ist(now) or datetime.now(IST_ZONE)
    instrument_key, display_symbol = resolve_instrument_key(db, symbol)
    rows = _load_recent_candles(
        db,
        instrument_key=instrument_key,
        interval=LIVE_INTERVAL,
        limit=max(chart_limit, signal_limit),
    )
    if not rows:
        raise ValueError(f"No {LIVE_INTERVAL} candles found for {display_symbol}")

    signal_rows = _closed_signal_rows(rows, now)
    working_rows = signal_rows[-signal_limit:] if len(signal_rows) > signal_limit else signal_rows
    frame = _candles_to_frame(working_rows)
    features = build_price_features(frame) if len(frame) >= 30 else frame
    technical_context = features.iloc[-1].to_dict() if not features.empty else {}
    current = rows[-1]
    current_bar = {
        "open": float(current.open),
        "high": float(current.high),
        "low": float(current.low),
        "close": float(current.close),
        "volume": float(current.volume or 0.0),
    }
    return MarketContext(
        symbol=display_symbol,
        instrument_key=instrument_key,
        latest_price=float(current.close),
        latest_candle_ts=_ensure_ist(current.ts),
        chart_rows=rows[-chart_limit:],
        signal_rows=working_rows,
        technical_context=technical_context,
        current_bar=current_bar,
    )


def _timeframe_confirmation(frame: pd.DataFrame) -> tuple[bool, bool]:
    if frame.empty or len(frame) < 20:
        return False, False
    row = frame.iloc[-1]
    close = _to_float(row.get("close")) or 0.0
    ema_21 = _to_float(row.get("ema_21")) or close
    ema_50 = _to_float(row.get("ema_50")) or close
    vwap = _to_float(row.get("vwap")) or close
    rsi = _to_float(row.get("rsi_14")) or 50.0
    macd_hist = _to_float(row.get("macd_hist")) or 0.0
    bullish = close > ema_21 > ema_50 and close >= vwap and rsi >= 54.0 and macd_hist >= 0.0
    bearish = close < ema_21 < ema_50 and close <= vwap and rsi <= 46.0 and macd_hist <= 0.0
    return bullish, bearish


def _signal_guardrails(
    db: Session,
    *,
    symbol: str,
    now: datetime,
    settings: Settings,
) -> tuple[int, int]:
    today = now.date()
    count = db.scalar(
        select(func.count(SignalLog.id)).where(
            and_(
                SignalLog.trade_date == today,
                symbol_value_filter(SignalLog.symbol, symbol),
                SignalLog.consensus.in_(["BUY", "SELL"]),
            )
        )
    )
    latest = db.scalar(
        select(SignalLog.timestamp)
        .where(
            and_(
                symbol_value_filter(SignalLog.symbol, symbol),
                SignalLog.consensus.in_(["BUY", "SELL"]),
            )
        )
        .order_by(SignalLog.timestamp.desc())
        .limit(1)
    )
    cooldown_minutes = max(1, int(getattr(settings, "signal_cooldown_minutes", DEFAULT_SIGNAL_COOLDOWN_MINUTES)))
    cooldown_seconds = 0
    latest_ist = _ensure_ist(latest)
    if latest_ist is not None:
        elapsed = int((now - latest_ist).total_seconds())
        cooldown_seconds = max(0, (cooldown_minutes * 60) - elapsed)
    return int(count or 0), cooldown_seconds


def _parse_time_setting(value: str, fallback: time) -> time:
    try:
        hour, minute = str(value).split(":", 1)
        return time(int(hour), int(minute))
    except Exception:
        return fallback


def _strategy_window_status(now: datetime, settings: Settings | None = None) -> tuple[str, bool]:
    settings = settings or get_settings()
    entry_start = _parse_time_setting(getattr(settings, "entry_window_start", ""), SIGNAL_ENTRY_START)
    entry_end = _parse_time_setting(getattr(settings, "entry_window_end", ""), SIGNAL_ENTRY_END)
    now_time = now.timetz().replace(tzinfo=None)
    if now_time < entry_start:
        return "avoid_open", False
    if now_time > entry_end:
        if now_time < time(15, 30):
            return "avoid_close", False
        return "market_closed", False
    if now_time < time(11, 30):
        return "best_window", True
    if now_time < time(13, 30):
        return "slow_window", True
    return "good_window", True


def _resample_five_minute_signal_frame(rows: list[RawCandle]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    frame = _candles_to_frame(rows).copy()
    if frame.empty:
        return frame
    frame["ts"] = pd.to_datetime(frame["ts"])

    bucket_starts: list[datetime] = []
    for value in frame["ts"]:
        ts = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
        ts_ist = _ensure_ist(ts)
        if ts_ist is None:
            continue
        session_start, _ = market_session_bounds(ts_ist.date())
        minutes_from_open = max(0, int((ts_ist - session_start).total_seconds() // 60))
        bucket_index = minutes_from_open // 5
        bucket_starts.append(session_start + timedelta(minutes=bucket_index * 5))

    if len(bucket_starts) != len(frame):
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    frame["bucket_start"] = bucket_starts
    grouped = (
        frame.groupby("bucket_start", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            bar_count=("ts", "count"),
        )
    )
    grouped = grouped[grouped["bar_count"] >= 4].copy()  # Relaxed from 5 to 4 bars
    if grouped.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    grouped["ts"] = grouped["bucket_start"] + timedelta(minutes=5)
    return build_price_features(grouped[["ts", "open", "high", "low", "close", "volume"]])


def _nearest_atm_option_quote(
    chain_rows: list[dict[str, Any]],
    *,
    spot_price: float,
    strike_step: int,
    option_type: str,
) -> tuple[float, dict[str, Any]] | tuple[None, None]:
    requested_strike = nearest_strike(spot_price, strike_step)
    quote_key = "ce" if str(option_type).upper() == "CE" else "pe"
    candidates = [
        row for row in chain_rows
        if row.get("strike") is not None and (row.get(quote_key) or {}).get("ltp") is not None
    ]
    if not candidates:
        return None, None
    selected = min(candidates, key=lambda row: abs(float(row["strike"]) - requested_strike))
    return float(selected["strike"]), dict(selected.get(quote_key) or {})


def build_technical_signal(
    db: Session,
    *,
    context: MarketContext,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> TechnicalSignal:
    settings = settings or get_settings()
    now = _ensure_ist(now) or datetime.now(IST_ZONE)

    if not DIRECTIONAL_SIGNALS_ENABLED:
        return _disabled_signal(context, now=now)

    rows = context.signal_rows

    # --- Data freshness guard: halt if feed is stale ---
    if rows:
        last_ts = _ensure_ist(rows[-1].ts)
        if last_ts is not None:
            age_seconds = int((now - last_ts).total_seconds())
            if age_seconds > _DATA_STALE_SECONDS:
                return TechnicalSignal(
                    symbol=context.symbol,
                    interval=SIGNAL_INTERVAL,
                    timestamp=now,
                    action="HOLD",
                    bias="NEUTRAL",
                    score=0.0,
                    confidence=0.0,
                    conviction="low",
                    entry_price=context.latest_price,
                    stop_loss=None,
                    take_profit=None,
                    cooldown_seconds=0,
                    max_signals_reached=False,
                    reasons=[f"Data feed stale: last candle is {age_seconds}s old — no signal."],
                    details={
                        "data_stale": True,
                        "last_candle_age_seconds": age_seconds,
                        "strategy_interval": SIGNAL_INTERVAL,
                        "strategy_name": "ema_rsi_volume_breakout_1m",
                        "option_preference": "ATM_ONLY",
                    },
                )

    if len(rows) < 60:
        return TechnicalSignal(
            symbol=context.symbol,
            interval=SIGNAL_INTERVAL,
            timestamp=now,
            action="HOLD",
            bias="NEUTRAL",
            score=0.0,
            confidence=0.0,
            conviction="low",
            entry_price=context.latest_price,
            stop_loss=None,
            take_profit=None,
            cooldown_seconds=0,
            max_signals_reached=False,
            reasons=["Waiting for enough 1-minute candles to build a stable signal."],
            details={"warmup_candles": len(rows), "strategy_interval": SIGNAL_INTERVAL},
        )
    
    # Use 1-minute candles directly (no resampling)
    frame = _candles_to_frame(rows)
    signal_frame = build_price_features(frame) if len(frame) >= 30 else frame
    
    if len(signal_frame) < 25:
        return TechnicalSignal(
            symbol=context.symbol,
            interval=SIGNAL_INTERVAL,
            timestamp=now,
            action="HOLD",
            bias="NEUTRAL",
            score=0.0,
            confidence=0.0,
            conviction="low",
            entry_price=context.latest_price,
            stop_loss=None,
            take_profit=None,
            cooldown_seconds=0,
            max_signals_reached=False,
            reasons=["Waiting for enough 1-minute candles to build the EMA/RSI setup."],
            details={"strategy_interval": SIGNAL_INTERVAL, "warmup_candles_1m": len(signal_frame)},
        )

    row = signal_frame.iloc[-1]
    prev = signal_frame.iloc[-2]
    close = _to_float(row.get("close")) or context.latest_price
    open_ = _to_float(row.get("open")) or close
    high = _to_float(row.get("high")) or close
    low = _to_float(row.get("low")) or close
    prev_high = _to_float(prev.get("high")) or high
    prev_low = _to_float(prev.get("low")) or low

    ema_9 = _to_float(row.get("ema_9")) or close
    ema_21 = _to_float(row.get("ema_21")) or close
    prev_ema_9 = _to_float(prev.get("ema_9")) or ema_9
    prev_ema_21 = _to_float(prev.get("ema_21")) or ema_21
    rsi = _to_float(row.get("rsi_14")) or 50.0
    atr = max(1e-9, _to_float(row.get("atr_14")) or 0.0)
    volume = _to_float(row.get("volume")) or 0.0
    volume_avg = _to_float(row.get("volume_sma_20")) or 0.0
    volume_ratio = _to_float(row.get("volume_ratio_20")) or 1.0
    candle_ts = _parse_iso_datetime(row.get("ts")) or now
    window_status, entry_window_open = _strategy_window_status(now, settings)
    vix_level, vix_ma, vix_ratio = get_vix_context(db)
    vix_min = float(getattr(settings, "signal_vix_min", VIX_MIN_THRESHOLD))
    vix_max = float(getattr(settings, "signal_vix_max", VIX_MAX_THRESHOLD))
    vix_too_high = bool(vix_level is not None and vix_level > vix_max)
    vix_too_low = bool(vix_level is not None and vix_level < vix_min)

    ema_cross_up = prev_ema_9 <= prev_ema_21 and ema_9 > ema_21
    ema_cross_down = prev_ema_9 >= prev_ema_21 and ema_9 < ema_21
    bullish_candle = close > open_
    bearish_candle = close < open_
    rsi_buy_min = float(getattr(settings, "signal_rsi_buy_min", 52.0))
    rsi_sell_max = float(getattr(settings, "signal_rsi_sell_max", 48.0))
    rsi_buy_ok = rsi >= rsi_buy_min
    rsi_sell_ok = rsi <= rsi_sell_max
    min_volume_ratio = float(getattr(settings, "signal_min_volume_ratio", 1.15))
    volume_ok = (volume_ratio >= min_volume_ratio) if volume_avg > 0 else volume_ratio >= min_volume_ratio
    break_prev_high = close > prev_high
    break_prev_low = close < prev_low
    require_volume = bool(getattr(settings, "signal_require_volume_confirmation", True))
    require_breakout = bool(getattr(settings, "signal_require_breakout", True))
    atr_min = float(getattr(settings, "signal_atr_min_points", 4.0))
    atr_max = float(getattr(settings, "signal_atr_max_points", 80.0))
    atr_ok = atr_min <= atr <= atr_max

    # ============================================================================
    # 1-MINUTE SIGNAL SYSTEM
    # Fast execution on live 1-minute data
    # ============================================================================

    score_buy = 0.0
    score_sell = 0.0

    # 1. EMA TREND ALIGNMENT (40 points) - Most important
    #    Catches both crossovers AND trend continuation
    if ema_9 > ema_21:
        score_buy += 40.0
    if ema_9 < ema_21:
        score_sell += 40.0

    # 2. EMA CROSSOVER BONUS (20 points) - Extra points for fresh crossover
    if ema_cross_up:
        score_buy += 20.0
    if ema_cross_down:
        score_sell += 20.0

    # 3. RSI MOMENTUM (20 points) - More permissive range
    if rsi_buy_ok:
        score_buy += 20.0
    if rsi_sell_ok:
        score_sell += 20.0

    # 4. CANDLE DIRECTION (10 points) - Bonus, not required
    if bullish_candle:
        score_buy += 10.0
    if bearish_candle:
        score_sell += 10.0

    # 5. VOLUME CONFIRMATION (10 points) - Bonus, not required
    if volume_ok:
        score_buy += 10.0
        score_sell += 10.0

    # 6. BREAKOUT BONUS (10 points) - Extra points, not required
    if break_prev_high:
        score_buy += 10.0
    if break_prev_low:
        score_sell += 10.0

    min_score = float(getattr(settings, "signal_min_score", DEFAULT_SIGNAL_MIN_SCORE))
    buy_ready = (
        score_buy >= min_score
        and rsi_buy_ok
        and (volume_ok or not require_volume)
        and (break_prev_high or not require_breakout)
        and atr_ok
    )
    sell_ready = (
        score_sell >= min_score
        and rsi_sell_ok
        and (volume_ok or not require_volume)
        and (break_prev_low or not require_breakout)
        and atr_ok
    )

    fresh_marker = _latest_fresh_pine_marker(rows, settings=settings, candle_ts=candle_ts)
    pine_action = str((fresh_marker or {}).get("text") or "").upper()
    raw_action = pine_action if pine_action in {"BUY", "SELL"} else "HOLD"
    raw_score = 100.0 if raw_action in {"BUY", "SELL"} else max(score_buy, score_sell)
    bias = raw_action if raw_action in {"BUY", "SELL"} else (
        "BUY" if score_buy > score_sell else ("SELL" if score_sell > score_buy else "NEUTRAL")
    )

    max_signals_today = max(1, int(getattr(settings, "signal_max_per_day", DEFAULT_MAX_SIGNALS_PER_DAY)))
    signal_count_today, cooldown_seconds = _signal_guardrails(
        db, symbol=context.symbol, now=now, settings=settings,
    )
    max_reached = signal_count_today >= max_signals_today

    reasons: list[str] = []
    action = raw_action
    if raw_action == "BUY":
        reasons.extend(
            [
                "Fresh graph BUY marker printed on the latest closed 1-minute candle.",
                "Execution is using the same Pine-style signal logic as the chart markers.",
                "Trend filter passed for the graph marker.",
                "Momentum filter passed for the graph marker.",
            ]
        )
    elif raw_action == "SELL":
        reasons.extend(
            [
                "Fresh graph SELL marker printed on the latest closed 1-minute candle.",
                "Execution is using the same Pine-style signal logic as the chart markers.",
                "Trend filter passed for the graph marker.",
                "Momentum filter passed for the graph marker.",
            ]
        )
    else:
        reasons.append("No fresh graph BUY/SELL marker on the latest closed 1-minute candle.")
        if not ema_cross_up and not ema_cross_down:
            reasons.append("EMA 9 and EMA 21 have not crossed on the 1-minute candle.")
        if rsi_sell_max < rsi < rsi_buy_min:
            reasons.append(f"RSI is in the {rsi_sell_max:.0f}-{rsi_buy_min:.0f} neutral zone.")
        elif rsi >= rsi_buy_min:
            reasons.append("Momentum is bullish but EMA alignment is missing.")
        elif rsi <= rsi_sell_max:
            reasons.append("Momentum is bearish but EMA alignment is missing.")
        if not volume_ok:
            reasons.append("Signal candle volume is not above the 20-bar average.")
        if bullish_candle and not break_prev_high:
            reasons.append("Bullish candle did not close above the previous candle high.")
        if bearish_candle and not break_prev_low:
            reasons.append("Bearish candle did not close below the previous candle low.")
        if raw_score >= min_score:
            if require_volume and not volume_ok:
                reasons.append(f"Volume ratio {volume_ratio:.2f} is below required {min_volume_ratio:.2f}.")
            if require_breakout and score_buy >= score_sell and not break_prev_high:
                reasons.append("BUY setup rejected because close did not break previous high.")
            if require_breakout and score_sell > score_buy and not break_prev_low:
                reasons.append("SELL setup rejected because close did not break previous low.")
            if score_buy >= score_sell and not rsi_buy_ok:
                reasons.append(f"BUY setup rejected because RSI is below {rsi_buy_min:.0f}.")
            if score_sell > score_buy and not rsi_sell_ok:
                reasons.append(f"SELL setup rejected because RSI is above {rsi_sell_max:.0f}.")

    if not entry_window_open:
        action = "HOLD"
        reasons.append(
            f"Outside the strategy entry window of {settings.entry_window_start}-{settings.entry_window_end} IST."
        )
    if not atr_ok:
        action = "HOLD"
        reasons.append(f"ATR {atr:.2f} is outside the configured {atr_min:.2f}-{atr_max:.2f} point range.")
    if vix_too_high:
        action = "HOLD"
        reasons.append(f"VIX is above {vix_max:.1f}; skipping fresh option entries.")
    if vix_too_low:
        action = "HOLD"
        reasons.append(f"VIX is below {vix_min:.1f}; skipping weak premium expansion.")
    if cooldown_seconds > 0:
        action = "HOLD"
        reasons.append(f"Cooldown active — {cooldown_seconds}s remaining.")
    if max_reached:
        action = "HOLD"
        reasons.append(f"Daily signal cap reached ({signal_count_today}/{max_signals_today}).")

    stop_offset = max(atr, abs(close - prev_low), abs(prev_high - close))
    take_offset = round(stop_offset * 1.5, 2)
    if action == "BUY":
        stop_loss = round(min(prev_low, close - stop_offset), 2)
        take_profit = round(close + take_offset, 2)
    elif action == "SELL":
        stop_loss = round(max(prev_high, close + stop_offset), 2)
        take_profit = round(close - take_offset, 2)
    else:
        stop_loss = None
        take_profit = None

    confidence = round(_clip(raw_score / 100.0, 0.0, 0.95), 2)
    conviction = "high" if raw_score == 100.0 and action in {"BUY", "SELL"} else ("medium" if raw_score >= 60.0 else "low")
    expected_move = round(max(atr, abs(take_offset)), 2)
    expected_move_pct = round(expected_move / max(close, 1e-9), 4)

    details = {
        "strategy_name": "ema_rsi_volume_breakout_1m",
        "strategy_interval": SIGNAL_INTERVAL,
        "close": round(close, 2),
        "open": round(open_, 2),
        "high": round(high, 2),
        "low": round(low, 2),
        "prev_high": round(prev_high, 2),
        "prev_low": round(prev_low, 2),
        "ema_9": round(ema_9, 2),
        "ema_21": round(ema_21, 2),
        "prev_ema_9": round(prev_ema_9, 2),
        "prev_ema_21": round(prev_ema_21, 2),
        "rsi_14": round(rsi, 2),
        "atr_14": round(atr, 2),
        "volume": round(volume, 2),
        "volume_sma_20": round(volume_avg, 2),
        "volume_ratio_20": round(volume_ratio, 2),
        "min_volume_ratio": round(min_volume_ratio, 2),
        "ema_cross_up": ema_cross_up,
        "ema_cross_down": ema_cross_down,
        "rsi_buy_ok": rsi_buy_ok,
        "rsi_sell_ok": rsi_sell_ok,
        "rsi_buy_min": rsi_buy_min,
        "rsi_sell_max": rsi_sell_max,
        "bullish_candle": bullish_candle,
        "bearish_candle": bearish_candle,
        "volume_ok": volume_ok,
        "volume_required": require_volume,
        "break_prev_high": break_prev_high,
        "break_prev_low": break_prev_low,
        "breakout_required": require_breakout,
        "atr_ok": atr_ok,
        "atr_min_points": atr_min,
        "atr_max_points": atr_max,
        "buy_ready": buy_ready,
        "sell_ready": sell_ready,
        "pine_signal": raw_action if raw_action in {"BUY", "SELL"} else "OFF",
        "pine_marker_time": fresh_marker.get("time") if fresh_marker else None,
        "pine_marker_text": fresh_marker.get("text") if fresh_marker else None,
        "execution_signal_source": "graph_marker_pine_overlay",
        "fresh_graph_marker": bool(fresh_marker),
        "signal_min_score": min_score,
        "score_buy": round(score_buy, 1),
        "score_sell": round(score_sell, 1),
        "entry_window_start": settings.entry_window_start,
        "entry_window_end": settings.entry_window_end,
        "window_status": window_status,
        "entry_window_open": entry_window_open,
        "vix_level": vix_level,
        "vix_ma": vix_ma,
        "vix_ratio": round(vix_ratio, 3) if vix_ratio is not None else None,
        "vix_min": vix_min,
        "vix_max": vix_max,
        "vix_too_high": vix_too_high,
        "vix_too_low": vix_too_low,
        "option_preference": "ATM_ONLY",
        "option_stop_loss_pct": OPTION_STOP_LOSS_PCT,
        "option_target_pct": OPTION_TARGET_PCT,
        "option_trail_trigger_pct": OPTION_TRAIL_TRIGGER_PCT,
        "option_trail_stop_pct": OPTION_TRAIL_STOP_PCT,
        "expected_move_points": expected_move,
        "expected_move_pct": expected_move_pct,
        "signal_count_today": signal_count_today,
        "signal_candle_ts": candle_ts.isoformat(),
    }
    timestamp = candle_ts
    return TechnicalSignal(
        symbol=context.symbol,
        interval=SIGNAL_INTERVAL,
        timestamp=timestamp or now,
        action=action,
        bias=bias,
        score=round(raw_score, 1),
        confidence=confidence,
        conviction=conviction,
        entry_price=round(close, 2),
        stop_loss=stop_loss,
        take_profit=take_profit,
        cooldown_seconds=cooldown_seconds,
        max_signals_reached=max_reached,
        reasons=reasons[:6],
        details=details,
    )


def log_signal_decision(
    db: Session,
    *,
    signal: TechnicalSignal,
    trade_placed: bool = False,
    extra_details: dict[str, Any] | None = None,
    skip_reason: str | None = None,
) -> SignalLog:
    details = {**signal.details, **(extra_details or {})}
    row = SignalLog(
        timestamp=signal.timestamp,
        trade_date=signal.timestamp.date(),
        symbol=signal.symbol,
        interval=signal.interval,
        ml_signal=signal.bias,
        ml_confidence=signal.confidence,
        ml_expected_move=_to_float(signal.details.get("expected_move_points")),
        pine_signal=str(signal.details.get("pine_signal") or "OFF"),
        pine_age_seconds=None,
        ai_score=0.0,
        news_sentiment=0.0,
        combined_score=round(signal.score / 100.0, 4),
        consensus=signal.action if signal.action in {"BUY", "SELL"} else "non_trade_signal",
        skip_reason=skip_reason or (None if signal.action in {"BUY", "SELL"} else (signal.reasons[0] if signal.reasons else None)),
        trade_placed=bool(trade_placed),
        details=details,
    )
    db.add(row)
    db.flush()
    return row


def _load_option_quotes(
    db: Session,
    *,
    symbol: str,
    expiry_date: date,
    max_rows: int = 2000,
) -> list[OptionQuoteView]:
    rows = _load_latest_option_quote_rows(
        db,
        symbol=symbol,
        expiry_date=expiry_date,
        max_rows=max_rows,
    )
    return [
        OptionQuoteView(
            instrument_key=str(row.instrument_key),
            strike=float(row.strike),
            option_type=str(row.option_type),  # type: ignore[arg-type]
            ltp=float(row.ltp),
            bid=(float(row.bid) if row.bid is not None else None),
            ask=(float(row.ask) if row.ask is not None else None),
            volume=(float(row.volume) if row.volume is not None else None),
            oi=(float(row.oi) if row.oi is not None else None),
            close_price=(float(row.close_price) if row.close_price is not None else None),
            bid_qty=(float(row.bid_qty) if row.bid_qty is not None else None),
            ask_qty=(float(row.ask_qty) if row.ask_qty is not None else None),
            prev_oi=(float(row.prev_oi) if row.prev_oi is not None else None),
            iv=(float(row.iv) if row.iv is not None else None),
            delta=(float(row.delta) if row.delta is not None else None),
            gamma=(float(row.gamma) if row.gamma is not None else None),
            theta=(float(row.theta) if row.theta is not None else None),
            vega=(float(row.vega) if row.vega is not None else None),
            pop=(float(row.pop) if row.pop is not None else None),
            pcr=(float(row.pcr) if row.pcr is not None else None),
            underlying_spot_price=(
                float(row.underlying_spot_price) if row.underlying_spot_price is not None else None
            ),
            source=str(row.source or "db"),
        )
        for row in rows
    ]


def _load_latest_option_quote_rows(
    db: Session,
    *,
    symbol: str,
    expiry_date: date,
    max_rows: int = 2000,
) -> list[OptionQuote]:
    rows = (
        db.execute(
            select(OptionQuote)
            .where(
                symbol_value_filter(OptionQuote.underlying_symbol, symbol),
                OptionQuote.expiry_date == expiry_date,
            )
            .order_by(OptionQuote.ts.desc())
            .limit(max_rows)
        )
        .scalars()
        .all()
    )
    by_contract: dict[tuple[float, str], OptionQuote] = {}
    for row in rows:
        key = (float(row.strike), str(row.option_type).upper())
        selected = by_contract.get(key)
        selected_priority = (
            int(_is_real_upstox_option_source(selected.source)),
            int(_is_positive_option_ltp(selected.ltp)),
        ) if selected is not None else (-1, -1)
        row_priority = (
            int(_is_real_upstox_option_source(row.source)),
            int(_is_positive_option_ltp(row.ltp)),
        )
        if selected is not None and selected_priority >= row_priority:
            continue
        by_contract[key] = row
    return sorted(by_contract.values(), key=lambda item: (float(item.strike), str(item.option_type)))


def _is_real_upstox_option_source(source: Any) -> bool:
    return str(source or "").strip().lower().startswith("upstox_option_chain")


def _is_positive_option_ltp(value: Any) -> bool:
    ltp = _to_float(value)
    return ltp is not None and ltp > 0.0


def _option_quote_book_is_crossed(bid: Any, ask: Any) -> bool:
    bid_value = _to_float(bid)
    ask_value = _to_float(ask)
    return bid_value is not None and ask_value is not None and ask_value < bid_value


def _option_quote_age_seconds(quote_ts: datetime | None, *, now: datetime | None = None) -> float | None:
    current = _ensure_ist(now) or datetime.now(IST_ZONE)
    latest = _ensure_ist(quote_ts)
    if latest is None:
        return None
    return round(max(0.0, (current - latest).total_seconds()), 3)


def resolve_underlying_key(db: Session, symbol: str, settings: Settings | None = None) -> str | None:
    settings = settings or get_settings()
    aliases = {alias.upper().replace(" ", "") for alias in symbol_aliases(symbol)}
    for instrument_key in settings.instrument_keys:
        display = instrument_key.split("|", 1)[1] if "|" in instrument_key else instrument_key
        if display.upper().replace(" ", "") in aliases:
            return instrument_key
    key = db.scalar(
        select(OptionQuote.underlying_key)
        .where(symbol_value_filter(OptionQuote.underlying_symbol, symbol))
        .order_by(OptionQuote.ts.desc())
        .limit(1)
    )
    return str(key) if key else None


def _resolve_expiry(
    *,
    symbol: str,
    underlying_key: str | None,
    settings: Settings,
) -> tuple[date, list[date]]:
    expiries: list[date] = []
    if underlying_key and settings.has_market_data_access:
        try:
            expiries = UpstoxOptionChainCollector().list_expiries(underlying_key, max_items=8)
        except Exception:
            expiries = []
    if not expiries:
        expiries = next_weekly_expiries(symbol=symbol, count=6)
    today = datetime.now(IST_ZONE).date()
    valid = [exp for exp in expiries if (exp - today).days >= 1]
    expiries = valid or expiries
    return expiries[0], expiries


def _latest_option_quote_ts(db: Session, symbol: str, expiry_date: date) -> datetime | None:
    return db.scalar(
        select(func.max(OptionQuote.ts)).where(
            symbol_value_filter(OptionQuote.underlying_symbol, symbol),
            OptionQuote.expiry_date == expiry_date,
        )
    )


def _latest_real_option_quote_ts(db: Session, symbol: str, expiry_date: date) -> datetime | None:
    return db.scalar(
        select(func.max(OptionQuote.ts)).where(
            symbol_value_filter(OptionQuote.underlying_symbol, symbol),
            OptionQuote.expiry_date == expiry_date,
            func.lower(OptionQuote.source).like("upstox_option_chain%"),
            OptionQuote.ltp > 0,
        )
    )


def _latest_upstox_option_chain_success_ts(
    db: Session,
    symbol: str,
    expiry_date: date,
) -> datetime | None:
    freshness = db.scalar(
        select(DataFreshness)
        .where(DataFreshness.source_name == f"upstox_option_chain:{symbol}")
        .limit(1)
    )
    if freshness is None:
        return None
    details = freshness.details or {}
    recorded_expiry = str(details.get("expiry_date") or "").strip()
    if recorded_expiry and recorded_expiry != expiry_date.isoformat():
        return None
    return freshness.last_success_at


def _latest_upstox_option_chain_snapshot_ts(
    db: Session,
    symbol: str,
    expiry_date: date,
) -> datetime | None:
    freshness = db.scalar(
        select(DataFreshness)
        .where(DataFreshness.source_name == f"upstox_option_chain:{symbol}")
        .limit(1)
    )
    if freshness is None:
        return None
    details = freshness.details or {}
    recorded_expiry = str(details.get("expiry_date") or "").strip()
    if recorded_expiry and recorded_expiry != expiry_date.isoformat():
        return None
    return _parse_iso_datetime(details.get("snapshot_ts"))


def _effective_real_quote_ts(
    quote_ts: datetime | None,
    chain_success_ts: datetime | None,
    chain_snapshot_ts: datetime | None = None,
) -> datetime | None:
    quote = _ensure_ist(quote_ts)
    success = _ensure_ist(chain_success_ts)
    snapshot = _ensure_ist(chain_snapshot_ts)
    if quote is None:
        return None
    if success is None or snapshot is None:
        return quote
    if quote == snapshot:
        return max(quote, success)
    return quote


def _option_quote_is_fresh(
    quote_ts: datetime | None,
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> bool:
    settings = settings or get_settings()
    if quote_ts is None:
        return False
    current = _ensure_ist(now) or datetime.now(IST_ZONE)
    latest = _ensure_ist(quote_ts)
    if latest is None:
        return False
    max_age_seconds = max(2, int(getattr(settings, "option_chain_refresh_seconds", 4)) * 3)
    age_seconds = (current - latest).total_seconds()
    return -2.0 <= age_seconds <= max_age_seconds


def resolve_live_option_quote(
    db: Session,
    *,
    symbol: str,
    expiry_date: date,
    strike: float,
    option_type: str,
    instrument_key: str | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    settings = settings or get_settings()
    current = _ensure_ist(now) or datetime.now(IST_ZONE)
    normalized_type = str(option_type).upper()
    chain_success_ts = _latest_upstox_option_chain_success_ts(db, symbol, expiry_date)
    chain_snapshot_ts = _latest_upstox_option_chain_snapshot_ts(db, symbol, expiry_date)

    def _load_row() -> OptionQuote | None:
        filters = [
            symbol_value_filter(OptionQuote.underlying_symbol, symbol),
            OptionQuote.expiry_date == expiry_date,
            OptionQuote.strike == float(strike),
            OptionQuote.option_type == normalized_type,
            func.lower(OptionQuote.source).like("upstox_option_chain%"),
            OptionQuote.ltp > 0,
        ]
        if instrument_key:
            filters.insert(0, OptionQuote.instrument_key == str(instrument_key))
        return db.scalar(
            select(OptionQuote)
            .where(*filters)
            .order_by(OptionQuote.ts.desc())
            .limit(1)
        )

    row = _load_row()
    effective_ts = _effective_real_quote_ts(
        row.ts if row is not None else None,
        chain_success_ts,
        chain_snapshot_ts,
    )
    if (
        row is not None
        and _is_positive_option_ltp(row.ltp)
        and not _option_quote_book_is_crossed(row.bid, row.ask)
        and _option_quote_is_fresh(effective_ts, settings=settings, now=current)
    ):
        return {
            "instrument_key": str(row.instrument_key),
            "ltp": float(row.ltp),
            "bid": float(row.bid) if row.bid is not None else None,
            "ask": float(row.ask) if row.ask is not None else None,
            "source": str(row.source or "db"),
            "ts": effective_ts,
            "age_seconds": _option_quote_age_seconds(effective_ts, now=current),
            "stale": False,
        }

    underlying_key = resolve_underlying_key(db, symbol, settings=settings)
    _maybe_refresh_option_chain(
        db,
        symbol=symbol,
        underlying_key=underlying_key,
        expiry_date=expiry_date,
        settings=settings,
    )
    row = _load_row()
    chain_success_ts = _latest_upstox_option_chain_success_ts(db, symbol, expiry_date)
    chain_snapshot_ts = _latest_upstox_option_chain_snapshot_ts(db, symbol, expiry_date)
    effective_ts = _effective_real_quote_ts(
        row.ts if row is not None else None,
        chain_success_ts,
        chain_snapshot_ts,
    )
    if (
        row is not None
        and _is_positive_option_ltp(row.ltp)
        and not _option_quote_book_is_crossed(row.bid, row.ask)
        and _option_quote_is_fresh(effective_ts, settings=settings, now=current)
    ):
        return {
            "instrument_key": str(row.instrument_key),
            "ltp": float(row.ltp),
            "bid": float(row.bid) if row.bid is not None else None,
            "ask": float(row.ask) if row.ask is not None else None,
            "source": str(row.source or "db"),
            "ts": effective_ts,
            "age_seconds": _option_quote_age_seconds(effective_ts, now=current),
            "stale": False,
        }
    return None


def compute_paper_portfolio_metrics(
    db: Session,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    starting_balance = float(get_paper_starting_balance(db, settings=settings))
    reset_at = get_paper_reset_at(db)
    open_positions = (
        db.execute(select(ExecutionPosition).where(ExecutionPosition.status == "OPEN"))
        .scalars()
        .all()
    )
    closed_positions = (
        db.execute(select(ExecutionPosition).where(ExecutionPosition.status == "CLOSED"))
        .scalars()
        .all()
    )
    def _is_paper(row: ExecutionPosition) -> bool:
        metadata = row.metadata_json or {}
        return str(metadata.get("execution_mode") or "paper").lower() == "paper"
    def _in_window(row: ExecutionPosition) -> bool:
        opened = _ensure_ist(row.opened_at)
        return reset_at is None or (opened is not None and opened >= _ensure_ist(reset_at))
    open_positions = [row for row in open_positions if _is_paper(row) and _in_window(row)]
    closed_positions = [row for row in closed_positions if _is_paper(row) and _in_window(row)]
    invested_amount = round(
        sum(float(row.entry_premium or row.entry_price or 0.0) * int(row.quantity or 0) for row in open_positions),
        2,
    )
    realized_pnl = round(
        sum(float(row.realized_pnl or row.pnl_value or 0.0) for row in closed_positions),
        2,
    )
    unrealized_pnl = round(
        sum(float(row.unrealized_pnl or 0.0) for row in open_positions),
        2,
    )
    unpriced_positions = [
        row
        for row in open_positions
        if str((row.metadata_json or {}).get("latest_quote_status") or "").lower() == "unavailable"
    ]
    unpriced_unrealized_pnl = round(
        sum(float(row.unrealized_pnl or 0.0) for row in unpriced_positions),
        2,
    )
    available_balance = round(starting_balance + realized_pnl - invested_amount, 2)
    equity = round(available_balance + invested_amount + unrealized_pnl, 2)
    return {
        "starting_balance": starting_balance,
        "available_balance": available_balance,
        "invested_amount": invested_amount,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "unpriced_positions_count": len(unpriced_positions),
        "unpriced_unrealized_pnl": unpriced_unrealized_pnl,
        "priced_unrealized_pnl": round(unrealized_pnl - unpriced_unrealized_pnl, 2),
        "total_pnl": round(realized_pnl + unrealized_pnl, 2),
        "equity": equity,
        "reset_at": reset_at.isoformat() if reset_at is not None else "",
    }


def _position_execution_mode(row: ExecutionPosition) -> str:
    metadata = row.metadata_json or {}
    return str(metadata.get("execution_mode") or "paper").lower()


def _position_matches_mode(row: ExecutionPosition, mode: str) -> bool:
    return _position_execution_mode(row) == str(mode or "paper").lower()


def _order_matches_mode(row: ExecutionOrder, mode: str) -> bool:
    broker_name = str(row.broker_name or "paper").lower()
    normalized_mode = str(mode or "paper").lower()
    if normalized_mode == "live":
        return broker_name != "paper"
    return broker_name == "paper"


def _compute_iv_rank(
    db: Session,
    symbol: str,
    current_atm_iv: float | None,
) -> float | None:
    """IV rank = (current IV − 90d low) / (90d high − 90d low).

    Returns None when there is insufficient historical data.
    Returns a value in [0, 1]: 0 = historically cheap, 1 = historically expensive.
    """
    if not current_atm_iv or current_atm_iv <= 0:
        return None
    cutoff = datetime.now(IST_ZONE) - timedelta(days=90)
    stats = db.execute(
        select(func.min(OptionQuote.iv), func.max(OptionQuote.iv)).where(
            symbol_value_filter(OptionQuote.underlying_symbol, symbol),
            OptionQuote.iv.isnot(None),
            OptionQuote.ts >= cutoff,
        )
    ).one_or_none()
    if stats is None or stats[0] is None or stats[1] is None:
        return None
    iv_low, iv_high = float(stats[0]), float(stats[1])
    if iv_high <= iv_low + 1e-9:
        return None
    rank = (current_atm_iv - iv_low) / (iv_high - iv_low)
    return round(max(0.0, min(1.0, rank)), 3)


def _maybe_refresh_option_chain(
    db: Session,
    *,
    symbol: str,
    underlying_key: str | None,
    expiry_date: date,
    settings: Settings,
) -> None:
    if underlying_key is None or not settings.has_market_data_access:
        return
    latest_ts = _latest_real_option_quote_ts(db, symbol, expiry_date)
    now = datetime.now(IST_ZONE)
    stale = latest_ts is None or (now - (_ensure_ist(latest_ts) or now)).total_seconds() > max(
        2,
        int(settings.option_chain_refresh_seconds),
    )
    if not stale:
        return
    try:
        UpstoxOptionChainCollector().sync_option_chain(
            db,
            underlying_key=underlying_key,
            underlying_symbol=symbol,
            expiry_date=expiry_date,
        )
        db.commit()
    except Exception:
        db.rollback()


def _option_candidate_diagnostics(
    rows: list[OptionQuote],
    *,
    requested_atm: float,
    strike_step: int,
    option_type: str,
    settings: Settings,
    now: datetime,
    chain_success_ts: datetime | None = None,
    chain_snapshot_ts: datetime | None = None,
) -> list[dict[str, Any]]:
    normalized_type = str(option_type).upper()
    expected_strikes = [
        float(requested_atm),
        float(requested_atm - strike_step),
        float(requested_atm + strike_step),
        float(requested_atm - (2 * strike_step)),
        float(requested_atm + (2 * strike_step)),
    ]
    rows_by_strike = {
        float(row.strike): row
        for row in rows
        if str(row.option_type).upper() == normalized_type
    }
    premium_min = float(settings.execution_premium_min)
    premium_max = float(settings.execution_premium_max)
    min_volume = max(1.0, float(getattr(settings, "option_min_volume", 500.0)))
    min_oi = max(1.0, float(getattr(settings, "option_min_oi", 1000.0)))
    max_spread_pct = max(0.0001, float(getattr(settings, "option_max_spread_pct", 0.08)))
    diagnostics: list[dict[str, Any]] = []

    for strike in expected_strikes:
        row = rows_by_strike.get(strike)
        distance_steps = int(round(abs(strike - requested_atm) / max(1, strike_step)))
        if row is None:
            diagnostics.append(
                {
                    "strike": strike,
                    "option_type": normalized_type,
                    "distance_steps": distance_steps,
                    "status": "rejected",
                    "source": None,
                    "quote_ts": None,
                    "quote_age_seconds": None,
                    "ltp": None,
                    "bid": None,
                    "ask": None,
                    "spread_pct": None,
                    "volume": None,
                    "oi": None,
                    "ranking_score": None,
                    "rejections": ["No quote is available for this strike."],
                }
            )
            continue

        quote_ts = _effective_real_quote_ts(row.ts, chain_success_ts, chain_snapshot_ts)
        age_seconds = _option_quote_age_seconds(quote_ts, now=now)
        quote = {
            "instrument_key": str(row.instrument_key or ""),
            "ltp": _to_float(row.ltp),
            "bid": _to_float(row.bid),
            "ask": _to_float(row.ask),
            "volume": _to_float(row.volume),
            "oi": _to_float(row.oi),
            "source": str(row.source or ""),
        }
        entry_price = _to_float(row.ltp)
        bid = _to_float(row.bid)
        ask = _to_float(row.ask)
        volume = _to_float(row.volume) or 0.0
        oi = _to_float(row.oi) or 0.0
        spread_pct = (
            (float(ask) - float(bid)) / max(float(entry_price or 0.0), 1.0)
            if bid is not None and ask is not None and bid > 0 and ask >= bid
            else None
        )
        rejections: list[str] = []
        if not _is_real_upstox_option_source(row.source):
            rejections.append(f"Quote source {row.source or 'unknown'} is not a real Upstox option-chain quote.")
        if not _option_quote_is_fresh(quote_ts, settings=settings, now=now):
            max_age = max(2, int(getattr(settings, "option_chain_refresh_seconds", 4)) * 3)
            age_label = "unknown" if age_seconds is None else f"{age_seconds:.1f}s"
            rejections.append(f"Quote is stale (age {age_label}, maximum {max_age}s).")
        if entry_price is None or not (premium_min <= entry_price <= premium_max):
            price_label = "missing" if entry_price is None else f"{entry_price:.2f}"
            rejections.append(
                f"Premium {price_label} is outside configured range {premium_min:.2f}-{premium_max:.2f}."
            )
        rejections.extend(_option_liquidity_failures(quote, settings))

        ranking_score = None
        if not rejections and spread_pct is not None:
            distance_component = distance_steps * 0.35
            spread_component = spread_pct / max_spread_pct
            liquidity_component = (
                min(volume / min_volume, 5.0) * 0.15
                + min(oi / min_oi, 5.0) * 0.15
            )
            ranking_score = round(distance_component + spread_component - liquidity_component, 6)

        diagnostics.append(
            {
                "strike": strike,
                "option_type": normalized_type,
                "distance_steps": distance_steps,
                "status": "eligible" if not rejections else "rejected",
                "instrument_key": str(row.instrument_key or ""),
                "source": str(row.source or ""),
                "quote_ts": quote_ts.isoformat() if quote_ts is not None else None,
                "quote_age_seconds": age_seconds,
                "ltp": entry_price,
                "bid": bid,
                "ask": ask,
                "spread_pct": round(spread_pct, 6) if spread_pct is not None else None,
                "volume": volume,
                "oi": oi,
                "ranking_score": ranking_score,
                "rejections": rejections,
            }
        )
    return diagnostics


def build_option_selection(
    db: Session,
    *,
    context: MarketContext,
    signal: TechnicalSignal,
    settings: Settings | None = None,
) -> OptionSelection:
    settings = settings or get_settings()
    underlying_key = resolve_underlying_key(db, context.symbol, settings=settings)
    expiry_date, available_expiries = _resolve_expiry(
        symbol=context.symbol,
        underlying_key=underlying_key,
        settings=settings,
    )
    strike_step = strike_step_for_symbol(context.symbol)
    if signal.action not in {"BUY", "SELL"}:
        return OptionSelection(
            expiry_date=expiry_date,
            strike_step=strike_step,
            chain_source="standby",
            chain_generated_at=None,
            available_expiries=available_expiries,
            chain_rows=[],
            signal={
                "action": "HOLD",
                "option_type": None,
                "strike": None,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "confidence": signal.confidence,
                "reasons": ["No option contract is loaded until the signal qualifies for a trade."],
            },
        )
    _maybe_refresh_option_chain(
        db,
        symbol=context.symbol,
        underlying_key=underlying_key,
        expiry_date=expiry_date,
        settings=settings,
    )
    current = datetime.now(IST_ZONE)
    quote_rows = _load_latest_option_quote_rows(db, symbol=context.symbol, expiry_date=expiry_date)
    quotes = _load_option_quotes(db, symbol=context.symbol, expiry_date=expiry_date)
    real_quote_rows = [row for row in quote_rows if _is_real_upstox_option_source(row.source)]
    chain_success_ts = _latest_upstox_option_chain_success_ts(db, context.symbol, expiry_date)
    chain_snapshot_ts = _latest_upstox_option_chain_snapshot_ts(db, context.symbol, expiry_date)
    real_chain_generated_at = _effective_real_quote_ts(
        max(
            (_ensure_ist(row.ts) for row in real_quote_rows if _ensure_ist(row.ts) is not None),
            default=None,
        ),
        chain_success_ts,
        chain_snapshot_ts,
    )
    chain_generated_at = _latest_option_quote_ts(db, context.symbol, expiry_date)
    chain_source = next((str(item.source or "db") for item in quotes if item.source), "unavailable")
    if not quotes:
        display_quotes = synthetic_option_chain(
            symbol=context.symbol,
            underlying_price=context.latest_price,
            expiry_date=expiry_date,
            strike_step=strike_step,
        )
        quotes = display_quotes
        chain_source = "synthetic_display_only"
        chain_generated_at = datetime.now(IST_ZONE)
    elif not real_quote_rows:
        chain_source = "synthetic_display_only"
    chain_rows = build_chain_rows(quotes)

    signal_payload: dict[str, Any] = {
        "action": "HOLD",
        "option_type": None,
        "strike": None,
        "entry_price": None,
        "stop_loss": None,
        "take_profit": None,
        "confidence": signal.confidence,
        "reasons": ["Waiting for a qualified directional signal."],
    }

    dte = max(1, (expiry_date - datetime.now(IST_ZONE).date()).days)
    option_type = "CE" if signal.action == "BUY" else "PE"
    requested_atm = nearest_strike(context.latest_price, strike_step)
    candidate_diagnostics = _option_candidate_diagnostics(
        quote_rows,
        requested_atm=requested_atm,
        strike_step=strike_step,
        option_type=option_type,
        settings=settings,
        now=current,
        chain_success_ts=chain_success_ts,
        chain_snapshot_ts=chain_snapshot_ts,
    )
    real_chain_fresh = _option_quote_is_fresh(
        real_chain_generated_at,
        settings=settings,
        now=current,
    )
    eligible_candidates = [
        item for item in candidate_diagnostics
        if item["status"] == "eligible" and item["ranking_score"] is not None
    ]
    selected = min(
        eligible_candidates,
        key=lambda item: (
            float(item["ranking_score"]),
            int(item["distance_steps"]),
            float(item["spread_pct"]),
            -float(item["oi"]),
            -float(item["volume"]),
        ),
        default=None,
    )
    if not real_chain_fresh or selected is None:
        reasons: list[str] = []
        if not real_quote_rows:
            reasons.append("No real Upstox option chain is available; synthetic quotes are display-only.")
        elif not real_chain_fresh:
            chain_age = _option_quote_age_seconds(real_chain_generated_at, now=current)
            age_label = "unknown" if chain_age is None else f"{chain_age:.1f}s"
            reasons.append(f"Real Upstox option chain is stale (latest age {age_label}).")
        rejected = [item for item in candidate_diagnostics if item["rejections"]]
        for item in rejected:
            reasons.append(
                f"{item['strike']:.0f} {option_type}: {item['rejections'][0]}"
            )
        signal_payload = {
            "action": "HOLD",
            "option_type": None,
            "strike": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "confidence": signal.confidence,
            "quote_status": "unavailable",
            "quote_source": None,
            "quote_age_seconds": None,
            "requested_atm": float(requested_atm),
            "candidate_diagnostics": candidate_diagnostics,
            "reasons": (reasons or ["No liquid fresh real option quote passed the execution filter."])[:6],
        }
    else:
        selected["status"] = "selected"
        selected_strike = float(selected["strike"])
        entry_price = float(selected["ltp"])
        quote_source = str(selected["source"])
        risk_plan = _option_risk_plan(entry_price, signal, settings)
        iv_rank = _compute_iv_rank(db, context.symbol, _strike_get_atm_iv(chain_rows, requested_atm))
        signal_payload = {
            "action": "BUY",
            "option_type": option_type,
            "strike": selected_strike,
            "entry_price": entry_price,
            "stop_loss": float(risk_plan["stop_loss"]),
            "take_profit": float(risk_plan["take_profit"]),
            "confidence": signal.confidence,
            "instrument_key": selected["instrument_key"],
            "quote_status": "available",
            "quote_source": quote_source,
            "quote_ts": selected["quote_ts"],
            "quote_age_seconds": selected["quote_age_seconds"],
            "requested_atm": float(requested_atm),
            "candidate_diagnostics": candidate_diagnostics,
            "iv_rank": iv_rank,
            "days_to_expiry": dte,
            "trail_trigger_price": float(risk_plan["trail_trigger_price"]),
            "trailing_stop_loss": float(risk_plan["trailing_stop_loss"]),
            "trail_step_pct": float(OPTION_TRAIL_STOP_PCT),
            "stop_loss_pct": float(risk_plan["stop_pct"]),
            "target_pct": float(risk_plan["target_pct"]),
            "liquidity": {
                "volume": selected["volume"],
                "oi": selected["oi"],
                "bid": selected["bid"],
                "ask": selected["ask"],
                "spread_pct": selected["spread_pct"],
                "max_spread_pct": float(getattr(settings, "option_max_spread_pct", 0.08)),
            },
            "reasons": [
                f"{signal.action} signal mapped to {option_type}.",
                f"Evaluated ATM +/- 2 strikes around {requested_atm:.0f}; selected {selected_strike:.0f}.",
                f"Selected by distance, spread, OI and volume score {selected['ranking_score']:.3f}.",
                f"Fresh real quote source: {quote_source}, age {float(selected['quote_age_seconds'] or 0.0):.1f}s.",
                f"ATR-aware option risk: SL {risk_plan['stop_pct']:.0%}, target {risk_plan['target_pct']:.0%}.",
            ],
        }

    return OptionSelection(
        expiry_date=expiry_date,
        strike_step=strike_step,
        chain_source=chain_source,
        chain_generated_at=_ensure_ist(chain_generated_at),
        available_expiries=available_expiries,
        chain_rows=chain_rows,
        signal=signal_payload,
    )


def _option_liquidity_failures(quote: dict[str, Any], settings: Settings) -> list[str]:
    failures: list[str] = []
    entry_price = _to_float(quote.get("ltp")) or 0.0
    volume = _to_float(quote.get("volume")) or 0.0
    oi = _to_float(quote.get("oi")) or 0.0
    bid = _to_float(quote.get("bid"))
    ask = _to_float(quote.get("ask"))
    min_volume = float(getattr(settings, "option_min_volume", 500.0))
    min_oi = float(getattr(settings, "option_min_oi", 1000.0))
    max_spread_pct = float(getattr(settings, "option_max_spread_pct", 0.08))
    if not quote.get("instrument_key"):
        failures.append("Selected option has no broker instrument key.")
    if entry_price <= 0:
        failures.append("Option LTP must be positive.")
    if volume < min_volume:
        failures.append(f"Option volume {volume:.0f} is below minimum {min_volume:.0f}.")
    if oi < min_oi:
        failures.append(f"Option OI {oi:.0f} is below minimum {min_oi:.0f}.")
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        failures.append("Option bid/ask is missing.")
    elif ask < bid:
        failures.append("Option order book is crossed because ask is below bid.")
    else:
        spread_pct = (ask - bid) / max(entry_price, 1.0)
        if spread_pct > max_spread_pct:
            failures.append(f"Option spread {spread_pct:.1%} exceeds maximum {max_spread_pct:.1%}.")
    return failures


def _option_risk_plan(entry_price: float, signal: TechnicalSignal, settings: Settings) -> dict[str, float]:
    if not bool(getattr(settings, "enhanced_risk_enabled", True)):
        stop_pct = float(OPTION_STOP_LOSS_PCT)
        target_pct = float(OPTION_TARGET_PCT)
    else:
        atr_points = float(signal.details.get("atr_14") or 0.0)
        atr_risk_points = atr_points * float(getattr(settings, "atr_sl_multiplier", 1.8))
        stop_pct = _clip(atr_risk_points / 100.0, 0.18, 0.35)
        target_pct = _clip(stop_pct * float(getattr(settings, "target_rr_ratio", 2.2)), 0.30, 0.80)
    return {
        "stop_pct": float(stop_pct),
        "target_pct": float(target_pct),
        "stop_loss": round(entry_price * (1.0 - stop_pct), 2),
        "take_profit": round(entry_price * (1.0 + target_pct), 2),
        "trail_trigger_price": round(entry_price * (1.0 + max(0.30, target_pct * 0.6)), 2),
        "trailing_stop_loss": round(entry_price * (1.0 - min(0.20, stop_pct * 0.65)), 2),
    }


def latest_option_premium(
    db: Session,
    *,
    symbol: str,
    expiry_date: date,
    strike: float,
    option_type: str,
    instrument_key: str | None = None,
    settings: Settings | None = None,
) -> float | None:
    quote = resolve_live_option_quote(
        db,
        symbol=symbol,
        expiry_date=expiry_date,
        strike=float(strike),
        option_type=option_type,
        instrument_key=instrument_key,
        settings=settings,
    )
    if quote is None:
        return None
    return float(quote["ltp"])


def _mark_position_to_market(
    row: ExecutionPosition,
    *,
    premium: float,
    quote_source: str,
    quote_ts: datetime | None,
    quote_age_seconds: float | None,
) -> None:
    if premium <= 0:
        raise ValueError("Option mark premium must be positive.")
    row.current_price = float(premium)
    row.current_premium = float(premium)
    row.peak_premium = max(
        float(row.peak_premium or row.entry_premium or row.entry_price or 0.0),
        float(premium),
    )
    row.unrealized_pnl = round(
        (float(premium) - float(row.entry_premium or row.entry_price or 0.0)) * int(row.quantity or 0),
        2,
    )
    row.pnl_value = float(row.unrealized_pnl)
    row.pnl_points = round(float(premium) - float(row.entry_premium or row.entry_price or 0.0), 2)
    metadata = dict(row.metadata_json or {})
    metadata["latest_quote_status"] = "available"
    metadata["latest_quote_source"] = quote_source
    metadata["latest_quote_ts"] = quote_ts.isoformat() if quote_ts is not None else None
    metadata["latest_quote_age_seconds"] = quote_age_seconds
    metadata.pop("latest_quote_unavailable_reason", None)
    row.metadata_json = metadata


def refresh_open_positions_snapshot(
    db: Session,
    *,
    settings: Settings | None = None,
) -> list[ExecutionPosition]:
    settings = settings or get_settings()
    runtime_mode = get_runtime_trading_mode(db, settings=settings)
    rows = (
        db.execute(
            select(ExecutionPosition)
            .where(
                ExecutionPosition.status.in_(
                    ["OPEN", "ENTRY_PENDING", "EXIT_SUBMITTING", "EXIT_PENDING"]
                )
            )
            .order_by(ExecutionPosition.opened_at.desc())
        )
        .scalars()
        .all()
    )
    rows = [row for row in rows if _position_matches_mode(row, runtime_mode)]
    for row in rows:
        if str(row.status).upper() != "OPEN":
            continue
        metadata = row.metadata_json or {}
        quote = resolve_live_option_quote(
            db,
            symbol=str(row.symbol),
            expiry_date=row.expiry_date,
            strike=float(row.strike),
            option_type=str(row.option_type),
            instrument_key=str(metadata.get("instrument_key") or "") or None,
            settings=settings,
        )
        if quote is None:
            unavailable_metadata = dict(metadata)
            unavailable_metadata["latest_quote_status"] = "unavailable"
            unavailable_metadata["latest_quote_source"] = None
            unavailable_metadata["latest_quote_ts"] = None
            unavailable_metadata["latest_quote_age_seconds"] = None
            unavailable_metadata["latest_quote_checked_at"] = datetime.now(IST_ZONE).isoformat()
            unavailable_metadata["latest_quote_unavailable_reason"] = (
                "No fresh real Upstox option quote is available; position P&L was not changed."
            )
            row.metadata_json = unavailable_metadata
            continue
        _mark_position_to_market(
            row,
            premium=float(quote["ltp"]),
            quote_source=str(quote.get("source") or "unknown"),
            quote_ts=_ensure_ist(quote.get("ts")),
            quote_age_seconds=_to_float(quote.get("age_seconds")),
        )
    if rows:
        db.flush()
    return rows


def _serialize_position(row: ExecutionPosition) -> dict[str, Any]:
    metadata = row.metadata_json or {}
    return {
        "position_id": row.id,
        "symbol": row.symbol,
        "strategy_name": row.strategy_name,
        "strike": row.strike,
        "option_type": row.option_type,
        "expiry": row.expiry_date.isoformat(),
        "quantity": row.quantity,
        "entry_premium": row.entry_premium or row.entry_price,
        "current_premium": row.current_premium or row.current_price,
        "exit_premium": row.exit_premium,
        "unrealized_pnl": row.unrealized_pnl,
        "realized_pnl": row.realized_pnl,
        "entry_time": _ensure_ist(row.opened_at).isoformat() if row.opened_at else None,
        "exit_time": _ensure_ist(row.closed_at).isoformat() if row.closed_at else None,
        "current_sl": row.current_sl or row.stop_loss,
        "initial_sl": row.initial_sl or row.stop_loss,
        "target_premium": row.target_premium or row.take_profit,
        "status": row.status,
        "exit_reason": row.exit_reason,
        "instrument_key": metadata.get("instrument_key"),
        "latest_quote_source": metadata.get("latest_quote_source"),
        "latest_quote_ts": metadata.get("latest_quote_ts"),
        "latest_quote_age_seconds": metadata.get("latest_quote_age_seconds"),
        "latest_quote_status": metadata.get("latest_quote_status"),
        "latest_quote_unavailable_reason": metadata.get("latest_quote_unavailable_reason"),
        "premium_history": metadata.get("premium_history") or [],
    }


def _serialize_order(row: ExecutionOrder) -> dict[str, Any]:
    return {
        "id": row.id,
        "symbol": row.symbol,
        "strike_price": row.strike_price,
        "option_type": row.option_type,
        "expiry_date": row.expiry_date.isoformat() if row.expiry_date else None,
        "order_kind": row.order_kind,
        "side": row.side,
        "quantity": row.quantity,
        "price": row.price,
        "trigger_price": row.trigger_price,
        "status": row.status,
        "realized_pnl": row.realized_pnl,
        "unrealized_pnl": row.unrealized_pnl,
        "created_at": _ensure_ist(row.created_at).isoformat() if row.created_at else None,
        "exit_reason": row.exit_reason,
        "consensus_reason": row.consensus_reason,
    }


def _serialize_signal_log(row: SignalLog) -> dict[str, Any]:
    details = dict(row.details or {})
    option_selection = (
        dict(details.get("option_selection") or {})
        if isinstance(details.get("option_selection"), dict)
        else {}
    )
    return {
        "id": row.id,
        "timestamp": _ensure_ist(row.timestamp).isoformat() if row.timestamp else None,
        "symbol": row.symbol,
        "interval": row.interval,
        "consensus": row.consensus,
        "combined_score": row.combined_score,
        "pine_signal": row.pine_signal,
        "trade_placed": bool(row.trade_placed),
        "skip_reason": row.skip_reason,
        "details": details,
        "option_selection": option_selection,
        "quote_status": option_selection.get("quote_status"),
        "quote_source": option_selection.get("quote_source"),
        "quote_ts": option_selection.get("quote_ts"),
        "quote_age_seconds": option_selection.get("quote_age_seconds"),
        "requested_atm": option_selection.get("requested_atm"),
        "candidate_diagnostics": option_selection.get("candidate_diagnostics") or [],
        "selection_reasons": option_selection.get("reasons") or [],
        "fresh_graph_marker": bool(details.get("fresh_graph_marker")),
        "pine_marker_time": details.get("pine_marker_time"),
        "pine_marker_text": details.get("pine_marker_text"),
    }


def _freshness_payload(
    db: Session,
    *,
    instrument_key: str,
    symbol: str,
    latest_candle_ts: datetime | None,
) -> dict[str, Any]:
    now = datetime.now(IST_ZONE)
    session_start, session_end = market_session_bounds(now.date())
    latest_session_date = latest_candle_ts.date() if latest_candle_ts is not None else None
    if not is_trading_day(now.date()) or now < session_start:
        expected_session_date = previous_trading_day(now.date())
    else:
        expected_session_date = now.date()
    age_seconds = None
    if latest_candle_ts is not None:
        age_seconds = max(0.0, (now - latest_candle_ts).total_seconds())
    is_live = bool(
        latest_session_date == now.date()
        and session_start <= now <= session_end
        and age_seconds is not None
        and age_seconds <= 90.0
    )
    market_status = "live" if is_live else ("complete_previous_session" if latest_session_date == expected_session_date else "stale")
    return {
        "symbol": symbol,
        "instrument_key": instrument_key,
        "latest_candle_ts": latest_candle_ts.isoformat() if latest_candle_ts is not None else None,
        "latest_candle_age_seconds": round(age_seconds, 1) if age_seconds is not None else None,
        "latest_session_date": latest_session_date.isoformat() if latest_session_date is not None else None,
        "expected_session_date": expected_session_date.isoformat(),
        "market_status": market_status,
        "is_live": is_live,
    }


def _stream_diagnostics_payload(
    db: Session,
    *,
    instrument_key: str,
    latest_candle_ts: datetime | None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    cfg = settings or get_settings()
    now = datetime.now(IST_ZONE)
    row = db.scalar(select(DataFreshness).where(DataFreshness.source_name == "upstox_market_stream"))
    details = row.details if row is not None and isinstance(row.details, dict) else {}
    latest_candle = _ensure_ist(latest_candle_ts)
    last_success_at = _ensure_ist(row.last_success_at) if row is not None else None
    latest_exchange_ts = _parse_iso_datetime(details.get("latest_exchange_ts"))
    message_received_at = _parse_iso_datetime(details.get("message_received_at"))
    write_completed_at = _parse_iso_datetime(details.get("write_completed_at"))
    runtime = get_market_stream_runtime_status(cfg)
    latest_event_ts = latest_exchange_ts or latest_candle

    return {
        "provider": "upstox_market_stream",
        "instrument_key": instrument_key,
        "status": row.status if row is not None else "missing",
        "stream_last_success_at": last_success_at.isoformat() if last_success_at is not None else None,
        "stream_last_success_age_seconds": (
            round(max(0.0, (now - last_success_at).total_seconds()), 1)
            if last_success_at is not None
            else None
        ),
        "latest_exchange_ts": latest_exchange_ts.isoformat() if latest_exchange_ts is not None else None,
        "latest_candle_ts": latest_candle.isoformat() if latest_candle is not None else None,
        "message_received_at": message_received_at.isoformat() if message_received_at is not None else None,
        "write_completed_at": write_completed_at.isoformat() if write_completed_at is not None else None,
        "exchange_timestamp_precision": details.get("exchange_timestamp_precision"),
        "estimated_exchange_to_receive_latency_ns": _to_int(
            details.get("estimated_exchange_to_receive_latency_ns")
        ),
        "estimated_receive_to_persist_latency_ns": _to_int(
            details.get("estimated_receive_to_persist_latency_ns")
        ),
        "estimated_exchange_to_persist_latency_ns": _to_int(
            details.get("estimated_exchange_to_persist_latency_ns")
        ),
        "estimated_exchange_to_now_latency_ns": _ns_between(now, latest_event_ts),
        "estimated_persist_to_now_latency_ns": _ns_between(now, write_completed_at),
        "candles_flushed": _to_int(details.get("candles_flushed")),
        "order_books_flushed": _to_int(details.get("order_books_flushed")),
        "source": details.get("source"),
        "runtime": runtime,
    }


def _stats_payload(db: Session, *, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    today = datetime.now(IST_ZONE).date()
    runtime_mode = get_runtime_trading_mode(db, settings=settings)
    closed_positions = (
        db.execute(
            select(ExecutionPosition).where(
                ExecutionPosition.trade_date == today,
                ExecutionPosition.status == "CLOSED",
            )
        )
        .scalars()
        .all()
    )
    closed_positions = [
        row for row in closed_positions if _position_matches_mode(row, runtime_mode)
    ]
    pnl_values = [float(row.realized_pnl or row.pnl_value or 0.0) for row in closed_positions]
    wins = sum(1 for value in pnl_values if value > 0.0)
    open_positions = refresh_open_positions_snapshot(db, settings=settings)
    paper = compute_paper_portfolio_metrics(db, settings=settings)
    unpriced_positions = [
        row
        for row in open_positions
        if str((row.metadata_json or {}).get("latest_quote_status") or "").lower() == "unavailable"
    ]
    unpriced_unrealized_pnl = round(
        sum(float(row.unrealized_pnl or 0.0) for row in unpriced_positions),
        2,
    )
    return {
        "win_rate": round((wins / len(closed_positions) * 100.0) if closed_positions else 0.0, 2),
        "total_pnl_today": round(sum(pnl_values), 2),
        "open_positions_count": len(open_positions),
        "open_positions_unrealized_pnl": round(
            sum(float(row.unrealized_pnl or 0.0) for row in open_positions),
            2,
        ),
        "open_positions_unpriced_count": len(unpriced_positions),
        "open_positions_unpriced_unrealized_pnl": unpriced_unrealized_pnl,
        "total_trades_today": len(closed_positions),
        "wins_today": wins,
        "paper_starting_balance": float(paper["starting_balance"]),
        "paper_available_balance": float(paper["available_balance"]),
        "paper_invested_amount": float(paper["invested_amount"]),
        "paper_realized_pnl": float(paper["realized_pnl"]),
        "paper_unrealized_pnl": float(paper["unrealized_pnl"]),
        "paper_total_pnl": float(paper["total_pnl"]),
        "paper_equity": float(paper["equity"]),
    }


def _notification_payload(settings: Settings | None = None) -> dict[str, Any]:
    cfg = settings or get_settings()
    return {
        "smtp_enabled": bool(cfg.smtp_enabled),
        "smtp_ready": smtp_ready(cfg),
        "from_email": cfg.smtp_from_email.strip() or None,
        "recipient_count": len(cfg.smtp_recipients),
    }


def _calendar_payload(*, option_selection: OptionSelection | None = None) -> dict[str, Any]:
    now = datetime.now(IST_ZONE)
    today = now.date()
    session_start, session_end = market_session_bounds(today)
    expiry_dates = {item.isoformat() for item in option_selection.available_expiries} if option_selection is not None else set()
    session_status = "closed"
    if is_trading_day(today):
        if now < session_start:
            session_status = "pre_open"
        elif now <= session_end:
            session_status = "open"

    month_start = today.replace(day=1)
    days_in_month = month_calendar.monthrange(month_start.year, month_start.month)[1]
    month_days = []
    for day_number in range(1, days_in_month + 1):
        current_day = date(month_start.year, month_start.month, day_number)
        iso_day = current_day.isoformat()
        month_days.append(
            {
                "date": iso_day,
                "day": day_number,
                "weekday": current_day.strftime("%a"),
                "is_today": current_day == today,
                "is_trading_day": is_trading_day(current_day),
                "is_expiry": iso_day in expiry_dates,
            }
        )

    upcoming_days = []
    for offset in range(14):
        current_day = today + timedelta(days=offset)
        iso_day = current_day.isoformat()
        upcoming_days.append(
            {
                "date": iso_day,
                "label": current_day.strftime("%a, %d %b"),
                "is_trading_day": is_trading_day(current_day),
                "is_expiry": iso_day in expiry_dates,
            }
        )

    return {
        "timezone": "Asia/Kolkata",
        "now_ist": now.isoformat(),
        "today_ist": today.isoformat(),
        "session_status": session_status,
        "is_trading_day_today": is_trading_day(today),
        "market_session": {
            "start": session_start.isoformat(),
            "end": session_end.isoformat(),
        },
        "previous_trading_day": previous_trading_day(today).isoformat(),
        "next_trading_day": next_trading_day(today).isoformat(),
        "current_month": {
            "label": now.strftime("%B %Y"),
            "month": month_start.strftime("%Y-%m"),
            "leading_blanks": month_start.weekday(),
            "days": month_days,
        },
        "upcoming_days": upcoming_days,
    }


def _attach_execution_signal_markers(
    db: Session,
    payload: dict[str, Any],
    *,
    symbol: str,
) -> dict[str, Any]:
    candles = list(payload.get("candles") or [])
    if not candles:
        return payload
    start_ts = _parse_iso_datetime(candles[0].get("x"))
    end_ts = _parse_iso_datetime(candles[-1].get("x"))
    if start_ts is None or end_ts is None:
        return payload

    rows = (
        db.execute(
            select(SignalLog)
            .where(
                and_(
                    symbol_value_filter(SignalLog.symbol, symbol),
                    SignalLog.timestamp >= start_ts,
                    SignalLog.timestamp <= end_ts,
                    SignalLog.pine_signal.in_(["BUY", "SELL"]),
                )
            )
            .order_by(SignalLog.timestamp.asc(), SignalLog.id.asc())
            .limit(500)
        )
        .scalars()
        .all()
    )
    markers_by_key = {
        (str(marker.get("time") or ""), str(marker.get("text") or "").upper()): dict(marker)
        for marker in payload.get("markers") or []
        if marker.get("time") and marker.get("text")
    }
    for row in rows:
        action = str(row.pine_signal or "").upper()
        timestamp = _ensure_ist(row.timestamp)
        if timestamp is None or action not in {"BUY", "SELL"}:
            continue
        marker = {
            "time": timestamp.isoformat(),
            "position": "belowBar" if action == "BUY" else "aboveBar",
            "color": "#16a34a" if action == "BUY" else "#dc2626",
            "shape": "arrowUp" if action == "BUY" else "arrowDown",
            "text": action,
            "trade_placed": bool(row.trade_placed),
        }
        markers_by_key[(marker["time"], action)] = marker

    markers = sorted(
        markers_by_key.values(),
        key=lambda marker: _parse_iso_datetime(marker.get("time")) or start_ts,
    )
    return {
        **payload,
        "markers": markers,
    }


def build_chart_payload(
    db: Session,
    *,
    symbol: str,
    range_key: str = DEFAULT_CHART_RANGE,
    interval_key: str | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    current = _ensure_ist(now) or datetime.now(IST_ZONE)
    instrument_key, display_symbol = resolve_instrument_key(db, symbol)
    range_name = str(range_key or DEFAULT_CHART_RANGE).strip().lower()
    selected_interval = normalize_interval(interval_key) if interval_key else None
    plan = _chart_range_plan(range_name, current, interval_override=selected_interval)
    source_interval = str(plan["interval"])
    latest_source_ts = _latest_chart_source_ts(db, instrument_key=instrument_key, interval=source_interval)
    earliest_source_ts = _earliest_chart_source_ts(db, instrument_key=instrument_key, interval=source_interval)
    if source_interval != LIVE_INTERVAL and latest_source_ts is None:
        latest_source_ts = _latest_chart_source_ts(db, instrument_key=instrument_key, interval=LIVE_INTERVAL)
        earliest_source_ts = _earliest_chart_source_ts(db, instrument_key=instrument_key, interval=LIVE_INTERVAL)
    if range_name != "all" and (source_interval.endswith("minute") or source_interval.endswith("hour")):
        latest_source_ts = (
            _latest_complete_intraday_ts(db, instrument_key=instrument_key, now=current)
            or latest_source_ts
        )
    cache_key = (
        instrument_key,
        range_name,
        source_interval,
        latest_source_ts.isoformat() if latest_source_ts is not None else None,
    )
    cached = _CHART_PAYLOAD_CACHE.get(cache_key)
    if cached is not None:
        return _attach_execution_signal_markers(db, cached, symbol=display_symbol)
    redis_key = f"chart:v7:{instrument_key}:{range_name}:{source_interval}:{cache_key[3] or 'none'}"
    redis_cached = redis_get_json(redis_key)
    if redis_cached is not None:
        _CHART_PAYLOAD_CACHE[cache_key] = redis_cached
        return _attach_execution_signal_markers(db, redis_cached, symbol=display_symbol)

    plan = _chart_range_plan(
        range_name,
        current,
        latest_available_ts=latest_source_ts,
        earliest_available_ts=earliest_source_ts,
        interval_override=selected_interval,
    )
    rows = _chart_rows_from_range(
        db,
        instrument_key=instrument_key,
        interval=plan["interval"],
        start_ts=plan["start_ts"],
        end_ts=plan["end_ts"],
    )
    def serialize_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "x": row["ts"].isoformat() if row["ts"] is not None else None,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"] or 0.0),
            }
            for row in source_rows
        ]

    candles = serialize_rows(rows)
    pine_overlay = _build_pine_chart_overlay(
        rows,
        interval=str(plan["interval"]),
        settings=settings,
        range_key=str(plan["key"]),
    )
    markers = list(pine_overlay.get("markers") or [])
    pine_levels = list(pine_overlay.get("levels") or [])
    interval_payloads: dict[str, Any] = {}
    payload = {
        "symbol": display_symbol,
        "instrument_key": instrument_key,
        "range": plan["key"],
        "label": plan["label"],
        "interval": plan["interval"],
        "source_interval": LIVE_INTERVAL if plan["interval"] != LIVE_INTERVAL else LIVE_INTERVAL,
        "is_resampled": plan["interval"] != LIVE_INTERVAL,
        "supports_live": bool(plan["supports_live"]),
        "start_date": plan["start_date"].isoformat(),
        "end_date": plan["end_date"].isoformat(),
        "generated_at": current.isoformat(),
        "candles": candles,
        "oldest": candles[0]["x"] if candles else None,
        "latest": candles[-1]["x"] if candles else None,
        "interval_payloads": interval_payloads,
        "markers": markers,
        "pine_levels": pine_levels,
        "available_ranges": _chart_range_options(),
        "available_intervals": _chart_interval_options(),
    }
    stale_keys = [
        key
        for key in _CHART_PAYLOAD_CACHE
        if key[0] == instrument_key and key[1] == range_name and key[2] == source_interval and key != cache_key
    ]
    for key in stale_keys:
        _CHART_PAYLOAD_CACHE.pop(key, None)
    while len(_CHART_PAYLOAD_CACHE) >= 32:
        _CHART_PAYLOAD_CACHE.pop(next(iter(_CHART_PAYLOAD_CACHE)))
    _CHART_PAYLOAD_CACHE[cache_key] = payload
    redis_set_json(
        redis_key,
        payload,
        ttl_seconds=max(1, int(getattr(settings, "redis_chart_cache_ttl_seconds", 900))),
    )
    return _attach_execution_signal_markers(db, payload, symbol=display_symbol)


def _history_payload(
    db: Session,
    *,
    instrument_key: str,
    symbol: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    now = datetime.now(IST_ZONE)
    today = now.date()
    retention_years = max(1, int(getattr(settings, "history_retention_years", 2)))
    target_start_date = today - relativedelta(years=retention_years)
    expected_start_date = target_start_date if is_trading_day(target_start_date) else next_trading_day(target_start_date)
    session_start, _session_end = market_session_bounds(today)
    if not is_trading_day(today):
        expected_end_date = previous_trading_day(today)
    elif now < session_start:
        expected_end_date = previous_trading_day(today)
    else:
        expected_end_date = today
    target_start_ts = datetime.combine(target_start_date, time.min, tzinfo=IST_ZONE)

    latest_option_quote_ts = db.scalar(
        select(func.max(OptionQuote.ts)).where(symbol_value_filter(OptionQuote.underlying_symbol, symbol))
    )
    return {
        "timezone": "Asia/Kolkata",
        "retention_years": retention_years,
        "target_start_date": target_start_date.isoformat(),
        "expected_start_date": expected_start_date.isoformat(),
        "expected_end_date": expected_end_date.isoformat(),
        "today_ist": today.isoformat(),
        "latest_option_quote_ts": _ensure_ist(latest_option_quote_ts).isoformat() if latest_option_quote_ts else None,
        "intervals": [],
        "records": {
            "option_quotes": int(
                db.scalar(
                    select(func.count(OptionQuote.id)).where(
                        and_(
                            symbol_value_filter(OptionQuote.underlying_symbol, symbol),
                            OptionQuote.ts >= target_start_ts,
                        )
                    )
                )
                or 0
            ),
            "signals": int(
                db.scalar(
                    select(func.count(SignalLog.id)).where(
                        and_(
                            symbol_value_filter(SignalLog.symbol, symbol),
                            SignalLog.timestamp >= target_start_ts,
                        )
                    )
                )
                or 0
            ),
            "orders": int(
                db.scalar(
                    select(func.count(ExecutionOrder.id)).where(
                        and_(
                            symbol_value_filter(ExecutionOrder.symbol, symbol),
                            ExecutionOrder.created_at >= target_start_ts,
                        )
                    )
                )
                or 0
            ),
            "closed_trades": int(
                db.scalar(
                    select(func.count(ExecutionPosition.id)).where(
                        and_(
                            symbol_value_filter(ExecutionPosition.symbol, symbol),
                            ExecutionPosition.status == "CLOSED",
                            ExecutionPosition.opened_at >= target_start_ts,
                        )
                    )
                )
                or 0
            ),
        },
    }


def _chart_payload(context: MarketContext, db: Session) -> dict[str, Any]:
    return build_chart_payload(db, symbol=context.symbol, range_key=DEFAULT_CHART_RANGE)


def build_live_price_update(
    db: Session,
    *,
    symbol: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    instrument_key, display_symbol = resolve_instrument_key(db, symbol)
    latest_row = db.scalar(
        select(RawCandle)
        .where(and_(RawCandle.instrument_key == instrument_key, RawCandle.interval == LIVE_INTERVAL))
        .order_by(RawCandle.ts.desc())
        .limit(1)
    )
    if latest_row is None:
        raise ValueError(f"No {LIVE_INTERVAL} candles found for {display_symbol}")

    latest_ts = _ensure_ist(latest_row.ts)
    current_open = float(latest_row.open)
    current_close = float(latest_row.close)
    change = current_close - current_open
    change_pct = (change / current_open * 100.0) if current_open else 0.0
    candle = {
        "x": latest_ts.isoformat() if latest_ts is not None else None,
        "open": current_open,
        "high": float(latest_row.high),
        "low": float(latest_row.low),
        "close": current_close,
        "volume": float(latest_row.volume or 0.0),
    }
    return {
        "generated_at": datetime.now(IST_ZONE).isoformat(),
        "symbol": display_symbol,
        "instrument_key": instrument_key,
        "price": {
            "last": round(current_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "open": current_open,
            "high": float(latest_row.high),
            "low": float(latest_row.low),
            "close": current_close,
        },
        "freshness": _freshness_payload(
            db,
            instrument_key=instrument_key,
            symbol=display_symbol,
            latest_candle_ts=latest_ts,
        ),
        "stream": _stream_diagnostics_payload(
            db,
            instrument_key=instrument_key,
            latest_candle_ts=latest_ts,
            settings=settings,
        ),
        "candle": candle,
    }


def build_live_snapshot(
    db: Session,
    *,
    symbol: str,
    settings: Settings | None = None,
    include_static: bool = True,
    include_chart: bool = True,
    include_option: bool = True,
) -> dict[str, Any]:
    settings = settings or get_settings()
    context = load_market_context(db, symbol=symbol, settings=settings)
    signal = build_technical_signal(db, context=context, settings=settings)
    option_selection = (
        build_option_selection(db, context=context, signal=signal, settings=settings)
        if include_option
        else None
    )

    open_positions = refresh_open_positions_snapshot(db, settings=settings)
    runtime_mode = get_runtime_trading_mode(db, settings=settings)
    recent_trades = (
        db.execute(
            select(ExecutionPosition)
            .where(
                and_(
                    ExecutionPosition.status == "CLOSED",
                    symbol_value_filter(ExecutionPosition.symbol, context.symbol),
                )
            )
            .order_by(ExecutionPosition.closed_at.desc())
            .limit(100)
        )
        .scalars()
        .all()
    )
    recent_trades = [row for row in recent_trades if _position_matches_mode(row, runtime_mode)][:15]
    recent_orders = (
        db.execute(
            select(ExecutionOrder)
            .where(symbol_value_filter(ExecutionOrder.symbol, context.symbol))
            .order_by(ExecutionOrder.created_at.desc())
            .limit(100)
        )
        .scalars()
        .all()
    )
    recent_orders = [row for row in recent_orders if _order_matches_mode(row, runtime_mode)][:20]
    recent_signals = (
        db.execute(
            select(SignalLog)
            .where(
                and_(
                    SignalLog.trade_date == datetime.now(IST_ZONE).date(),
                    symbol_value_filter(SignalLog.symbol, context.symbol),
                )
            )
            .order_by(SignalLog.id.desc())
            .limit(10)
        )
        .scalars()
        .all()
    )
    current_bar = context.current_bar
    change = float(current_bar["close"] or 0.0) - float(current_bar["open"] or 0.0)
    change_pct = (change / float(current_bar["open"] or 1.0)) * 100.0
    payload = {
        "generated_at": datetime.now(IST_ZONE).isoformat(),
        "symbol": context.symbol,
        "instrument_key": context.instrument_key,
        "execution": {
            "mode": runtime_mode,
            "max_daily_loss_amount": round(float(settings.execution_capital) * float(getattr(settings, "execution_max_daily_loss_pct", 0.05)), 2),
        },
        "price": {
            "last": round(context.latest_price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "open": current_bar["open"],
            "high": current_bar["high"],
            "low": current_bar["low"],
            "close": current_bar["close"],
        },
        "freshness": _freshness_payload(
            db,
            instrument_key=context.instrument_key,
            symbol=context.symbol,
            latest_candle_ts=context.latest_candle_ts,
        ),
        "stream": _stream_diagnostics_payload(
            db,
            instrument_key=context.instrument_key,
            latest_candle_ts=context.latest_candle_ts,
            settings=settings,
        ),
        "stats": _stats_payload(db, settings=settings),
        "signal": {
            "enabled": bool(DIRECTIONAL_SIGNALS_ENABLED),
            "action": signal.action,
            "bias": signal.bias,
            "score": signal.score,
            "confidence": signal.confidence,
            "conviction": signal.conviction,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "cooldown_seconds": signal.cooldown_seconds,
            "max_signals_reached": signal.max_signals_reached,
            "reasons": signal.reasons,
            "details": signal.details,
        },
        "option": {
            "expiry_date": option_selection.expiry_date.isoformat() if option_selection is not None else None,
            "available_expiries": [item.isoformat() for item in option_selection.available_expiries] if option_selection is not None else [],
            "strike_step": option_selection.strike_step if option_selection is not None else None,
            "chain_source": option_selection.chain_source if option_selection is not None else "deferred",
            "chain_generated_at": (
                option_selection.chain_generated_at.isoformat()
                if option_selection is not None and option_selection.chain_generated_at is not None
                else None
            ),
            "signal": option_selection.signal if option_selection is not None else {
                "action": "HOLD",
                "option_type": None,
                "strike": None,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "confidence": signal.confidence,
                "reasons": ["Option selection deferred during fast symbol switch."],
            },
        },
        "positions": [_serialize_position(row) for row in open_positions],
        "recent_trades": [_serialize_position(row) for row in recent_trades],
        "recent_orders": [_serialize_order(row) for row in recent_orders],
        "recent_signals": [_serialize_signal_log(row) for row in recent_signals],
    }
    if include_chart:
        payload["chart"] = _chart_payload(context, db)
    if include_static:
        payload["calendar"] = _calendar_payload(option_selection=option_selection)
        payload["history"] = _history_payload(
            db,
            instrument_key=context.instrument_key,
            symbol=context.symbol,
            settings=settings,
        )
        payload["notifications"] = _notification_payload(settings=settings)
    return payload
