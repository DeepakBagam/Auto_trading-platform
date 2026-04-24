from __future__ import annotations

import calendar as month_calendar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
from dateutil.relativedelta import relativedelta
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from api.market_stream_runtime import get_market_stream_runtime_status
from data_layer.collectors.upstox_option_chain import UpstoxOptionChainCollector
from db.models import DataFreshness, DailySummary, ExecutionOrder, ExecutionPosition, OptionQuote, RawCandle, SignalLog
from execution_engine.intraday_rules import (
    DEFAULT_EXECUTION_CONSTRAINTS,
    adaptive_stop_points,
    ema_separation_floor,
    ema_separation_is_valid,
    runner_target_points,
)
from execution_engine.risk_manager import build_risk_plan
from execution_engine.slippage_tracker import get_vix_context, get_vix_level
from execution_engine.strike_selector import get_atm_iv as _strike_get_atm_iv
from execution_engine.strike_selector import select_option_contract
from feature_engine.price_features import build_price_features
from prediction_engine.options_engine import (
    OptionQuoteView,
    build_chain_rows,
    nearest_strike,
    next_weekly_expiries,
    strike_step_for_symbol,
    synthetic_option_chain,
)
from utils.calendar_utils import is_trading_day, market_session_bounds, next_trading_day, previous_trading_day
from utils.config import Settings, get_settings
from utils.constants import IST_ZONE
from utils.notifications import smtp_ready
from utils.symbols import (
    canonical_symbol_name,
    display_symbol_from_instrument_key,
    instrument_key_filter,
    sort_display_symbols,
    symbol_aliases,
    symbol_value_filter,
)

LIVE_INTERVAL = "1minute"
DIRECTIONAL_SIGNALS_ENABLED = True
DEFAULT_SIGNAL_COOLDOWN_MINUTES = 12
DEFAULT_SIGNAL_MIN_SCORE = 63.0
VIX_MAX_THRESHOLD = 20.0   # Skip signals when VIX is too high (options too expensive)
VIX_MIN_THRESHOLD = 11.0   # Skip signals when VIX is too low (premiums too small)
DEFAULT_MAX_SIGNALS_PER_DAY = 2
DEFAULT_CHART_RANGE = "1d"
CHART_RANGE_SPECS: dict[str, dict[str, Any]] = {
    "1d": {"label": "1D Live", "interval": "1minute", "days": 1, "supports_live": True},
    "5d": {"label": "5D", "interval": "1minute", "days": 7, "supports_live": True},
    "1m": {"label": "1M", "interval": "30minute", "days": 31, "supports_live": False},
    "6m": {"label": "6M", "interval": "day", "days": 183, "supports_live": False},
    "1y": {"label": "1Y", "interval": "day", "days": 366, "supports_live": False},
    "2y": {"label": "2Y", "interval": "day", "years": 2, "supports_live": False},
}
CHART_CONFIRMATION_RULES: dict[str, tuple[str, str]] = {
    "1minute": ("3min", "5min"),
    "30minute": ("90min", "150min"),
    "day": ("3D", "5D"),
}
CHART_MARKER_LIMITS: dict[str, int] = {
    "1d": 40,
    "5d": 80,
    "1m": 60,
    "6m": 40,
    "1y": 30,
    "2y": 30,
}
_CHART_PAYLOAD_CACHE: dict[tuple[str, str, str | None], dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Market regime constants
# ---------------------------------------------------------------------------
_REGIME_TRENDING = "TRENDING"
_REGIME_RANGE = "RANGE_BOUND"
_REGIME_VOLATILE = "HIGH_VOLATILITY"

# Data staleness: if last candle is older than this, halt signal generation.
_DATA_STALE_SECONDS = 600


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
    key = db.scalar(
        select(RawCandle.instrument_key)
        .where(instrument_key_filter(RawCandle.instrument_key, symbol))
        .order_by(RawCandle.instrument_key.asc())
        .limit(1)
    )
    if key is None:
        raise ValueError(f"Symbol not found in candles: {symbol}")
    display = canonical_symbol_name(display_symbol_from_instrument_key(str(key)))
    return str(key), display


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
) -> dict[str, Any]:
    current = _ensure_ist(now) or datetime.now(IST_ZONE)
    key = str(range_key or DEFAULT_CHART_RANGE).strip().lower()
    if key not in CHART_RANGE_SPECS:
        key = DEFAULT_CHART_RANGE
    spec = CHART_RANGE_SPECS[key]
    end_date = _expected_chart_end_date(current, latest_available_ts=latest_available_ts)
    if "years" in spec:
        start_date = end_date - relativedelta(years=int(spec["years"]))
    else:
        start_date = end_date - timedelta(days=max(1, int(spec.get("days", 1))) - 1)
    if not is_trading_day(start_date):
        start_date = next_trading_day(start_date)
    start_ts = datetime.combine(start_date, time.min, tzinfo=IST_ZONE)
    if spec["interval"] == "1minute":
        start_ts = datetime.combine(start_date, time(9, 15), tzinfo=IST_ZONE)
    return {
        "key": key,
        "label": spec["label"],
        "interval": spec["interval"],
        "supports_live": bool(spec["supports_live"]),
        "start_date": start_date,
        "end_date": end_date,
        "start_ts": start_ts,
        "end_ts": current,
    }


def _chart_rows_from_range(
    db: Session,
    *,
    instrument_key: str,
    interval: str,
    start_ts: datetime,
) -> list[dict[str, Any]]:
    target_rows = (
        db.execute(
            select(RawCandle)
            .where(
                and_(
                    RawCandle.instrument_key == instrument_key,
                    RawCandle.interval == interval,
                    RawCandle.ts >= start_ts,
                )
            )
            .order_by(RawCandle.ts.asc())
        )
        .scalars()
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
            select(RawCandle)
            .where(
                and_(
                    RawCandle.instrument_key == instrument_key,
                    RawCandle.interval == LIVE_INTERVAL,
                    RawCandle.ts >= start_ts,
                )
            )
            .order_by(RawCandle.ts.asc())
        )
        .scalars()
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
    rule = "90min" if interval == "30minute" else "1D"
    if interval == "30minute":
        rule = "30min"
    elif interval == "day":
        rule = "1D"
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


def _build_chart_markers(
    rows: list[dict[str, Any]],
    *,
    interval: str,
    settings: Settings,
    range_key: str,
) -> list[dict[str, Any]]:
    if not DIRECTIONAL_SIGNALS_ENABLED:
        return []
    if len(rows) < 60:
        return []

    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"])
    features = build_price_features(frame.copy())
    if features.empty or len(features) < 60:
        return []

    confirm_rule_3, confirm_rule_5 = CHART_CONFIRMATION_RULES.get(interval, ("3min", "5min"))
    confirm_3 = _timeframe_confirmation_columns(build_price_features(_resample_chart_frame(frame, confirm_rule_3)))
    confirm_5 = _timeframe_confirmation_columns(build_price_features(_resample_chart_frame(frame, confirm_rule_5)))
    merged = pd.merge_asof(features.sort_values("ts"), confirm_3.sort_values("ts"), on="ts", direction="backward")
    merged = pd.merge_asof(
        merged.sort_values("ts"),
        confirm_5.sort_values("ts"),
        on="ts",
        direction="backward",
        suffixes=("_3", "_5"),
    )
    for column in ("confirm_buy_3", "confirm_sell_3", "confirm_buy_5", "confirm_sell_5"):
        if column in merged:
            normalized = pd.array(merged[column], dtype="boolean")
            merged[column] = pd.Series(normalized, index=merged.index).fillna(False).astype(bool)
        else:
            merged[column] = pd.Series(False, index=merged.index, dtype=bool)

    cooldown_minutes = max(1, int(getattr(settings, "signal_cooldown_minutes", DEFAULT_SIGNAL_COOLDOWN_MINUTES)))
    max_signals_today = max(1, int(getattr(settings, "signal_max_per_day", DEFAULT_MAX_SIGNALS_PER_DAY)))
    min_score = float(getattr(settings, "signal_min_score", DEFAULT_SIGNAL_MIN_SCORE))
    entry_start = _parse_time(settings.entry_window_start, time(9, 20))
    entry_end = _parse_time(settings.entry_window_end, time(12, 30))
    second_trade_entry_end = _parse_time(getattr(settings, "second_trade_entry_end", "11:00"), time(11, 0))

    markers: list[dict[str, Any]] = []
    signal_count_by_day: dict[date, int] = {}
    last_signal_ts: datetime | None = None

    for index in range(1, len(merged)):
        row = merged.iloc[index]
        prev = merged.iloc[index - 1]
        ts = _ensure_ist(row.get("ts").to_pydatetime() if hasattr(row.get("ts"), "to_pydatetime") else row.get("ts"))
        if ts is None:
            continue

        close = _to_float(row.get("close")) or 0.0
        high = _to_float(row.get("high")) or close
        low = _to_float(row.get("low")) or close
        prev_high = _to_float(prev.get("high")) or high
        prev_low = _to_float(prev.get("low")) or low
        ema_21 = _to_float(row.get("ema_21")) or close
        ema_50 = _to_float(row.get("ema_50")) or close
        ema_21_slope = _to_float(row.get("ema_21_slope_3")) or 0.0
        vwap = _to_float(row.get("vwap")) or close
        rsi = _to_float(row.get("rsi_14")) or 50.0
        macd_hist = _to_float(row.get("macd_hist")) or 0.0
        macd_delta = _to_float(row.get("macd_hist_delta_1")) or 0.0
        body_pct_range = _to_float(row.get("body_pct_range")) or 0.0
        volume_ratio = _to_float(row.get("volume_ratio_20")) or 1.0
        breakout_high = _to_float(row.get("breakout_high_20"))
        breakout_low = _to_float(row.get("breakout_low_20"))
        atr = max(1e-9, _to_float(row.get("atr_14")) or 0.0)
        candle_range = max(0.0, high - low)

        trend_buy = close > ema_21 > ema_50 and close >= vwap and ema_21_slope > 0.0
        trend_sell = close < ema_21 < ema_50 and close <= vwap and ema_21_slope < 0.0
        breakout_buy = bool(
            breakout_high is not None
            and close > breakout_high
            and body_pct_range >= 0.52
            and candle_range >= atr * 0.75
        )
        breakout_sell = bool(
            breakout_low is not None
            and close < breakout_low
            and body_pct_range >= 0.52
            and candle_range >= atr * 0.75
        )
        continuation_buy = bool(
            trend_buy
            and close > prev_high
            and 56.0 <= rsi <= 72.0
            and macd_hist > 0.0
            and macd_delta >= -0.02
        )
        continuation_sell = bool(
            trend_sell
            and close < prev_low
            and 28.0 <= rsi <= 44.0
            and macd_hist < 0.0
            and macd_delta <= 0.02
        )
        range_bound = abs(close - ema_21) <= max(atr * 0.25, close * 0.0008) and 45.0 <= rsi <= 55.0

        score_buy = 0.0
        score_sell = 0.0
        if trend_buy:
            score_buy += 28.0
        if trend_sell:
            score_sell += 28.0
        if breakout_buy:
            score_buy += 28.0
        elif continuation_buy:
            score_buy += 22.0
        if breakout_sell:
            score_sell += 28.0
        elif continuation_sell:
            score_sell += 22.0
        if bool(row.get("confirm_buy_3")):
            score_buy += 18.0
        if bool(row.get("confirm_sell_3")):
            score_sell += 18.0
        if bool(row.get("confirm_buy_5")):
            score_buy += 14.0
        if bool(row.get("confirm_sell_5")):
            score_sell += 14.0
        if rsi >= 58.0 and macd_hist > 0.0:
            score_buy += 8.0
        if rsi <= 42.0 and macd_hist < 0.0:
            score_sell += 8.0
        if volume_ratio >= 1.10 or candle_range >= atr:
            score_buy += 4.0
            score_sell += 4.0

        action = "BUY" if score_buy >= score_sell else "SELL"
        score = score_buy if action == "BUY" else score_sell
        if range_bound or score < min_score:
            continue
        if action == "BUY" and not (breakout_buy or continuation_buy):
            continue
        if action == "SELL" and not (breakout_sell or continuation_sell):
            continue
        if interval != "day":
            now_time = ts.timetz().replace(tzinfo=None)
            if not (entry_start <= now_time <= entry_end):
                continue
        if last_signal_ts is not None:
            elapsed = int((ts - last_signal_ts).total_seconds())
            if elapsed < cooldown_minutes * 60:
                continue
        day_count = signal_count_by_day.get(ts.date(), 0)
        if day_count >= max_signals_today:
            continue

        signal_count_by_day[ts.date()] = day_count + 1
        last_signal_ts = ts
        markers.append(
            {
                "time": ts.isoformat(),
                "position": "belowBar" if action == "BUY" else "aboveBar",
                "color": "#00b16a" if action == "BUY" else "#ff6b35",
                "shape": "arrowUp" if action == "BUY" else "arrowDown",
                "text": f"{action} {int(score)}",
            }
        )
    limit = max(1, CHART_MARKER_LIMITS.get(range_key, 40))
    return markers[-limit:]


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
                    interval=LIVE_INTERVAL,
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
                    details={"data_stale": True, "last_candle_age_seconds": age_seconds},
                )

    if len(rows) < 60:
        return TechnicalSignal(
            symbol=context.symbol,
            interval=LIVE_INTERVAL,
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
            details={"warmup_candles": len(rows)},
        )

    # --- Build features ---
    frame = build_price_features(_candles_to_frame(rows))
    row = frame.iloc[-1]
    prev = frame.iloc[-2]
    prev2 = frame.iloc[-3] if len(frame) >= 3 else prev

    # --- Multi-timeframe confirmation ---
    frame_3m = _resample_frame(frame[["ts", "open", "high", "low", "close", "volume"]], "3min")
    frame_5m = _resample_frame(frame[["ts", "open", "high", "low", "close", "volume"]], "5min")
    confirm_3m_buy, confirm_3m_sell = _timeframe_confirmation(frame_3m)
    confirm_5m_buy, confirm_5m_sell = _timeframe_confirmation(frame_5m)

    # --- Extract all indicators ---
    close = _to_float(row.get("close")) or context.latest_price
    open_ = _to_float(row.get("open")) or close
    high = _to_float(row.get("high")) or close
    low = _to_float(row.get("low")) or close
    prev_high = _to_float(prev.get("high")) or high
    prev_low = _to_float(prev.get("low")) or low
    prev_close = _to_float(prev.get("close")) or close

    ema_21 = _to_float(row.get("ema_21")) or close
    ema_50 = _to_float(row.get("ema_50")) or close
    ema_21_slope = _to_float(row.get("ema_21_slope_3")) or 0.0
    ema_separation = abs(ema_21 - ema_50)
    vwap = _to_float(row.get("vwap")) or close
    rsi = _to_float(row.get("rsi_14")) or 50.0
    macd_hist = _to_float(row.get("macd_hist")) or 0.0
    macd_delta = _to_float(row.get("macd_hist_delta_1")) or 0.0
    body_pct_range = _to_float(row.get("body_pct_range")) or 0.0
    volume_ratio = _to_float(row.get("volume_ratio_20")) or 1.0
    breakout_high = _to_float(row.get("breakout_high_20"))
    breakout_low = _to_float(row.get("breakout_low_20"))
    atr = max(1e-9, _to_float(row.get("atr_14")) or 0.0)
    atr_mean = max(atr, _to_float(row.get("atr_14_mean_20")) or atr)
    adx = _to_float(row.get("adx_14")) or 15.0
    plus_di = _to_float(row.get("plus_di_14")) or 20.0
    minus_di = _to_float(row.get("minus_di_14")) or 20.0
    candle_range = max(0.0, high - low)
    execution_rules = DEFAULT_EXECUTION_CONSTRAINTS

    # --- Market regime detection ---
    regime = _detect_regime(adx, plus_di, minus_di, atr, atr_mean)

    # --- Relative VIX filter (not fixed threshold) ---
    vix_level, vix_ma20, vix_ratio = get_vix_context(db)
    # Spike: 40%+ above own MA → options too expensive
    vix_too_high = vix_ratio > 1.40
    # Dead calm: 30%+ below MA AND absolute level below 12 → premium too small
    vix_too_low = vix_ratio < 0.70 and vix_level < 12.0

    # --- Expiry-day awareness ---
    _today = now.date()
    try:
        _expiries = next_weekly_expiries(symbol=context.symbol, count=2, start_dt=now)
        days_to_expiry = int((_expiries[0] - _today).days) if _expiries else 7
    except Exception:
        days_to_expiry = 7

    # --- Adaptive RSI thresholds ---
    rsi_buy_lo, rsi_buy_hi, rsi_sell_lo, rsi_sell_hi = _rsi_buy_sell_bands(regime)

    # --- Primary structural conditions ---
    trend_buy = close > ema_21 > ema_50 and close >= vwap and ema_21_slope > 0.0
    trend_sell = close < ema_21 < ema_50 and close <= vwap and ema_21_slope < 0.0

    # 2-candle confirmed breakout: body must close above level, prev bar had to approach it
    breakout_buy = bool(
        breakout_high is not None
        and close > breakout_high        # Body close above (green candle only)
        and open_ < close                # Green candle — not a wick tag
        and body_pct_range >= 0.55       # Solid body, not a doji
        and candle_range >= atr * 0.75
        and prev_close >= breakout_high * 0.996  # Prev bar was testing the level
    )
    breakout_sell = bool(
        breakout_low is not None
        and close < breakout_low
        and open_ > close                # Red candle
        and body_pct_range >= 0.55
        and candle_range >= atr * 0.75
        and prev_close <= breakout_low * 1.004
    )
    continuation_buy = bool(
        trend_buy
        and close > prev_high
        and rsi_buy_lo <= rsi <= rsi_buy_hi
        and macd_hist > 0.0
        and macd_delta >= -0.02
    )
    continuation_sell = bool(
        trend_sell
        and close < prev_low
        and rsi_sell_lo <= rsi <= rsi_sell_hi
        and macd_hist < 0.0
        and macd_delta <= 0.02
    )
    range_bound = abs(close - ema_21) <= max(atr * 0.25, close * 0.0008) and 45.0 <= rsi <= 55.0
    ema_sep_ok = ema_separation_is_valid(
        ema_21=ema_21,
        ema_50=ema_50,
        atr=atr,
        close=close,
        constraints=execution_rules,
    )

    # --- Regime-aware scoring ---
    score_buy = 0.0
    score_sell = 0.0
    reasons_buy: list[str] = []
    reasons_sell: list[str] = []

    if trend_buy:
        score_buy += 28.0
        reasons_buy.append("Price stacked above EMA 21/50 with VWAP support.")
    if trend_sell:
        score_sell += 28.0
        reasons_sell.append("Price stacked below EMA 21/50 with VWAP resistance.")

    if breakout_buy:
        score_buy += 28.0
        reasons_buy.append("2-bar confirmed breakout above 20-bar high.")
    elif continuation_buy:
        score_buy += 22.0
        reasons_buy.append("Bullish continuation above prior bar high.")

    if breakout_sell:
        score_sell += 28.0
        reasons_sell.append("2-bar confirmed breakdown below 20-bar low.")
    elif continuation_sell:
        score_sell += 22.0
        reasons_sell.append("Bearish continuation below prior bar low.")

    if confirm_3m_buy:
        score_buy += 18.0
        reasons_buy.append("3-minute timeframe confirms bullish alignment.")
    if confirm_3m_sell:
        score_sell += 18.0
        reasons_sell.append("3-minute timeframe confirms bearish alignment.")

    if confirm_5m_buy:
        score_buy += 14.0
        reasons_buy.append("5-minute timeframe aligned long.")
    if confirm_5m_sell:
        score_sell += 14.0
        reasons_sell.append("5-minute timeframe aligned short.")

    # Adaptive RSI+MACD momentum bonus
    if rsi >= (rsi_buy_lo + 5) and macd_hist > 0.0:
        score_buy += 8.0
    if rsi <= (rsi_sell_hi - 5) and macd_hist < 0.0:
        score_sell += 8.0

    if volume_ratio >= 1.10 or candle_range >= atr:
        score_buy += 4.0
        score_sell += 4.0

    # ADX regime bonus: strong directional market → reward aligned side
    if regime == _REGIME_TRENDING and adx >= 30.0:
        if plus_di > minus_di:
            score_buy += 5.0
            reasons_buy.append(f"ADX {adx:.0f} strong uptrend (+DI dominant).")
        else:
            score_sell += 5.0
            reasons_sell.append(f"ADX {adx:.0f} strong downtrend (-DI dominant).")

    # Range suppression: break-even both sides in choppy market
    if regime == _REGIME_RANGE:
        score_buy *= 0.85
        score_sell *= 0.85

    # --- Direction & dynamic threshold ---
    bias = "BUY" if score_buy > score_sell else ("SELL" if score_sell > score_buy else "NEUTRAL")
    raw_action = "BUY" if score_buy >= score_sell else "SELL"
    raw_score = score_buy if raw_action == "BUY" else score_sell
    base_reasons = reasons_buy if raw_action == "BUY" else reasons_sell

    min_score = _dynamic_threshold(regime, vix_ratio)

    # --- Guard rails ---
    max_signals_today = max(1, int(getattr(settings, "signal_max_per_day", DEFAULT_MAX_SIGNALS_PER_DAY)))
    signal_count_today, cooldown_seconds = _signal_guardrails(
        db, symbol=context.symbol, now=now, settings=settings,
    )
    entry_start = _parse_time(settings.entry_window_start, time(9, 20))
    entry_end = _parse_time(settings.entry_window_end, time(12, 30))
    second_trade_entry_end = _parse_time(getattr(settings, "second_trade_entry_end", "11:00"), time(11, 0))

    # Expiry day: cut off entries after 11:00
    expiry_cutoff = _expiry_entry_cutoff(days_to_expiry)
    if expiry_cutoff is not None:
        entry_end = min(entry_end, expiry_cutoff)

    now_time = now.timetz().replace(tzinfo=None)
    reasons = list(base_reasons)
    action = raw_action

    if vix_too_high:
        action = "HOLD"
        reasons.append(
            f"VIX {vix_level:.1f} is {(vix_ratio - 1) * 100:.0f}% above its MA — options overpriced."
        )
    if vix_too_low:
        action = "HOLD"
        reasons.append(f"VIX {vix_level:.1f} unusually low — option premiums too small.")
    if adx < execution_rules.min_adx:
        action = "HOLD"
        reasons.append(f"ADX {adx:.1f} below live minimum {execution_rules.min_adx:.0f}.")
    if not ema_sep_ok:
        sep_floor = ema_separation_floor(atr=atr, close=close, constraints=execution_rules)
        action = "HOLD"
        reasons.append(f"EMA 21/50 separation {ema_separation:.1f} below required {sep_floor:.1f}.")
    if range_bound:
        action = "HOLD"
        reasons.append("Market is compressed around EMA 21 — no clean edge.")
    # In range-bound regime, only allow confirmed breakouts
    if regime == _REGIME_RANGE and not (breakout_buy or breakout_sell):
        action = "HOLD"
        reasons.append("Range-bound market — only confirmed breakouts are allowed.")
    if raw_score < min_score:
        action = "HOLD"
        reasons.append(f"Score {raw_score:.0f} below adaptive threshold {min_score:.0f} ({regime}).")
    if raw_action == "BUY" and not (breakout_buy or continuation_buy):
        action = "HOLD"
        reasons.append("Long side has no breakout or continuation trigger.")
    if raw_action == "SELL" and not (breakout_sell or continuation_sell):
        action = "HOLD"
        reasons.append("Short side has no breakout or continuation trigger.")
    if not (entry_start <= now_time <= entry_end):
        action = "HOLD"
        reasons.append(
            f"Outside entry window {entry_start.strftime('%H:%M')}–{entry_end.strftime('%H:%M')} IST."
        )
    if signal_count_today >= 1 and now_time > second_trade_entry_end:
        action = "HOLD"
        reasons.append(f"Second-trade cutoff passed at {second_trade_entry_end.strftime('%H:%M')} IST.")
    if cooldown_seconds > 0:
        action = "HOLD"
        reasons.append(f"Cooldown active — {cooldown_seconds}s remaining.")
    max_reached = signal_count_today >= max_signals_today
    if max_reached:
        action = "HOLD"
        reasons.append(f"Daily signal cap reached ({signal_count_today}/{max_signals_today}).")
    # Expiry day + non-trending market = skip (gamma + directionless = bad combination)
    if days_to_expiry == 0 and regime != _REGIME_TRENDING:
        action = "HOLD"
        reasons.append("Expiry day with non-trending market — skipping to avoid gamma risk.")

    # --- Regime-aware exit levels ---
    stop_offset = adaptive_stop_points(atr=atr, constraints=execution_rules)
    t1_offset = runner_target_points(stop_points=stop_offset, constraints=execution_rules)
    t2_offset = t1_offset

    if action == "BUY":
        stop_loss = round(close - stop_offset, 2)
        take_profit = round(close + t1_offset, 2)
        take_profit_2 = round(close + t2_offset, 2)
    elif action == "SELL":
        stop_loss = round(close + stop_offset, 2)
        take_profit = round(close - t1_offset, 2)
        take_profit_2 = round(close - t2_offset, 2)
    else:
        stop_loss = None
        take_profit = None
        take_profit_2 = None

    confidence = round(_clip(raw_score / 100.0, 0.0, 0.95), 2)
    conviction = "high" if raw_score >= 88.0 else ("medium" if raw_score >= min_score else "low")
    expected_move = round(atr * 1.5, 2)
    expected_move_pct = round(expected_move / max(close, 1e-9), 4)

    details = {
        "close": round(close, 2),
        "vwap": round(vwap, 2),
        "ema_21": round(ema_21, 2),
        "ema_50": round(ema_50, 2),
        "ema_separation": round(ema_separation, 2),
        "rsi_14": round(rsi, 2),
        "rsi_buy_band": [round(rsi_buy_lo, 1), round(rsi_buy_hi, 1)],
        "rsi_sell_band": [round(rsi_sell_lo, 1), round(rsi_sell_hi, 1)],
        "macd_hist": round(macd_hist, 4),
        "macd_hist_delta_1": round(macd_delta, 4),
        "atr_14": round(atr, 2),
        "atr_14_mean_20": round(atr_mean, 2),
        "adx_14": round(adx, 1),
        "plus_di_14": round(plus_di, 1),
        "minus_di_14": round(minus_di, 1),
        "volume_ratio_20": round(volume_ratio, 2),
        "body_pct_range": round(body_pct_range, 2),
        "breakout_high_20": round(breakout_high, 2) if breakout_high is not None else None,
        "breakout_low_20": round(breakout_low, 2) if breakout_low is not None else None,
        "trend_buy": trend_buy,
        "trend_sell": trend_sell,
        "ema_separation_ok": ema_sep_ok,
        "breakout_buy": breakout_buy,
        "breakout_sell": breakout_sell,
        "continuation_buy": continuation_buy,
        "continuation_sell": continuation_sell,
        "confirm_3m_buy": confirm_3m_buy,
        "confirm_3m_sell": confirm_3m_sell,
        "confirm_5m_buy": confirm_5m_buy,
        "confirm_5m_sell": confirm_5m_sell,
        "score_buy": round(score_buy, 1),
        "score_sell": round(score_sell, 1),
        "regime": regime,
        "adx_regime_note": f"ADX={adx:.1f} +DI={plus_di:.1f} -DI={minus_di:.1f}",
        "adaptive_threshold": round(min_score, 1),
        "days_to_expiry": days_to_expiry,
        "vix_level": round(vix_level, 2),
        "vix_ma20": round(vix_ma20, 2),
        "vix_ratio": round(vix_ratio, 3),
        "vix_too_high": vix_too_high,
        "take_profit_2": take_profit_2,
        "expected_move_points": expected_move,
        "expected_move_pct": expected_move_pct,
        "signal_count_today": signal_count_today,
        "signal_candle_ts": _ensure_ist(rows[-1].ts).isoformat() if rows else None,
        "execution_plan": {
            "stop_range_points": [execution_rules.stop_points_min, execution_rules.stop_points_max],
            "partial_exit_points": execution_rules.partial_exit_points,
            "breakeven_trigger_points": execution_rules.breakeven_trigger_points,
            "lock_trigger_points": execution_rules.lock_trigger_points,
            "lock_points": execution_rules.lock_points,
            "trail_trigger_points": execution_rules.trail_trigger_points,
            "trail_distance_points": execution_rules.trail_distance_points,
            "max_hold_minutes": execution_rules.max_hold_minutes,
        },
    }
    timestamp = _ensure_ist(rows[-1].ts) if rows else now
    return TechnicalSignal(
        symbol=context.symbol,
        interval=LIVE_INTERVAL,
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
        pine_signal="OFF",
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
    by_contract: dict[tuple[float, str], OptionQuoteView] = {}
    for row in rows:
        key = (float(row.strike), str(row.option_type))
        if key in by_contract:
            continue
        by_contract[key] = OptionQuoteView(
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
    return sorted(by_contract.values(), key=lambda item: (item.strike, item.option_type))


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
    latest_ts = _latest_option_quote_ts(db, symbol, expiry_date)
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
    quotes = _load_option_quotes(db, symbol=context.symbol, expiry_date=expiry_date)
    chain_generated_at = _latest_option_quote_ts(db, context.symbol, expiry_date)
    chain_source = next((str(item.source or "db") for item in quotes if item.source), "synthetic")
    if not quotes:
        quotes = synthetic_option_chain(
            symbol=context.symbol,
            underlying_price=context.latest_price,
            expiry_date=expiry_date,
            strike_step=strike_step,
        )
        chain_source = "synthetic"
        chain_generated_at = datetime.now(IST_ZONE)
    chain_rows = build_chain_rows(quotes)

    # Compute IV rank for this symbol using 90-day history
    _atm = nearest_strike(context.latest_price, strike_step)
    _atm_iv = _strike_get_atm_iv(chain_rows, _atm)
    iv_rank = _compute_iv_rank(db, context.symbol, _atm_iv)

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
    pick = select_option_contract(
        signal_action=signal.action,  # type: ignore[arg-type]
        spot_price=context.latest_price,
        strike_step=strike_step,
        chain_rows=chain_rows,
        confidence=signal.confidence,
        expected_return_pct=float(signal.details.get("expected_move_pct") or 0.0),
        premium_min=float(settings.execution_premium_min),
        premium_max=float(settings.execution_premium_max),
        days_to_expiry=dte,
        capital_per_trade=float(settings.execution_capital) * float(settings.execution_per_trade_risk_pct),
        iv_rank=iv_rank,
    )
    if pick is not None:
        risk_plan = build_risk_plan(
            entry_premium=float(pick.premium),
            tsl_activation_percent=float(settings.tsl_activation_percent),
            target_profit_percent=float(settings.target_profit_percent),
        )
        signal_payload = {
            "action": "BUY",
            "option_type": pick.option_type,
            "strike": float(pick.strike),
            "entry_price": float(pick.premium),
            "stop_loss": float(risk_plan.initial_sl),
            "take_profit": float(risk_plan.target_price),
            "confidence": signal.confidence,
            "instrument_key": pick.instrument_key,
            "iv_rank": iv_rank,
            "days_to_expiry": dte,
            "reasons": [
                f"{signal.action} signal mapped to {pick.option_type}.",
                "Selected best-scored strike: delta-targeted, OI-delta-weighted, PCR-filtered.",
                pick.reason,
                f"Chain source: {chain_source}.",
            ],
        }
    else:
        signal_payload = {
            "action": "HOLD",
            "option_type": None,
            "strike": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "confidence": signal.confidence,
            "reasons": ["No liquid CE/PE strike passed the live option filters."],
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


def latest_option_premium(
    db: Session,
    *,
    symbol: str,
    expiry_date: date,
    strike: float,
    option_type: str,
) -> float | None:
    premium = db.scalar(
        select(OptionQuote.ltp)
        .where(
            and_(
                symbol_value_filter(OptionQuote.underlying_symbol, symbol),
                OptionQuote.expiry_date == expiry_date,
                OptionQuote.strike == float(strike),
                OptionQuote.option_type == option_type,
            )
        )
        .order_by(OptionQuote.ts.desc())
        .limit(1)
    )
    if premium is not None:
        return float(premium)

    underlying_key, display_symbol = resolve_instrument_key(db, symbol)
    del display_symbol
    rows = _load_recent_candles(db, instrument_key=underlying_key, interval=LIVE_INTERVAL, limit=5)
    if not rows:
        return None
    quotes = synthetic_option_chain(
        symbol=symbol,
        underlying_price=float(rows[-1].close),
        expiry_date=expiry_date,
        strike_step=strike_step_for_symbol(symbol),
    )
    for quote in quotes:
        if float(quote.strike) == float(strike) and str(quote.option_type).upper() == str(option_type).upper():
            return float(quote.ltp)
    return None


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
        "premium_history": metadata.get("premium_history") or [],
    }


def _serialize_order(row: ExecutionOrder) -> dict[str, Any]:
    return {
        "id": row.id,
        "symbol": row.symbol,
        "strike_price": row.strike_price,
        "option_type": row.option_type,
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


def _stats_payload(db: Session) -> dict[str, Any]:
    today = datetime.now(IST_ZONE).date()
    today_summary = db.get(DailySummary, today)
    open_positions = (
        db.execute(select(ExecutionPosition).where(ExecutionPosition.status == "OPEN"))
        .scalars()
        .all()
    )
    return {
        "win_rate": float(today_summary.win_rate if today_summary is not None else 0.0),
        "total_pnl_today": float(today_summary.total_pnl if today_summary is not None else 0.0),
        "open_positions_count": len(open_positions),
        "open_positions_unrealized_pnl": round(
            sum(float(row.unrealized_pnl or 0.0) for row in open_positions),
            2,
        ),
        "total_trades_today": int(today_summary.total_trades if today_summary is not None else 0),
        "wins_today": int(today_summary.winning_trades if today_summary is not None else 0),
    }


def _notification_payload(settings: Settings | None = None) -> dict[str, Any]:
    cfg = settings or get_settings()
    return {
        "smtp_enabled": bool(cfg.smtp_enabled),
        "smtp_ready": smtp_ready(cfg),
        "from_email": cfg.smtp_from_email.strip() or None,
        "recipient_count": len(cfg.smtp_recipients),
    }


def _calendar_payload(*, option_selection: OptionSelection) -> dict[str, Any]:
    now = datetime.now(IST_ZONE)
    today = now.date()
    session_start, session_end = market_session_bounds(today)
    expiry_dates = {item.isoformat() for item in option_selection.available_expiries}
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


def build_chart_payload(
    db: Session,
    *,
    symbol: str,
    range_key: str = DEFAULT_CHART_RANGE,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    current = _ensure_ist(now) or datetime.now(IST_ZONE)
    instrument_key, display_symbol = resolve_instrument_key(db, symbol)
    range_name = str(range_key or DEFAULT_CHART_RANGE).strip().lower()
    if range_name not in CHART_RANGE_SPECS:
        range_name = DEFAULT_CHART_RANGE
    source_interval = str(CHART_RANGE_SPECS[range_name]["interval"])
    latest_source_ts = _latest_chart_source_ts(db, instrument_key=instrument_key, interval=source_interval)
    cache_key = (
        instrument_key,
        range_name,
        latest_source_ts.isoformat() if latest_source_ts is not None else None,
    )
    cached = _CHART_PAYLOAD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    plan = _chart_range_plan(range_name, current, latest_available_ts=latest_source_ts)
    rows = _chart_rows_from_range(
        db,
        instrument_key=instrument_key,
        interval=plan["interval"],
        start_ts=plan["start_ts"],
    )
    markers = _build_chart_markers(
        rows,
        interval=plan["interval"],
        settings=settings,
        range_key=plan["key"],
    )
    candles = [
        {
            "x": row["ts"].isoformat() if row["ts"] is not None else None,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"] or 0.0),
        }
        for row in rows
    ]
    payload = {
        "symbol": display_symbol,
        "instrument_key": instrument_key,
        "range": plan["key"],
        "label": plan["label"],
        "interval": plan["interval"],
        "source_interval": LIVE_INTERVAL,
        "is_resampled": plan["interval"] != LIVE_INTERVAL,
        "supports_live": bool(plan["supports_live"]),
        "start_date": plan["start_date"].isoformat(),
        "end_date": plan["end_date"].isoformat(),
        "generated_at": current.isoformat(),
        "candles": candles,
        "markers": markers,
        "available_ranges": _chart_range_options(),
    }
    stale_keys = [
        key
        for key in _CHART_PAYLOAD_CACHE
        if key[0] == instrument_key and key[1] == range_name and key != cache_key
    ]
    for key in stale_keys:
        _CHART_PAYLOAD_CACHE.pop(key, None)
    while len(_CHART_PAYLOAD_CACHE) >= 32:
        _CHART_PAYLOAD_CACHE.pop(next(iter(_CHART_PAYLOAD_CACHE)))
    _CHART_PAYLOAD_CACHE[cache_key] = payload
    return payload


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

    intervals = []
    for interval in ("1minute", "30minute", "day"):
        count, min_ts, max_ts = db.execute(
            select(
                func.count(RawCandle.id),
                func.min(RawCandle.ts),
                func.max(RawCandle.ts),
            ).where(
                and_(
                    RawCandle.instrument_key == instrument_key,
                    RawCandle.interval == interval,
                )
            )
        ).one()
        oldest = _ensure_ist(min_ts)
        latest = _ensure_ist(max_ts)
        covers_start = bool(oldest is not None and oldest.date() <= expected_start_date)
        covers_today = bool(latest is not None and latest.date() >= expected_end_date)
        intervals.append(
            {
                "interval": interval,
                "count": int(count or 0),
                "oldest_ts": oldest.isoformat() if oldest is not None else None,
                "latest_ts": latest.isoformat() if latest is not None else None,
                "covers_target_start": covers_start,
                "covers_today": covers_today,
                "expected_start_date": expected_start_date.isoformat(),
                "expected_end_date": expected_end_date.isoformat(),
                "window_ready": covers_start and covers_today,
                "coverage_days": (
                    max(0, (latest.date() - oldest.date()).days)
                    if oldest is not None and latest is not None
                    else 0
                ),
            }
        )

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
        "intervals": intervals,
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
) -> dict[str, Any]:
    settings = settings or get_settings()
    context = load_market_context(db, symbol=symbol, settings=settings)
    signal = build_technical_signal(db, context=context, settings=settings)
    option_selection = build_option_selection(db, context=context, signal=signal, settings=settings)

    open_positions = (
        db.execute(
            select(ExecutionPosition)
            .where(ExecutionPosition.status == "OPEN")
            .order_by(ExecutionPosition.opened_at.desc())
        )
        .scalars()
        .all()
    )
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
            .limit(15)
        )
        .scalars()
        .all()
    )
    recent_orders = (
        db.execute(
            select(ExecutionOrder)
            .where(symbol_value_filter(ExecutionOrder.symbol, context.symbol))
            .order_by(ExecutionOrder.created_at.desc())
            .limit(20)
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
        "stats": _stats_payload(db),
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
            "expiry_date": option_selection.expiry_date.isoformat(),
            "available_expiries": [item.isoformat() for item in option_selection.available_expiries],
            "strike_step": option_selection.strike_step,
            "chain_source": option_selection.chain_source,
            "chain_generated_at": (
                option_selection.chain_generated_at.isoformat()
                if option_selection.chain_generated_at is not None
                else None
            ),
            "signal": option_selection.signal,
        },
        "positions": [_serialize_position(row) for row in open_positions],
        "recent_trades": [_serialize_position(row) for row in recent_trades],
        "recent_orders": [_serialize_order(row) for row in recent_orders],
        "recent_signals": [],
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
