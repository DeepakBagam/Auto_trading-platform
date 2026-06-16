"""Walk-forward signal backtest.

Runs the full signal engine on historical 1-minute candles (no lookahead),
simulates non-overlapping intraday trades with ATR-based stops, staged
trailing, partial exits, and momentum-based time exits, then prints a
detailed performance report.

Usage:
    python backend/scripts/backtest_signal.py
    python backend/scripts/backtest_signal.py --symbol "Nifty 50" --days 60
    python backend/scripts/backtest_signal.py --symbol "Nifty Bank" --days 30
"""
try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from backend.db.connection import SessionLocal
from backend.execution_engine.intraday_rules import (
    DEFAULT_EXECUTION_CONSTRAINTS,
    ExecutionConstraints,
    move_points,
    structured_stop_price,
    time_exit_reason,
)
from backend.feature_engine.price_features import build_price_features
from backend.utils.config import Settings, get_settings
from backend.utils.constants import IST_ZONE
from backend.utils.logger import setup_logging

setup_logging("backtest")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _build_confirmation_frame(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ts", "confirm_buy", "confirm_sell"])
    rs = (
        df.set_index("ts")
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
    if rs.empty:
        return pd.DataFrame(columns=["ts", "confirm_buy", "confirm_sell"])
    feat = build_price_features(rs)
    close = pd.to_numeric(feat.get("close"), errors="coerce")
    ema_21 = pd.to_numeric(feat.get("ema_21"), errors="coerce").fillna(close)
    ema_50 = pd.to_numeric(feat.get("ema_50"), errors="coerce").fillna(close)
    vwap = pd.to_numeric(feat.get("vwap"), errors="coerce").fillna(close)
    rsi = pd.to_numeric(feat.get("rsi_14"), errors="coerce").fillna(50.0)
    macd_hist = pd.to_numeric(feat.get("macd_hist"), errors="coerce").fillna(0.0)
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(feat["ts"]),
            "confirm_buy": (close > ema_21) & (ema_21 > ema_50) & (close >= vwap) & (rsi >= 54.0) & (macd_hist >= 0.0),
            "confirm_sell": (close < ema_21) & (ema_21 < ema_50) & (close <= vwap) & (rsi <= 46.0) & (macd_hist <= 0.0),
        }
    )


def _attach_multi_timeframe_confirmation(features: pd.DataFrame) -> pd.DataFrame:
    base = features.reset_index().copy()
    base["ts"] = pd.to_datetime(base["ts"])
    confirm_3 = _build_confirmation_frame(base[["ts", "open", "high", "low", "close", "volume"]], "3min")
    confirm_5 = _build_confirmation_frame(base[["ts", "open", "high", "low", "close", "volume"]], "5min")
    merged = pd.merge_asof(
        base.sort_values("ts"),
        confirm_3.sort_values("ts"),
        on="ts",
        direction="backward",
    ).rename(columns={"confirm_buy": "confirm_3m_buy", "confirm_sell": "confirm_3m_sell"})
    merged = pd.merge_asof(
        merged.sort_values("ts"),
        confirm_5.sort_values("ts"),
        on="ts",
        direction="backward",
    ).rename(columns={"confirm_buy": "confirm_5m_buy", "confirm_sell": "confirm_5m_sell"})
    for column in ("confirm_3m_buy", "confirm_3m_sell", "confirm_5m_buy", "confirm_5m_sell"):
        merged[column] = merged[column].fillna(False).astype(bool)
    return merged.set_index("ts")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_candles(symbol: str, days: int) -> pd.DataFrame:
    from sqlalchemy import and_, select
    from backend.db.models import RawCandle
    from backend.utils.symbols import instrument_key_filter

    db = SessionLocal()
    try:
        cutoff = datetime.now(IST_ZONE) - timedelta(days=days + 5)
        rows = (
            db.execute(
                select(RawCandle)
                .where(
                    and_(
                        instrument_key_filter(RawCandle.instrument_key, symbol),
                        RawCandle.interval == "1minute",
                        RawCandle.ts >= cutoff,
                    )
                )
                .order_by(RawCandle.ts.asc())
            )
            .scalars()
            .all()
        )
    finally:
        db.close()

    if not rows:
        raise ValueError(f"No 1-minute candles found for symbol '{symbol}'")

    df = pd.DataFrame(
        {
            "ts": [r.ts.astimezone(IST_ZONE) if r.ts.tzinfo else r.ts.replace(tzinfo=IST_ZONE) for r in rows],
            "open": [float(r.open) for r in rows],
            "high": [float(r.high) for r in rows],
            "low": [float(r.low) for r in rows],
            "close": [float(r.close) for r in rows],
            "volume": [float(r.volume or 0) for r in rows],
        }
    )
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    return df


def _load_vix(days: int) -> pd.DataFrame:
    from sqlalchemy import and_, select
    from backend.db.models import RawCandle
    from backend.utils.symbols import instrument_key_filter

    db = SessionLocal()
    try:
        cutoff = datetime.now(IST_ZONE) - timedelta(days=days + 5)
        rows = (
            db.execute(
                select(RawCandle)
                .where(
                    and_(
                        instrument_key_filter(RawCandle.instrument_key, "India VIX"),
                        RawCandle.interval == "1minute",
                        RawCandle.ts >= cutoff,
                    )
                )
                .order_by(RawCandle.ts.asc())
            )
            .scalars()
            .all()
        )
    finally:
        db.close()

    if not rows:
        return pd.DataFrame(columns=["vix"])
    df = pd.DataFrame(
        {
            "ts": [r.ts.astimezone(IST_ZONE) if r.ts.tzinfo else r.ts.replace(tzinfo=IST_ZONE) for r in rows],
            "vix": [float(r.close) for r in rows],
        }
    )
    df.set_index("ts", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Signal detection (mirrors live_service.build_technical_signal logic)
# ---------------------------------------------------------------------------

_REGIME_TRENDING = "TRENDING"
_REGIME_RANGE = "RANGE_BOUND"
_REGIME_VOLATILE = "HIGH_VOLATILITY"

EXECUTION_RULES = DEFAULT_EXECUTION_CONSTRAINTS
ENTRY_START = time(9, 45)
ENTRY_END = time(15, 0)
FORCE_SQUAREOFF_TIME = time(15, 0)
COOLDOWN_MINUTES = 12
COOLDOWN_AFTER_STOP_MINUTES = 20
MAX_SIGNALS_PER_DAY = 2
MIN_CANDLES = 60


def _parse_hhmm(value: str, fallback: time) -> time:
    try:
        hh, mm = str(value).strip().split(":", 1)
        return time(int(hh), int(mm))
    except Exception:
        return fallback


def _detect_regime(adx: float, plus_di: float, minus_di: float, atr: float, atr_mean: float) -> str:
    atr_ratio = atr / max(atr_mean, 1e-9)
    if atr_ratio > 1.6:
        return _REGIME_VOLATILE
    if adx >= 25.0 and abs(plus_di - minus_di) >= 10.0:
        return _REGIME_TRENDING
    if adx < 20.0:
        return _REGIME_RANGE
    return _REGIME_TRENDING


def _dynamic_threshold(regime: str, vix_ratio: float) -> float:
    base = 58.0 + (vix_ratio - 1.0) * 15.0
    if regime == _REGIME_TRENDING:
        base -= 6.0
    elif regime == _REGIME_RANGE:
        base += 10.0
    elif regime == _REGIME_VOLATILE:
        base += 8.0
    return max(48.0, min(80.0, base))


def _rsi_bands(regime: str) -> tuple[float, float, float, float]:
    if regime == _REGIME_TRENDING:
        return 50.0, 82.0, 18.0, 50.0
    if regime == _REGIME_RANGE:
        return 56.0, 72.0, 28.0, 44.0
    return 52.0, 78.0, 22.0, 48.0


def _exit_atr_multipliers(regime: str) -> tuple[float, float]:
    """Return (sl_atr, tp_atr)."""
    if regime == _REGIME_TRENDING:
        return 1.5, 2.5
    if regime == _REGIME_RANGE:
        return 0.8, 1.3
    return 1.2, 2.0


@dataclass
class _SignalResult:
    action: str          # BUY | SELL | HOLD
    score: float
    regime: str
    threshold: float
    atr: float
    adx: float
    ema_separation: float
    sl_offset: float
    tp_offset: float
    entry_close: float
    reason: str


def _resample_five_minute_signal_frame(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    base = features.reset_index().copy()
    base["ts"] = pd.to_datetime(base["ts"])
    bucket_start = []
    for value in base["ts"]:
        ts = _as_ist_datetime(value)
        session_start = ts.replace(hour=9, minute=15, second=0, microsecond=0)
        minutes_from_open = max(0, int((ts - session_start).total_seconds() // 60))
        bucket_index = minutes_from_open // 5
        bucket_start.append(session_start + timedelta(minutes=bucket_index * 5))
    base["bucket_start"] = bucket_start
    grouped = (
        base.groupby("bucket_start", as_index=False)
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


def detect_signal(
    features: pd.DataFrame,
    vix_candles: pd.DataFrame,
    now: datetime,
    *,
    last_signal_ts: datetime | None,
    signals_today: int,
    constraints: ExecutionConstraints = EXECUTION_RULES,
    settings: Settings | None = None,
) -> _SignalResult:
    """Run signal detection on the last row of `features`. Returns action and levels."""
    del vix_candles
    settings = settings or get_settings()
    if len(features) < MIN_CANDLES:
        return _SignalResult("HOLD", 0, _REGIME_RANGE, 63, 0, 0, 0, 0, 0, 0, "warmup")

    signal_frame = features
    if len(signal_frame) < 25:
        return _SignalResult("HOLD", 0, _REGIME_RANGE, 100, 0, 0, 0, 0, 0, 0, "warmup_1m")

    row = signal_frame.iloc[-1]
    prev = signal_frame.iloc[-2]
    close = _to_float(row.get("close")) or 0.0
    open_ = _to_float(row.get("open")) or close
    prev_high = _to_float(prev.get("high")) or close
    prev_low = _to_float(prev.get("low")) or close
    ema_9 = _to_float(row.get("ema_9")) or close
    ema_21 = _to_float(row.get("ema_21")) or close
    prev_ema_9 = _to_float(prev.get("ema_9")) or ema_9
    prev_ema_21 = _to_float(prev.get("ema_21")) or ema_21
    rsi = _to_float(row.get("rsi_14")) or 50.0
    atr = max(1e-9, _to_float(row.get("atr_14")) or 0.0)
    volume = _to_float(row.get("volume")) or 0.0
    volume_avg = _to_float(row.get("volume_sma_20")) or 0.0
    volume_ratio = _to_float(row.get("volume_ratio_20")) or 1.0

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

    # Balanced scoring system, aligned with live_service.build_technical_signal.
    score_buy = 0.0
    score_sell = 0.0

    # 1. EMA TREND ALIGNMENT (40 points)
    if ema_9 > ema_21:
        score_buy += 40.0
    if ema_9 < ema_21:
        score_sell += 40.0

    # 2. EMA CROSSOVER BONUS (20 points)
    if ema_cross_up:
        score_buy += 20.0
    if ema_cross_down:
        score_sell += 20.0

    # 3. RSI MOMENTUM (20 points)
    if rsi_buy_ok:
        score_buy += 20.0
    if rsi_sell_ok:
        score_sell += 20.0

    # 4. CANDLE DIRECTION (10 points)
    if bullish_candle:
        score_buy += 10.0
    if bearish_candle:
        score_sell += 10.0

    # 5. VOLUME CONFIRMATION (10 points)
    if volume_ok:
        score_buy += 10.0
        score_sell += 10.0

    # 6. BREAKOUT BONUS (10 points)
    if break_prev_high:
        score_buy += 10.0
    if break_prev_low:
        score_sell += 10.0

    min_score = float(getattr(settings, "signal_min_score", 63.0))
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
    raw_action = "BUY" if buy_ready else ("SELL" if sell_ready else "HOLD")
    raw_score = max(score_buy, score_sell)
    threshold = min_score

    now_time = now.time().replace(tzinfo=None)
    entry_start = _parse_hhmm(getattr(settings, "entry_window_start", ""), ENTRY_START)
    entry_end = _parse_hhmm(getattr(settings, "entry_window_end", ""), ENTRY_END)
    action = raw_action
    hold_reason = ""
    max_signals_per_day = max(1, int(getattr(settings, "signal_max_per_day", MAX_SIGNALS_PER_DAY)))
    if not (entry_start <= now_time <= entry_end):
        action, hold_reason = "HOLD", "outside_window"
    elif signals_today >= max_signals_per_day:
        action, hold_reason = "HOLD", "daily_cap"
    elif last_signal_ts is not None:
        elapsed = (now - last_signal_ts).total_seconds()
        if elapsed < COOLDOWN_MINUTES * 60:
            action, hold_reason = "HOLD", f"cooldown({int(COOLDOWN_MINUTES * 60 - elapsed)}s)"
    elif raw_action == "HOLD":
        action, hold_reason = "HOLD", "checklist_incomplete"

    sl_offset = max(atr, abs(close - prev_low), abs(prev_high - close))
    tp_offset = sl_offset * 1.5

    return _SignalResult(
        action=action,
        score=raw_score,
        regime=_REGIME_TRENDING if action in {"BUY", "SELL"} else _REGIME_RANGE,
        threshold=threshold,
        atr=atr,
        adx=0.0,
        ema_separation=abs(ema_9 - ema_21),
        sl_offset=sl_offset,
        tp_offset=tp_offset,
        entry_close=close,
        reason=hold_reason if action == "HOLD" else "ema_rsi_volume_breakout_5m",
    )


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    trade_id: int
    date: date
    entry_ts: datetime
    action: str          # BUY | SELL
    regime: str
    score: float
    entry_price: float
    sl: float
    tp: float
    partial_target: float
    stop_points: float
    exit_ts: datetime | None = None
    exit_price: float | None = None
    partial_exit_ts: datetime | None = None
    partial_exit_price: float | None = None
    exit_reason: str | None = None
    pnl_points: float | None = None
    holding_minutes: int = 0
    mfe_points: float = 0.0
    mae_points: float = 0.0
    is_win: bool | None = None


def _as_ist_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(IST_ZONE) if value.tzinfo else value.replace(tzinfo=IST_ZONE)
    out = pd.Timestamp(value).to_pydatetime(warn=False)
    return out.astimezone(IST_ZONE) if out.tzinfo else out.replace(tzinfo=IST_ZONE)


def _directional_price(action: str, entry_price: float, offset_points: float) -> float:
    return round(entry_price + (offset_points if action == "BUY" else -offset_points), 2)


def simulate_trades(
    df: pd.DataFrame,
    features_full: pd.DataFrame,
    vix_df: pd.DataFrame,
    constraints: ExecutionConstraints = EXECUTION_RULES,
    entry_dates: set[date] | None = None,
    settings: Settings | None = None,
) -> list[Trade]:
    """Walk forward through `df`, generate signals, and manage one trade at a time."""
    settings = settings or get_settings()
    trades: list[Trade] = []
    trade_id = 0
    signals_today: dict[date, int] = {}
    last_signal_ts: datetime | None = None
    next_eligible_ts: datetime | None = None

    i = MIN_CANDLES + 1
    while i < len(features_full) - 1:
        row_ts = _as_ist_datetime(features_full.index[i])
        if next_eligible_ts is not None and row_ts < next_eligible_ts:
            i += 1
            continue
        today = row_ts.date()
        if entry_dates is not None and today not in entry_dates:
            i += 1
            continue
        features_window = features_full.iloc[max(0, i - 250): i]

        sig = detect_signal(
            features_window,
            vix_df,
            row_ts,
            last_signal_ts=last_signal_ts,
            signals_today=signals_today.get(today, 0),
            constraints=constraints,
            settings=settings,
        )

        if sig.action not in ("BUY", "SELL"):
            i += 1
            continue

        # Entry at next bar
        if i + 1 >= len(df):
            break
        entry_bar = df.iloc[i + 1]
        entry_ts = _as_ist_datetime(entry_bar.name)
        entry_price = float(entry_bar["open"])
        stop_points = float(sig.sl_offset)
        runner_points = float(sig.tp_offset)
        partial_target = float(constraints.partial_exit_points)
        partial_fraction = max(0.0, min(1.0, float(constraints.partial_exit_fraction)))

        sl = _directional_price(sig.action, entry_price, -stop_points)
        tp = _directional_price(sig.action, entry_price, runner_points)
        partial_price = _directional_price(sig.action, entry_price, partial_target)

        current_sl = float(sl)
        best_price = float(entry_price)
        realized_points = 0.0
        remaining_fraction = 1.0
        partial_taken = False
        partial_exit_ts: datetime | None = None
        partial_exit_price: float | None = None
        mfe_points = 0.0
        mae_points = 0.0
        exit_ts: datetime | None = None
        exit_price: float | None = None
        exit_reason: str | None = None
        exit_idx = i + 1

        for j in range(i + 2, len(df)):
            future_ts_dt = _as_ist_datetime(df.index[j])
            if future_ts_dt.date() != entry_ts.date():
                prev_bar = df.iloc[j - 1]
                exit_ts = _as_ist_datetime(prev_bar.name)
                exit_price = float(prev_bar["close"])
                exit_reason = "EOD_CLOSE"
                exit_idx = j - 1
                break

            bar = df.iloc[j]
            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])
            old_sl = current_sl

            if sig.action == "BUY":
                mfe_points = max(mfe_points, bar_high - entry_price)
                mae_points = max(mae_points, entry_price - bar_low)
                if bar_low <= old_sl:
                    exit_ts = future_ts_dt
                    exit_price = old_sl
                    exit_reason = "SL_HIT" if old_sl <= sl else ("BREAKEVEN_HIT" if old_sl <= entry_price else "TSL_HIT")
                    exit_idx = j
                    break
                if (not partial_taken) and partial_fraction > 0.0 and bar_high >= partial_price:
                    partial_taken = True
                    partial_exit_ts = future_ts_dt
                    partial_exit_price = partial_price
                    realized_points += partial_fraction * partial_target
                    remaining_fraction = round(1.0 - partial_fraction, 4)
                if bar_high >= tp:
                    exit_ts = future_ts_dt
                    exit_price = tp
                    exit_reason = "RUNNER_TARGET_HIT"
                    exit_idx = j
                    break
                best_price = max(best_price, bar_high)
                current_sl = structured_stop_price(
                    action=sig.action,
                    entry_price=entry_price,
                    initial_stop_price=sl,
                    current_stop_price=current_sl,
                    best_price=best_price,
                    constraints=constraints,
                )
            else:
                mfe_points = max(mfe_points, entry_price - bar_low)
                mae_points = max(mae_points, bar_high - entry_price)
                if bar_high >= old_sl:
                    exit_ts = future_ts_dt
                    exit_price = old_sl
                    exit_reason = "SL_HIT" if old_sl >= sl else ("BREAKEVEN_HIT" if old_sl >= entry_price else "TSL_HIT")
                    exit_idx = j
                    break
                if (not partial_taken) and partial_fraction > 0.0 and bar_low <= partial_price:
                    partial_taken = True
                    partial_exit_ts = future_ts_dt
                    partial_exit_price = partial_price
                    realized_points += partial_fraction * partial_target
                    remaining_fraction = round(1.0 - partial_fraction, 4)
                if bar_low <= tp:
                    exit_ts = future_ts_dt
                    exit_price = tp
                    exit_reason = "RUNNER_TARGET_HIT"
                    exit_idx = j
                    break
                best_price = min(best_price, bar_low)
                current_sl = structured_stop_price(
                    action=sig.action,
                    entry_price=entry_price,
                    initial_stop_price=sl,
                    current_stop_price=current_sl,
                    best_price=best_price,
                    constraints=constraints,
                )

            elapsed_minutes = max(1, int((future_ts_dt - entry_ts).total_seconds() // 60))
            current_points = move_points(sig.action, entry_price, bar_close)
            timed_reason = time_exit_reason(
                elapsed_minutes=elapsed_minutes,
                mfe_points=mfe_points,
                current_points=current_points,
                partial_taken=partial_taken,
                stop_points=stop_points,
                constraints=constraints,
            )
            if timed_reason is not None:
                exit_ts = future_ts_dt
                exit_price = bar_close
                exit_reason = timed_reason
                exit_idx = j
                break

            if future_ts_dt.time() >= FORCE_SQUAREOFF_TIME:
                exit_ts = future_ts_dt
                exit_price = bar_open
                exit_reason = "FORCE_SQUAREOFF"
                exit_idx = j
                break

        if exit_price is None:
            fallback_bar = df.iloc[min(i + 2, len(df) - 1)]
            exit_ts = _as_ist_datetime(fallback_bar.name)
            exit_price = float(fallback_bar["close"])
            exit_reason = "NO_EXIT"
            exit_idx = min(i + 2, len(df) - 1)

        realized_points += remaining_fraction * move_points(sig.action, entry_price, exit_price)
        pnl = round(realized_points, 2)
        is_win = pnl > 0
        holding_minutes = max(1, int(((exit_ts or entry_ts) - entry_ts).total_seconds() // 60))

        trade_id += 1
        trades.append(Trade(
            trade_id=trade_id,
            date=today,
            entry_ts=entry_ts,
            action=sig.action,
            regime=sig.regime,
            score=sig.score,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            partial_target=partial_price,
            stop_points=stop_points,
            exit_ts=exit_ts,
            exit_price=exit_price,
            partial_exit_ts=partial_exit_ts,
            partial_exit_price=partial_exit_price,
            exit_reason=exit_reason,
            pnl_points=pnl,
            holding_minutes=holding_minutes,
            mfe_points=round(mfe_points, 2),
            mae_points=round(mae_points, 2),
            is_win=is_win,
        ))
        last_signal_ts = entry_ts
        signals_today[today] = signals_today.get(today, 0) + 1
        cooldown_minutes = COOLDOWN_AFTER_STOP_MINUTES if pnl <= 0 and exit_reason in {"SL_HIT", "BREAKEVEN_HIT"} else COOLDOWN_MINUTES
        next_eligible_ts = (exit_ts or entry_ts) + timedelta(minutes=cooldown_minutes)
        i = max(i + 1, exit_idx + 1)

    return trades


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_stats(trades: list[Trade]) -> dict[str, Any]:
    if not trades:
        return {}
    n = len(trades)
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    partials = [t for t in trades if t.partial_exit_ts is not None]
    win_rate = len(wins) / n * 100

    pnls = [t.pnl_points for t in trades]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / n
    avg_win = sum(t.pnl_points for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_points for t in losses) / len(losses) if losses else 0
    profit_factor = abs(sum(t.pnl_points for t in wins)) / max(1e-9, abs(sum(t.pnl_points for t in losses)))
    avg_holding = sum(t.holding_minutes for t in trades) / n
    avg_mfe = sum(t.mfe_points for t in trades) / n
    avg_mae = sum(t.mae_points for t in trades) / n

    # Drawdown (cumulative)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)

    # By regime
    regime_stats: dict[str, dict] = {}
    for t in trades:
        rs = regime_stats.setdefault(t.regime, {"n": 0, "wins": 0, "pnl": 0.0})
        rs["n"] += 1
        rs["wins"] += int(t.is_win)
        rs["pnl"] += t.pnl_points

    # By exit reason
    exits: dict[str, int] = {}
    for t in trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

    return {
        "total_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate, 1),
        "total_pnl_points": round(total_pnl, 2),
        "avg_pnl_per_trade_points": round(avg_pnl, 2),
        "avg_win_points": round(avg_win, 2),
        "avg_loss_points": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_points": round(max_dd, 2),
        "avg_holding_minutes": round(avg_holding, 1),
        "avg_mfe_points": round(avg_mfe, 2),
        "avg_mae_points": round(avg_mae, 2),
        "partial_exit_rate_pct": round(len(partials) / n * 100, 1),
        "by_regime": {k: {"n": v["n"], "win_pct": round(v["wins"] / v["n"] * 100, 1), "pnl": round(v["pnl"], 2)} for k, v in regime_stats.items()},
        "by_exit_reason": exits,
    }


def _print_report(symbol: str, days: int, trades: list[Trade], stats: dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  SIGNAL BACKTEST REPORT")
    print(f"  Symbol : {symbol}")
    print(f"  Period : last {days} calendar days")
    print("  Exec   : 1m EMA 9/21 + RSI + volume/breakout gates | configured entry window | max signals/day")
    print(sep)

    if not stats:
        print("  No trades generated.")
        return

    print(f"  Total Trades   : {stats['total_trades']}")
    print(f"  Wins           : {stats['wins']}  ({stats['win_rate_pct']}%)")
    print(f"  Losses         : {stats['losses']}")
    print(f"  Win Rate       : {stats['win_rate_pct']}%")
    print(f"  Profit Factor  : {stats['profit_factor']}x  (>1.0 = profitable)")
    print()
    print(f"  Total P&L (pts): {stats['total_pnl_points']:+.1f}")
    print(f"  Avg per Trade  : {stats['avg_pnl_per_trade_points']:+.1f} pts")
    print(f"  Avg Win        : +{stats['avg_win_points']:.1f} pts")
    print(f"  Avg Loss       : {stats['avg_loss_points']:.1f} pts")
    print(f"  Max Drawdown   : {stats['max_drawdown_points']:.1f} pts")
    print(f"  Avg Hold       : {stats['avg_holding_minutes']:.1f} min")
    print(f"  Avg MFE / MAE  : {stats['avg_mfe_points']:.1f} / {stats['avg_mae_points']:.1f} pts")
    print(f"  Partial Exits  : {stats['partial_exit_rate_pct']:.1f}%")
    print()
    print("  By Market Regime:")
    for regime, rs in stats["by_regime"].items():
        print(f"    {regime:<18} trades={rs['n']:3d}  win={rs['win_pct']:5.1f}%  pnl={rs['pnl']:+.1f} pts")
    print()
    print("  By Exit Reason:")
    for reason, cnt in sorted(stats["by_exit_reason"].items(), key=lambda x: -x[1]):
        print(f"    {reason:<20} {cnt:4d} trades")
    print(sep)

    # Accuracy note
    wr = stats["win_rate_pct"]
    pf = stats["profit_factor"]
    if wr >= 50 and pf >= 2.0:
        verdict = "GOOD - Execution quality is approaching production thresholds."
    elif wr >= 48 and pf >= 1.6:
        verdict = "ACCEPTABLE - Positive edge, but still needs risk tuning."
    else:
        verdict = "NEEDS WORK - Risk control or exit efficiency is still weak."
    print(f"\n  Verdict: {verdict}")
    print(sep)


# ---------------------------------------------------------------------------
# Trade log writer
# ---------------------------------------------------------------------------

def _save_logs(trades: list[Trade], stats: dict, symbol: str, days: int, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    sym_clean = symbol.replace(" ", "_").replace("/", "_")

    # CSV
    csv_path = out / f"backtest_{sym_clean}_{ts_str}.csv"
    rows = []
    for t in trades:
        rows.append({
            "trade_id": t.trade_id,
            "date": t.date.isoformat(),
            "entry_ts": t.entry_ts.isoformat() if t.entry_ts else "",
            "action": t.action,
            "regime": t.regime,
            "score": round(t.score, 1),
            "entry_price": t.entry_price,
            "sl": t.sl,
            "tp": t.tp,
            "partial_target": t.partial_target,
            "stop_points": t.stop_points,
            "partial_exit_ts": t.partial_exit_ts.isoformat() if t.partial_exit_ts else "",
            "partial_exit_price": t.partial_exit_price,
            "exit_ts": t.exit_ts.isoformat() if t.exit_ts else "",
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "pnl_points": t.pnl_points,
            "holding_minutes": t.holding_minutes,
            "mfe_points": t.mfe_points,
            "mae_points": t.mae_points,
            "is_win": t.is_win,
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Trade log  : {csv_path}")

    # JSON summary
    json_path = out / f"backtest_{sym_clean}_{ts_str}_summary.json"
    json_path.write_text(json.dumps({"symbol": symbol, "days": days, "stats": stats, "trades": rows}, indent=2))
    print(f"  JSON report: {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward signal backtest")
    parser.add_argument("--symbol", default="Nifty 50", help="Underlying symbol")
    parser.add_argument("--days", type=int, default=30, help="Calendar days to backtest")
    parser.add_argument("--output-dir", default="logs/backtests", help="Output directory")
    parser.add_argument("--entry-start", default=None, help="Override strategy entry start, HH:MM")
    parser.add_argument("--entry-end", default=None, help="Override strategy entry end, HH:MM")
    parser.add_argument("--max-signals-per-day", type=int, default=None, help="Override max signals per day")
    args = parser.parse_args()

    settings = get_settings()
    overrides: dict[str, Any] = {}
    if args.entry_start:
        overrides["entry_window_start"] = args.entry_start
    if args.entry_end:
        overrides["entry_window_end"] = args.entry_end
    if args.max_signals_per_day is not None:
        overrides["signal_max_per_day"] = max(1, int(args.max_signals_per_day))
    if overrides:
        settings = settings.model_copy(update=overrides)

    print(f"\nLoading candles for '{args.symbol}' (last {args.days} calendar days plus warmup)...")
    df_raw = _load_candles(args.symbol, args.days)
    print(f"Loaded {len(df_raw):,} bars  ({df_raw.index[0].date()} -> {df_raw.index[-1].date()})")
    sessions = sorted({ts.date() for ts in df_raw.index})
    entry_dates = set(sessions[-max(1, int(args.days)):])
    entry_window_start = min(entry_dates)
    entry_window_end = max(entry_dates)
    print(
        "Testing entries on latest "
        f"{len(entry_dates)} trading sessions ({entry_window_start} -> {entry_window_end}); "
        "earlier loaded bars are warmup only."
    )

    vix_df = _load_vix(args.days)

    print("Building features...")
    df_reset = df_raw.reset_index().rename(columns={"ts": "ts"})
    df_reset["ts"] = pd.to_datetime(df_reset["ts"])
    features_full = build_price_features(df_reset)
    features_full["ts"] = pd.to_datetime(features_full["ts"])
    features_full.set_index("ts", inplace=True)

    print("Running walk-forward simulation...")
    trades = simulate_trades(df_raw, features_full, vix_df, entry_dates=entry_dates, settings=settings)
    print(f"Simulation complete - {len(trades)} trades generated.")

    stats = _compute_stats(trades)
    _print_report(args.symbol, args.days, trades, stats)
    _save_logs(trades, stats, args.symbol, args.days, args.output_dir)


if __name__ == "__main__":
    main()
