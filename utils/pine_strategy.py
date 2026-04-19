from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

import pandas as pd

from utils.constants import IST_ZONE
from utils.symbols import normalize_symbol_key


@dataclass(frozen=True, slots=True)
class PineStrategyProfile:
    name: str
    entry_start: time
    entry_end: time
    manage_end: time
    buy_rsi_min: float
    buy_rsi_max: float
    sell_rsi_min: float
    sell_rsi_max: float
    body_pct_min: float
    cooldown_bars: int
    minimum_setup_score: float
    allow_long: bool
    allow_short: bool


_MAJOR_INDEX_PROFILE = PineStrategyProfile(
    name="major_index_multi_strategy",
    entry_start=time(9, 20),
    entry_end=time(12, 30),
    manage_end=time(15, 15),
    buy_rsi_min=52.0,  # Relaxed: was 56
    buy_rsi_max=75.0,  # Relaxed: was 72
    sell_rsi_min=25.0,  # Relaxed: was 28
    sell_rsi_max=48.0,  # Relaxed: was 44
    body_pct_min=0.10,  # Relaxed: was 0.15
    cooldown_bars=6,
    minimum_setup_score=0.85,  # Relaxed: was 1.10
    allow_long=True,
    allow_short=True,
)

_INDIA_VIX_PROFILE = PineStrategyProfile(
    name="india_vix_multi_strategy_short_bias",
    entry_start=time(9, 20),
    entry_end=time(11, 45),
    manage_end=time(15, 15),
    buy_rsi_min=56.0,
    buy_rsi_max=72.0,
    sell_rsi_min=28.0,
    sell_rsi_max=48.0,
    body_pct_min=0.10,
    cooldown_bars=5,
    minimum_setup_score=0.90,
    allow_long=False,
    allow_short=True,
)


def strategy_profile_for_symbol(symbol: str | None) -> PineStrategyProfile:
    normalized = normalize_symbol_key(symbol or "")
    if normalized == "INDIAVIX" or "VIX" in normalized:
        return _INDIA_VIX_PROFILE
    return _MAJOR_INDEX_PROFILE


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _get_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    if hasattr(row, key):
        return getattr(row, key)
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(key)
    return None


def _truthy_at_or_above(value: float | None, threshold: float) -> bool:
    return value is None or value >= threshold


def _truthy_at_or_below(value: float | None, threshold: float) -> bool:
    return value is None or value <= threshold


def _default_flags() -> dict[str, Any]:
    return {
        "base_buy_setup": False,
        "base_sell_setup": False,
        "buy_setup": False,
        "sell_setup": False,
        "buy_score": 0.0,
        "sell_score": 0.0,
        "buy_count": 0,
        "sell_count": 0,
        "buy_alignment_count": 0,
        "sell_alignment_count": 0,
        "short_interval_bias": "NEUTRAL",
        "market_regime": "range",
        "buy_strategy_names": [],
        "sell_strategy_names": [],
        "dominant_action": "NEUTRAL",
    }


def local_time_in_ist(ts: datetime) -> time:
    aware = ts if ts.tzinfo is not None else ts.replace(tzinfo=IST_ZONE)
    return aware.astimezone(IST_ZONE).timetz().replace(tzinfo=None)


def in_entry_window(ts: datetime, *, symbol: str | None) -> bool:
    profile = strategy_profile_for_symbol(symbol)
    current = local_time_in_ist(ts)
    return profile.entry_start <= current <= profile.entry_end


def in_manage_window(ts: datetime, *, symbol: str | None) -> bool:
    profile = strategy_profile_for_symbol(symbol)
    current = local_time_in_ist(ts)
    return profile.entry_start <= current <= profile.manage_end


def _market_regime(
    *,
    trend_buy: bool,
    trend_sell: bool,
    breakout_buy: bool,
    breakout_sell: bool,
    squeeze_on: bool,
    pullback_buy: bool,
    pullback_sell: bool,
) -> str:
    if breakout_buy or breakout_sell:
        return "breakout"
    if squeeze_on:
        return "compression"
    if trend_buy or trend_sell:
        return "trend"
    if pullback_buy or pullback_sell:
        return "pullback"
    return "range"


def row_setup_flags(row: Any, *, symbol: str | None) -> dict[str, Any]:
    profile = strategy_profile_for_symbol(symbol)
    close = _to_float(_get_value(row, "close"))
    open_ = _to_float(_get_value(row, "open"))
    ema9 = _to_float(_get_value(row, "ema_9"))
    ema21 = _to_float(_get_value(row, "ema_21"))
    ema50 = _to_float(_get_value(row, "ema_50"))
    rsi = _to_float(_get_value(row, "rsi_14"))
    macd = _to_float(_get_value(row, "macd"))
    macd_signal = _to_float(_get_value(row, "macd_signal"))
    macd_hist = _to_float(_get_value(row, "macd_hist"))
    macd_hist_delta = _to_float(_get_value(row, "macd_hist_delta_1"))
    ema21_slope = _to_float(_get_value(row, "ema_21_slope_3"))
    body_pct_range = _to_float(_get_value(row, "body_pct_range"))
    if any(
        value is None
        for value in [
            close,
            open_,
            ema9,
            ema21,
            ema50,
            rsi,
            macd,
            macd_signal,
            macd_hist,
            macd_hist_delta,
            ema21_slope,
            body_pct_range,
        ]
    ):
        return _default_flags()

    vwap = _to_float(_get_value(row, "vwap"))
    breakout_high = _to_float(_get_value(row, "breakout_high_20"))
    breakout_low = _to_float(_get_value(row, "breakout_low_20"))
    bb_upper = _to_float(_get_value(row, "bb_upper"))
    bb_lower = _to_float(_get_value(row, "bb_lower"))
    kc_upper = _to_float(_get_value(row, "kc_upper"))
    kc_lower = _to_float(_get_value(row, "kc_lower"))
    lower_wick_pct = _to_float(_get_value(row, "lower_wick_pct"))
    upper_wick_pct = _to_float(_get_value(row, "upper_wick_pct"))

    buy_rsi_soft_min = max(48.0, profile.buy_rsi_min - 6.0)
    buy_rsi_soft_max = min(78.0, profile.buy_rsi_max + 4.0)
    sell_rsi_soft_min = max(22.0, profile.sell_rsi_min - 4.0)
    sell_rsi_soft_max = min(52.0, profile.sell_rsi_max + 6.0)
    body_floor = max(0.08, profile.body_pct_min * 0.55)
    breakout_body_floor = max(0.10, profile.body_pct_min * 0.80)
    squeeze_on = bool(
        bb_upper is not None
        and bb_lower is not None
        and kc_upper is not None
        and kc_lower is not None
        and bb_upper <= kc_upper
        and bb_lower >= kc_lower
    )

    trend_buy = bool(
        profile.allow_long
        and ema9 > ema21 > ema50
        and close > ema21
        and _truthy_at_or_above(vwap, ema21)
        and profile.buy_rsi_min <= rsi <= profile.buy_rsi_max
        and macd > macd_signal
        and macd_hist > 0.0
        and macd_hist_delta > 0.0
        and ema21_slope > 0.0
        and close > open_
        and body_pct_range >= profile.body_pct_min
    )
    trend_sell = bool(
        profile.allow_short
        and ema9 < ema21 < ema50
        and close < ema21
        and _truthy_at_or_below(vwap, ema21)
        and profile.sell_rsi_min <= rsi <= profile.sell_rsi_max
        and macd < macd_signal
        and macd_hist < 0.0
        and macd_hist_delta < 0.0
        and ema21_slope < 0.0
        and close < open_
        and body_pct_range >= profile.body_pct_min
    )

    pullback_buy = bool(
        profile.allow_long
        and ema21 > ema50
        and close > ema21
        and _truthy_at_or_above(vwap, ema21)
        and (open_ <= ema21 or (lower_wick_pct or 0.0) >= 0.35)
        and buy_rsi_soft_min <= rsi <= buy_rsi_soft_max
        and macd_hist_delta > 0.0
        and ema21_slope > 0.0
        and close > open_
        and body_pct_range >= body_floor
    )
    pullback_sell = bool(
        profile.allow_short
        and ema21 < ema50
        and close < ema21
        and _truthy_at_or_below(vwap, ema21)
        and (open_ >= ema21 or (upper_wick_pct or 0.0) >= 0.35)
        and sell_rsi_soft_min <= rsi <= sell_rsi_soft_max
        and macd_hist_delta < 0.0
        and ema21_slope < 0.0
        and close < open_
        and body_pct_range >= body_floor
    )

    breakout_buy = bool(
        profile.allow_long
        and breakout_high is not None
        and close > breakout_high
        and ema21 > ema50
        and _truthy_at_or_above(vwap, ema21)
        and rsi >= max(52.0, profile.buy_rsi_min - 2.0)
        and macd_hist > 0.0
        and ema21_slope > 0.0
        and close > open_
        and body_pct_range >= breakout_body_floor
    )
    breakout_sell = bool(
        profile.allow_short
        and breakout_low is not None
        and close < breakout_low
        and ema21 < ema50
        and _truthy_at_or_below(vwap, ema21)
        and rsi <= min(48.0, profile.sell_rsi_max + 2.0)
        and macd_hist < 0.0
        and ema21_slope < 0.0
        and close < open_
        and body_pct_range >= breakout_body_floor
    )

    squeeze_buy = bool(
        profile.allow_long
        and squeeze_on
        and bb_upper is not None
        and close > bb_upper
        and close > ema9
        and _truthy_at_or_above(vwap, ema21)
        and rsi >= max(52.0, profile.buy_rsi_min - 2.0)
        and macd_hist_delta > 0.0
        and ema21_slope > 0.0
    )
    squeeze_sell = bool(
        profile.allow_short
        and squeeze_on
        and bb_lower is not None
        and close < bb_lower
        and close < ema9
        and _truthy_at_or_below(vwap, ema21)
        and rsi <= min(48.0, profile.sell_rsi_max + 2.0)
        and macd_hist_delta < 0.0
        and ema21_slope < 0.0
    )

    buy_candidates = [
        ("trend_continuation", trend_buy, 1.00),
        ("ema_pullback_reclaim", pullback_buy, 0.85),
        ("range_breakout", breakout_buy, 0.95),
        ("squeeze_expansion", squeeze_buy, 0.90),
    ]
    sell_candidates = [
        ("trend_continuation", trend_sell, 1.00),
        ("ema_pullback_reject", pullback_sell, 0.85),
        ("range_breakdown", breakout_sell, 0.95),
        ("squeeze_expansion", squeeze_sell, 0.90),
    ]

    buy_strategy_names = [name for name, passed, _weight in buy_candidates if passed]
    sell_strategy_names = [name for name, passed, _weight in sell_candidates if passed]
    buy_score = round(sum(weight for _name, passed, weight in buy_candidates if passed), 2)
    sell_score = round(sum(weight for _name, passed, weight in sell_candidates if passed), 2)
    base_buy_setup = bool(buy_score >= profile.minimum_setup_score and buy_score > sell_score)
    base_sell_setup = bool(sell_score >= profile.minimum_setup_score and sell_score > buy_score)
    mtf_actions = [
        str(_get_value(row, "mtf_3m_action") or "NEUTRAL").upper(),
        str(_get_value(row, "mtf_5m_action") or "NEUTRAL").upper(),
    ]
    mtf_present = any(_get_value(row, key) is not None for key in ["mtf_3m_action", "mtf_5m_action"])
    buy_alignment_count = sum(1 for action in mtf_actions if action == "BUY")
    sell_alignment_count = sum(1 for action in mtf_actions if action == "SELL")
    strong_buy_without_alignment = buy_score >= max(profile.minimum_setup_score + 0.90, 1.90) and len(buy_strategy_names) >= 2
    strong_sell_without_alignment = sell_score >= max(profile.minimum_setup_score + 0.90, 1.90) and len(sell_strategy_names) >= 2
    if mtf_present:
        buy_setup = bool(base_buy_setup and (buy_alignment_count >= 1 or strong_buy_without_alignment))
        sell_setup = bool(base_sell_setup and (sell_alignment_count >= 1 or strong_sell_without_alignment))
    else:
        buy_setup = base_buy_setup
        sell_setup = base_sell_setup
    short_interval_bias = "BUY" if buy_alignment_count > sell_alignment_count else ("SELL" if sell_alignment_count > buy_alignment_count else "NEUTRAL")
    market_regime = _market_regime(
        trend_buy=trend_buy,
        trend_sell=trend_sell,
        breakout_buy=breakout_buy,
        breakout_sell=breakout_sell,
        squeeze_on=squeeze_on,
        pullback_buy=pullback_buy,
        pullback_sell=pullback_sell,
    )
    dominant_action = "BUY" if buy_setup else ("SELL" if sell_setup else "NEUTRAL")

    return {
        "base_buy_setup": base_buy_setup,
        "base_sell_setup": base_sell_setup,
        "buy_setup": buy_setup,
        "sell_setup": sell_setup,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "buy_count": len(buy_strategy_names),
        "sell_count": len(sell_strategy_names),
        "buy_alignment_count": buy_alignment_count,
        "sell_alignment_count": sell_alignment_count,
        "short_interval_bias": short_interval_bias,
        "market_regime": market_regime,
        "buy_strategy_names": buy_strategy_names,
        "sell_strategy_names": sell_strategy_names,
        "dominant_action": dominant_action,
    }


def local_pine_signal(row: Any, *, symbol: str | None) -> dict[str, Any]:
    setups = row_setup_flags(row, symbol=symbol)
    return {
        "action": setups["dominant_action"],
        "buy_score": float(setups["buy_score"]),
        "sell_score": float(setups["sell_score"]),
        "buy_alignment_count": int(setups.get("buy_alignment_count") or 0),
        "sell_alignment_count": int(setups.get("sell_alignment_count") or 0),
        "market_regime": str(setups.get("market_regime") or "range"),
        "buy_strategy_names": list(setups["buy_strategy_names"]),
        "sell_strategy_names": list(setups["sell_strategy_names"]),
    }


def resample_ohlcv(frame: pd.DataFrame, *, minutes: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    local = frame[["ts", "open", "high", "low", "close", "volume"]].copy()
    local["ts"] = pd.to_datetime(local["ts"])
    local = local.sort_values("ts").set_index("ts")
    out = (
        local.resample(f"{int(minutes)}min", label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return out


def build_interval_signal_frame(frame: pd.DataFrame, *, symbol: str | None, minutes: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    from feature_engine.price_features import build_price_features

    resampled = resample_ohlcv(frame, minutes=minutes)
    if resampled.empty:
        return pd.DataFrame()
    enriched = build_price_features(resampled.copy()).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for row in enriched.itertuples(index=False):
        flags = row_setup_flags(row, symbol=symbol)
        rows.append(
            {
                "ts": getattr(row, "ts"),
                f"mtf_{minutes}m_action": str(flags["dominant_action"]),
                f"mtf_{minutes}m_buy_score": float(flags["buy_score"]),
                f"mtf_{minutes}m_sell_score": float(flags["sell_score"]),
                f"mtf_{minutes}m_market_regime": str(flags.get("market_regime") or "range"),
            }
        )
    return pd.DataFrame(rows)


def attach_short_interval_states(
    frame: pd.DataFrame,
    *,
    symbol: str | None,
    minutes_list: tuple[int, ...] = (3, 5),
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy().sort_values("ts").reset_index(drop=True)
    out["ts"] = pd.to_datetime(out["ts"])
    for minutes in minutes_list:
        state_frame = build_interval_signal_frame(out, symbol=symbol, minutes=minutes)
        if state_frame.empty:
            out[f"mtf_{minutes}m_action"] = "NEUTRAL"
            out[f"mtf_{minutes}m_buy_score"] = 0.0
            out[f"mtf_{minutes}m_sell_score"] = 0.0
            out[f"mtf_{minutes}m_market_regime"] = "range"
            continue
        state_frame = state_frame.sort_values("ts").reset_index(drop=True)
        out = pd.merge_asof(out, state_frame, on="ts", direction="backward")
        out[f"mtf_{minutes}m_action"] = out[f"mtf_{minutes}m_action"].fillna("NEUTRAL")
        out[f"mtf_{minutes}m_buy_score"] = out[f"mtf_{minutes}m_buy_score"].fillna(0.0)
        out[f"mtf_{minutes}m_sell_score"] = out[f"mtf_{minutes}m_sell_score"].fillna(0.0)
        out[f"mtf_{minutes}m_market_regime"] = out[f"mtf_{minutes}m_market_regime"].fillna("range")
    return out
