from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Literal

import pandas as pd

from backend.feature_engine.price_features import build_price_features

Action = Literal["BUY", "SELL", "HOLD"]
SetupName = Literal["VWAP_REVERSION", "MICRO_MOMENTUM", "MICRO_PULLBACK", "NONE"]


@dataclass(frozen=True, slots=True)
class ScalpingConfig:
    """Independent short-duration index scalping rules.

    This is intentionally separate from the directional trend system. It uses
    small fixed index-point exits, a per-day cap, and no position carry.
    """

    stop_points: float = 14.0
    target_points: float = 20.0
    max_hold_minutes: int = 10
    cooldown_minutes: int = 7
    max_trades_per_day: int = 8
    entry_start: time = time(9, 25)
    morning_end: time = time(11, 35)
    afternoon_start: time = time(13, 40)
    entry_end: time = time(14, 45)
    warmup_bars: int = 50
    min_atr_points: float = 5.0
    max_atr_points: float = 34.0
    reversion_adx_max: float = 24.0
    momentum_adx_min: float = 16.0
    momentum_adx_max: float = 34.0
    bb_std: float = 2.0
    reversion_rsi_long: float = 34.0
    reversion_rsi_short: float = 66.0
    momentum_rsi_long: float = 57.0
    momentum_rsi_short: float = 43.0
    max_vwap_extension_atr: float = 2.2
    ideal_vwap_extension_atr: float = 1.6
    min_body_pct_momentum: float = 0.52
    max_body_pct_momentum: float = 0.80
    max_momentum_wick_pct: float = 0.30
    max_breakout_range_atr: float = 1.45
    max_three_bar_move_atr: float = 2.0
    min_momentum_score: float = 88.0
    min_pullback_score: float = 88.0
    min_reversion_score: float = 88.0
    chop_skip_score: int = 2
    min_rr: float = 1.2


@dataclass(frozen=True, slots=True)
class ScalpSignal:
    action: Action
    setup: SetupName
    score: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    reasons: tuple[str, ...]
    details: dict[str, float | int | str | bool | None]


def _to_float(value, fallback: float = 0.0) -> float:
    try:
        out = float(value)
        return out if out == out else fallback
    except (TypeError, ValueError):
        return fallback


def _in_entry_window(ts: datetime, config: ScalpingConfig) -> bool:
    now_time = ts.time().replace(tzinfo=None)
    morning = config.entry_start <= now_time <= config.morning_end
    afternoon = config.afternoon_start <= now_time <= config.entry_end
    return morning or afternoon


def _session_vwap_proxy(frame: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP; falls back to equal-weight typical price for index feeds."""
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    has_volume = volume > 0
    weights = volume.where(has_volume, 1.0)
    session_key = pd.to_datetime(frame["ts"]).dt.date
    numerator = (typical * weights).groupby(session_key).cumsum()
    denominator = weights.groupby(session_key).cumsum().replace(0.0, pd.NA)
    return (numerator / denominator).ffill().bfill().fillna(frame["close"])


def build_scalping_features(candles: pd.DataFrame, config: ScalpingConfig | None = None) -> pd.DataFrame:
    cfg = config or ScalpingConfig()
    out = build_price_features(candles.copy())
    out["session_vwap"] = _session_vwap_proxy(out)
    out["rsi_7"] = _rsi_short(out["close"], 7)
    out["ema_9_slope_2"] = out["ema_9"] - out["ema_9"].shift(2)
    out["ema_sep_atr"] = (out["ema_9"] - out["ema_21"]).abs() / out["atr_14"].replace(0.0, pd.NA)
    out["micro_high_10"] = out["high"].rolling(10).max().shift(1)
    out["micro_low_10"] = out["low"].rolling(10).min().shift(1)
    out["bb_width"] = out["bb_upper"] - out["bb_lower"]
    out["bb_width_mean_20"] = out["bb_width"].rolling(20, min_periods=10).mean()
    out["vwap_distance_atr"] = (out["close"] - out["session_vwap"]).abs() / (
        out["atr_14"].replace(0.0, pd.NA)
    )
    out["atr_ratio_50"] = out["atr_14"] / out["atr_14"].rolling(50, min_periods=20).mean()
    out["candle_range_atr"] = (out["high"] - out["low"]) / out["atr_14"].replace(0.0, pd.NA)
    out["three_bar_move_atr"] = (out["close"] - out["close"].shift(3)).abs() / out[
        "atr_14"
    ].replace(0.0, pd.NA)
    out["close_position"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0.0, 1e-9)
    vwap_side = (out["close"] > out["session_vwap"]).astype(int)
    out["vwap_crosses_10"] = vwap_side.diff().abs().rolling(10, min_periods=5).sum()
    rolling_high_6 = out["high"].rolling(6, min_periods=6).max()
    rolling_low_6 = out["low"].rolling(6, min_periods=6).min()
    out["six_bar_range_atr"] = (rolling_high_6 - rolling_low_6) / out["atr_14"].replace(0.0, pd.NA)
    out["valid_volume"] = bool((pd.to_numeric(out["volume"], errors="coerce").fillna(0.0) > 0).any())
    out["scalp_rr"] = cfg.target_points / max(cfg.stop_points, 1e-9)
    return out


def _rsi_short(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def detect_scalp_signal(
    features: pd.DataFrame,
    *,
    now: datetime,
    trades_today: int,
    last_exit_ts: datetime | None,
    config: ScalpingConfig | None = None,
) -> ScalpSignal:
    cfg = config or ScalpingConfig()
    if len(features) < cfg.warmup_bars:
        return _hold("warmup", features, cfg)
    if cfg.target_points / max(cfg.stop_points, 1e-9) < cfg.min_rr:
        return _hold("rr_below_minimum", features, cfg)
    if trades_today >= cfg.max_trades_per_day:
        return _hold("daily_cap", features, cfg)
    if not _in_entry_window(now, cfg):
        return _hold("outside_scalp_window", features, cfg)
    if last_exit_ts is not None and now < last_exit_ts + timedelta(minutes=cfg.cooldown_minutes):
        return _hold("cooldown", features, cfg)

    row = features.iloc[-1]
    close = _to_float(row.get("close"))
    open_ = _to_float(row.get("open"), close)
    high = _to_float(row.get("high"), close)
    low = _to_float(row.get("low"), close)
    atr = _to_float(row.get("atr_14"))
    adx = _to_float(row.get("adx_14"), 20.0)
    rsi_7 = _to_float(row.get("rsi_7"), 50.0)
    ema_9 = _to_float(row.get("ema_9"), close)
    ema_21 = _to_float(row.get("ema_21"), close)
    ema_9_slope = _to_float(row.get("ema_9_slope_2"))
    vwap = _to_float(row.get("session_vwap"), close)
    micro_high = _to_float(row.get("micro_high_10"), high)
    micro_low = _to_float(row.get("micro_low_10"), low)
    bb_width = _to_float(row.get("bb_width"))
    bb_width_mean = _to_float(row.get("bb_width_mean_20"), bb_width)
    body_pct = _to_float(row.get("body_pct_range"))
    lower_wick = _to_float(row.get("lower_wick_pct"))
    upper_wick = _to_float(row.get("upper_wick_pct"))
    vwap_distance_atr = _to_float(row.get("vwap_distance_atr"), 0.0)
    atr_ratio = _to_float(row.get("atr_ratio_50"), 1.0)
    candle_range_atr = _to_float(row.get("candle_range_atr"), 1.0)
    three_bar_move_atr = _to_float(row.get("three_bar_move_atr"), 0.0)
    close_position = _to_float(row.get("close_position"), 0.5)
    volume_ratio = _to_float(row.get("volume_ratio_20"), 1.0)
    valid_volume = bool(row.get("valid_volume"))

    if atr < cfg.min_atr_points:
        return _hold("atr_too_low", features, cfg)
    if atr > cfg.max_atr_points:
        return _hold("atr_too_high", features, cfg)
    if atr_ratio > 1.8:
        return _hold("volatility_shock", features, cfg)
    if atr_ratio < 0.65:
        return _hold("dead_volatility", features, cfg)
    if vwap_distance_atr > cfg.max_vwap_extension_atr:
        return _hold("vwap_extension_too_large", features, cfg)

    squeeze_releasing = bb_width_mean > 0 and bb_width >= bb_width_mean * 0.92
    chop_score = _chop_score(row, bb_width, bb_width_mean)
    if chop_score >= cfg.chop_skip_score:
        return _hold("chop_noise", features, cfg)

    momentum_long_score = _score_micro_momentum(
        action="BUY",
        close=close,
        open_=open_,
        micro_level=micro_high,
        vwap=vwap,
        ema_9=ema_9,
        ema_21=ema_21,
        ema_9_slope=ema_9_slope,
        rsi_7=rsi_7,
        adx=adx,
        body_pct=body_pct,
        adverse_wick=upper_wick,
        candle_range_atr=candle_range_atr,
        three_bar_move_atr=three_bar_move_atr,
        close_position=close_position,
        vwap_distance_atr=vwap_distance_atr,
        volume_ratio=volume_ratio,
        valid_volume=valid_volume,
        squeeze_releasing=squeeze_releasing,
        chop_score=chop_score,
        cfg=cfg,
    )
    momentum_short_score = _score_micro_momentum(
        action="SELL",
        close=close,
        open_=open_,
        micro_level=micro_low,
        vwap=vwap,
        ema_9=ema_9,
        ema_21=ema_21,
        ema_9_slope=ema_9_slope,
        rsi_7=rsi_7,
        adx=adx,
        body_pct=body_pct,
        adverse_wick=lower_wick,
        candle_range_atr=candle_range_atr,
        three_bar_move_atr=three_bar_move_atr,
        close_position=close_position,
        vwap_distance_atr=vwap_distance_atr,
        volume_ratio=volume_ratio,
        valid_volume=valid_volume,
        squeeze_releasing=squeeze_releasing,
        chop_score=chop_score,
        cfg=cfg,
    )
    pullback_long_score = _score_pullback("BUY", features, cfg)
    pullback_short_score = _score_pullback("SELL", features, cfg)
    reversion_long_score = _score_vwap_reversion("BUY", features, cfg)
    reversion_short_score = _score_vwap_reversion("SELL", features, cfg)

    candidates: list[tuple[float, Action, SetupName, tuple[str, ...]]] = [
        (momentum_long_score, "BUY", "MICRO_MOMENTUM", ("A-grade 10-bar breakout", "VWAP/EMA aligned")),
        (momentum_short_score, "SELL", "MICRO_MOMENTUM", ("A-grade 10-bar breakdown", "VWAP/EMA aligned")),
        (pullback_long_score, "BUY", "MICRO_PULLBACK", ("Breakout pullback continuation", "No chase entry")),
        (pullback_short_score, "SELL", "MICRO_PULLBACK", ("Breakdown pullback continuation", "No chase entry")),
        (reversion_long_score, "BUY", "VWAP_REVERSION", ("BB/VWAP range rejection", "RSI divergence/range gate")),
        (reversion_short_score, "SELL", "VWAP_REVERSION", ("BB/VWAP range rejection", "RSI divergence/range gate")),
    ]
    score, action, setup, reasons = max(candidates, key=lambda item: item[0])
    threshold = (
        cfg.min_reversion_score
        if setup == "VWAP_REVERSION"
        else cfg.min_pullback_score
        if setup == "MICRO_PULLBACK"
        else cfg.min_momentum_score
    )
    if score >= threshold:
        return _signal(action, setup, close, cfg, score, reasons, row)

    return _hold("no_scalp_setup", features, cfg)


def _chop_score(row: pd.Series, bb_width: float, bb_width_mean: float) -> int:
    atr = _to_float(row.get("atr_14"))
    adx = _to_float(row.get("adx_14"), 20.0)
    ema_sep_atr = _to_float(row.get("ema_sep_atr"), 1.0)
    vwap_crosses = _to_float(row.get("vwap_crosses_10"), 0.0)
    six_bar_range_atr = _to_float(row.get("six_bar_range_atr"), 2.0)
    score = 0
    if adx < 16.0:
        score += 1
    if ema_sep_atr < 0.25:
        score += 1
    if vwap_crosses >= 3.0:
        score += 1
    if bb_width_mean > 0 and bb_width < bb_width_mean * 0.75:
        score += 1
    if atr > 0 and six_bar_range_atr < 1.0:
        score += 1
    return score


def _score_micro_momentum(
    *,
    action: Action,
    close: float,
    open_: float,
    micro_level: float,
    vwap: float,
    ema_9: float,
    ema_21: float,
    ema_9_slope: float,
    rsi_7: float,
    adx: float,
    body_pct: float,
    adverse_wick: float,
    candle_range_atr: float,
    three_bar_move_atr: float,
    close_position: float,
    vwap_distance_atr: float,
    volume_ratio: float,
    valid_volume: bool,
    squeeze_releasing: bool,
    chop_score: int,
    cfg: ScalpingConfig,
) -> float:
    long_side = action == "BUY"
    breaks_level = close > micro_level if long_side else close < micro_level
    candle_direction = close > open_ if long_side else close < open_
    vwap_aligned = close > vwap if long_side else close < vwap
    ema_aligned = ema_9 > ema_21 and ema_9_slope > 0 if long_side else ema_9 < ema_21 and ema_9_slope < 0
    rsi_ok = cfg.momentum_rsi_long <= rsi_7 <= 68.0 if long_side else 32.0 <= rsi_7 <= cfg.momentum_rsi_short
    close_location_ok = close_position >= 0.58 if long_side else close_position <= 0.42
    score = 0.0
    if breaks_level and candle_direction:
        score += 15
    if vwap_aligned:
        score += 10
    if ema_aligned:
        score += 10
    if rsi_ok:
        score += 10
    if cfg.momentum_adx_min <= adx <= cfg.momentum_adx_max:
        score += 10
    if cfg.min_body_pct_momentum <= body_pct <= cfg.max_body_pct_momentum:
        score += 10
    if adverse_wick <= cfg.max_momentum_wick_pct and close_location_ok:
        score += 8
    if 0.65 <= candle_range_atr <= cfg.max_breakout_range_atr:
        score += 8
    if vwap_distance_atr <= cfg.ideal_vwap_extension_atr:
        score += 8
    if three_bar_move_atr <= cfg.max_three_bar_move_atr and squeeze_releasing:
        score += 6
    if (valid_volume and volume_ratio >= 1.15) or ((not valid_volume) and candle_range_atr >= 0.75):
        score += 5
    if chop_score == 0:
        score += 6
    return score


def _score_pullback(action: Action, features: pd.DataFrame, cfg: ScalpingConfig) -> float:
    if len(features) < 6:
        return 0.0
    recent = features.iloc[-4:]
    row = features.iloc[-1]
    close = _to_float(row.get("close"))
    open_ = _to_float(row.get("open"), close)
    ema_9 = _to_float(row.get("ema_9"), close)
    ema_21 = _to_float(row.get("ema_21"), close)
    vwap = _to_float(row.get("session_vwap"), close)
    rsi_7 = _to_float(row.get("rsi_7"), 50.0)
    atr = max(_to_float(row.get("atr_14")), 1e-9)
    adx = _to_float(row.get("adx_14"), 20.0)
    body_pct = _to_float(row.get("body_pct_range"))
    vwap_distance_atr = _to_float(row.get("vwap_distance_atr"), 9.0)
    long_side = action == "BUY"
    prior = recent.iloc[:-1]
    breakout_level = prior["micro_high_10"].max() if long_side else prior["micro_low_10"].min()
    breakout_close = prior["close"].max() if long_side else prior["close"].min()
    had_breakout = breakout_close > breakout_level if long_side else breakout_close < breakout_level
    pullback_held = (
        recent["close"].min() >= breakout_level - 0.25 * atr
        if long_side
        else recent["close"].max() <= breakout_level + 0.25 * atr
    )
    resumes = close > open_ and close > ema_9 if long_side else close < open_ and close < ema_9
    trend_ok = close > vwap and ema_9 > ema_21 if long_side else close < vwap and ema_9 < ema_21
    rsi_ok = rsi_7 >= 52.0 if long_side else rsi_7 <= 48.0
    if not (had_breakout and pullback_held and resumes and trend_ok and rsi_ok):
        return 0.0
    score = 58.0
    if cfg.momentum_adx_min <= adx <= cfg.momentum_adx_max:
        score += 10
    if body_pct >= 0.40:
        score += 8
    if vwap_distance_atr <= cfg.ideal_vwap_extension_atr:
        score += 8
    if abs(close - breakout_level) <= 0.9 * atr:
        score += 8
    if _chop_score(row, _to_float(row.get("bb_width")), _to_float(row.get("bb_width_mean_20"))) == 0:
        score += 8
    return score


def _score_vwap_reversion(action: Action, features: pd.DataFrame, cfg: ScalpingConfig) -> float:
    if len(features) < 10:
        return 0.0
    row = features.iloc[-1]
    prev = features.iloc[-2]
    close = _to_float(row.get("close"))
    bb_upper = _to_float(row.get("bb_upper"), close)
    bb_lower = _to_float(row.get("bb_lower"), close)
    rsi_7 = _to_float(row.get("rsi_7"), 50.0)
    adx = _to_float(row.get("adx_14"), 20.0)
    lower_wick = _to_float(row.get("lower_wick_pct"))
    upper_wick = _to_float(row.get("upper_wick_pct"))
    vwap_distance_atr = _to_float(row.get("vwap_distance_atr"), 9.0)
    ema_sep_atr = _to_float(row.get("ema_sep_atr"), 9.0)
    long_side = action == "BUY"
    prev_close = _to_float(prev.get("close"), close)
    touched_extreme = prev_close < bb_lower and close > bb_lower if long_side else prev_close > bb_upper and close < bb_upper
    rsi_extreme = rsi_7 <= cfg.reversion_rsi_long if long_side else rsi_7 >= cfg.reversion_rsi_short
    wick_ok = lower_wick >= 0.30 if long_side else upper_wick >= 0.30
    no_breakout_risk = (
        close > _to_float(row.get("micro_low_10"), close) if long_side else close < _to_float(row.get("micro_high_10"), close)
    )
    divergence = _has_rsi_divergence(features, long_side=long_side)
    score = 0.0
    if touched_extreme:
        score += 18
    if rsi_extreme:
        score += 12
    if adx <= 20.0:
        score += 12
    if wick_ok:
        score += 12
    if vwap_distance_atr <= 1.4:
        score += 10
    if ema_sep_atr <= 0.35:
        score += 10
    if no_breakout_risk:
        score += 10
    if divergence:
        score += 8
    if _chop_score(row, _to_float(row.get("bb_width")), _to_float(row.get("bb_width_mean_20"))) <= 1:
        score += 8
    return score


def _has_rsi_divergence(features: pd.DataFrame, *, long_side: bool) -> bool:
    window = features.iloc[-8:]
    if len(window) < 6:
        return False
    first = window.iloc[:4]
    second = window.iloc[4:]
    if long_side:
        return float(second["close"].min()) < float(first["close"].min()) and float(second["rsi_7"].min()) > float(first["rsi_7"].min())
    return float(second["close"].max()) > float(first["close"].max()) and float(second["rsi_7"].max()) < float(first["rsi_7"].max())


def _signal(
    action: Action,
    setup: SetupName,
    close: float,
    cfg: ScalpingConfig,
    score: float,
    reasons: tuple[str, ...],
    row: pd.Series,
) -> ScalpSignal:
    mult = 1 if action == "BUY" else -1
    return ScalpSignal(
        action=action,
        setup=setup,
        score=score,
        entry_price=round(close, 2),
        stop_loss=round(close - (mult * cfg.stop_points), 2),
        take_profit=round(close + (mult * cfg.target_points), 2),
        reasons=reasons,
        details=_details(row, cfg),
    )


def _hold(reason: str, features: pd.DataFrame, cfg: ScalpingConfig) -> ScalpSignal:
    row = features.iloc[-1] if len(features) else pd.Series(dtype=float)
    close = _to_float(row.get("close")) if len(row) else 0.0
    return ScalpSignal(
        action="HOLD",
        setup="NONE",
        score=0.0,
        entry_price=round(close, 2),
        stop_loss=None,
        take_profit=None,
        reasons=(reason,),
        details=_details(row, cfg) if len(row) else {},
    )


def _details(row: pd.Series, cfg: ScalpingConfig) -> dict[str, float | int | str | bool | None]:
    return {
        "strategy": "independent_scalper",
        "timeframe": "1minute",
        "stop_points": cfg.stop_points,
        "target_points": cfg.target_points,
        "max_hold_minutes": cfg.max_hold_minutes,
        "cooldown_minutes": cfg.cooldown_minutes,
        "max_trades_per_day": cfg.max_trades_per_day,
        "session_vwap": round(_to_float(row.get("session_vwap")), 2),
        "rsi_7": round(_to_float(row.get("rsi_7"), 50.0), 2),
        "rsi_14": round(_to_float(row.get("rsi_14"), 50.0), 2),
        "atr_14": round(_to_float(row.get("atr_14")), 2),
        "adx_14": round(_to_float(row.get("adx_14"), 20.0), 2),
        "bb_upper": round(_to_float(row.get("bb_upper")), 2),
        "bb_mid": round(_to_float(row.get("bb_mid")), 2),
        "bb_lower": round(_to_float(row.get("bb_lower")), 2),
        "micro_high_10": round(_to_float(row.get("micro_high_10")), 2),
        "micro_low_10": round(_to_float(row.get("micro_low_10")), 2),
        "vwap_distance_atr": round(_to_float(row.get("vwap_distance_atr")), 2),
    }
