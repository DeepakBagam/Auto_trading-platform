from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from db.models import RawCandle
from utils.calendar_utils import next_trading_day
from utils.constants import DIRECTION_BUY, DIRECTION_HOLD, DIRECTION_SELL, IST_ZONE

MAX_HORIZON = {
    "1minute": 375,
    "30minute": 13,
}
SESSION_START = time(9, 15)
SLOTS_PER_SESSION = {
    "1minute": 375,
    "30minute": 13,
}


@dataclass
class IntradayPrediction:
    target_ts: datetime
    pred_open: float
    pred_high: float
    pred_low: float
    pred_close: float
    direction: str
    confidence: float
    direction_prob_up: float
    direction_prob_down: float
    direction_prob_flat: float


def _load_candles(db: Session, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    rows = (
        db.execute(
            select(RawCandle)
            .where(and_(RawCandle.interval == interval, RawCandle.instrument_key.like(f"%|{symbol}")))
            .order_by(RawCandle.ts.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    if not rows:
        return pd.DataFrame()
    rows = list(reversed(rows))
    out = pd.DataFrame(
        [
            {
                "ts": r.ts if r.ts.tzinfo is not None else r.ts.replace(tzinfo=IST_ZONE),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in rows
        ]
    )
    return out.sort_values("ts").reset_index(drop=True)


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["ret_1"] = work["close"].pct_change()
    work["ret_3"] = work["close"].pct_change(3)
    work["ret_5"] = work["close"].pct_change(5)
    work["range_pct"] = (work["high"] - work["low"]) / work["close"].replace(0, np.nan)
    work["body_pct"] = (work["close"] - work["open"]) / work["open"].replace(0, np.nan)
    work["vol_chg"] = work["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    work["vol_mean_20"] = work["volume"].rolling(20).mean()
    work["vol_std_20"] = work["volume"].rolling(20).std()
    work["vol_z_20"] = (work["volume"] - work["vol_mean_20"]) / work["vol_std_20"].replace(0, np.nan)
    work["ema_9"] = work["close"].ewm(span=9, adjust=False).mean()
    work["ema_21"] = work["close"].ewm(span=21, adjust=False).mean()
    work["ema_spread"] = (work["ema_9"] - work["ema_21"]) / work["close"].replace(0, np.nan)
    work["volatility_20"] = work["ret_1"].rolling(20).std()
    fill_cols = [
        "ret_1",
        "ret_3",
        "ret_5",
        "range_pct",
        "body_pct",
        "vol_chg",
        "vol_z_20",
        "ema_spread",
        "volatility_20",
    ]
    work[fill_cols] = work[fill_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return work


def _training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat = _feature_frame(df)
    feat["t_open"] = (feat["open"].shift(-1) - feat["close"]) / feat["close"].replace(0, np.nan)
    feat["t_high"] = (feat["high"].shift(-1) - feat["close"]) / feat["close"].replace(0, np.nan)
    feat["t_low"] = (feat["low"].shift(-1) - feat["close"]) / feat["close"].replace(0, np.nan)
    feat["t_close"] = (feat["close"].shift(-1) - feat["close"]) / feat["close"].replace(0, np.nan)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    feature_cols = [
        "ret_1",
        "ret_3",
        "ret_5",
        "range_pct",
        "body_pct",
        "vol_chg",
        "vol_z_20",
        "ema_spread",
        "volatility_20",
    ]
    X = feat[feature_cols].astype(float)
    y = feat[["t_open", "t_high", "t_low", "t_close"]].astype(float)
    return X, y


def _next_bar_ts(prev_ts: datetime, interval: str) -> datetime:
    prev_ts = prev_ts if prev_ts.tzinfo is not None else prev_ts.replace(tzinfo=IST_ZONE)
    if interval == "1minute":
        candidate = prev_ts + timedelta(minutes=1)
        if candidate.time() > time(15, 29):
            nxt_day = next_trading_day(prev_ts.date())
            return datetime.combine(nxt_day, time(9, 15), tzinfo=IST_ZONE)
        if candidate.time() < time(9, 15):
            return datetime.combine(prev_ts.date(), time(9, 15), tzinfo=IST_ZONE)
        return candidate

    # 30-minute slots on NSE session.
    slots = [time(hour=9, minute=15)]
    while slots[-1] < time(15, 15):
        prev = slots[-1]
        dt = datetime.combine(prev_ts.date(), prev, tzinfo=IST_ZONE) + timedelta(minutes=30)
        slots.append(dt.timetz().replace(tzinfo=None))
    cur_t = prev_ts.timetz().replace(tzinfo=None)
    for slot in slots:
        if slot > cur_t:
            return datetime.combine(prev_ts.date(), slot, tzinfo=IST_ZONE)
    nxt_day = next_trading_day(prev_ts.date())
    return datetime.combine(nxt_day, time(9, 15), tzinfo=IST_ZONE)


def _direction_from_probs(p_down: float, p_flat: float, p_up: float) -> str:
    if p_up >= 0.5 and p_up >= p_down:
        return DIRECTION_BUY
    if p_down >= 0.5 and p_down >= p_up:
        return DIRECTION_SELL
    return DIRECTION_HOLD


def _session_slot_index(ts: datetime, interval: str) -> int:
    local_ts = ts if ts.tzinfo is not None else ts.replace(tzinfo=IST_ZONE)
    minutes = (local_ts.hour * 60 + local_ts.minute) - (SESSION_START.hour * 60 + SESSION_START.minute)
    step = 1 if interval == "1minute" else 30
    slot = max(0, minutes // step)
    return int(min(slot, SLOTS_PER_SESSION[interval] - 1))


def _build_seasonal_profile(df: pd.DataFrame, interval: str) -> tuple[dict[int, dict[str, float]], float]:
    work = df.copy().sort_values("ts").reset_index(drop=True)
    work["session_date"] = pd.to_datetime(work["ts"]).dt.date
    work["prev_close"] = work.groupby("session_date")["close"].shift(1).fillna(work["open"])
    work["slot_idx"] = work["ts"].apply(lambda ts: _session_slot_index(ts, interval))
    work["close_ret"] = (work["close"] - work["prev_close"]) / work["prev_close"].replace(0, np.nan)
    work["body_abs_pct"] = (work["close"] - work["open"]).abs() / work["prev_close"].replace(0, np.nan)
    work["range_pct"] = (work["high"] - work["low"]) / work["prev_close"].replace(0, np.nan)
    metric_cols = ["close_ret", "body_abs_pct", "range_pct"]
    work[metric_cols] = work[metric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    profile_df = (
        work.groupby("slot_idx", as_index=False)
        .agg(
            close_ret=("close_ret", "median"),
            abs_close_ret=("close_ret", lambda s: float(np.quantile(np.abs(np.asarray(s, dtype=float)), 0.65))),
            body_abs_pct=("body_abs_pct", "median"),
            range_pct=("range_pct", "median"),
        )
        .sort_values("slot_idx")
    )
    profile = {
        int(row["slot_idx"]): {
            "close_ret": float(row["close_ret"]),
            "abs_close_ret": float(max(row["abs_close_ret"], 1e-5)),
            "body_abs_pct": float(max(row["body_abs_pct"], 1e-5)),
            "range_pct": float(max(row["range_pct"], 2e-5)),
        }
        for _, row in profile_df.iterrows()
    }
    recent_floor = float(work["close_ret"].tail(SLOTS_PER_SESSION[interval]).abs().median())
    return profile, max(recent_floor, 1e-5)


def _profile_for_slot(profile: dict[int, dict[str, float]], slot_idx: int) -> dict[str, float]:
    if slot_idx in profile:
        return profile[slot_idx]
    if not profile:
        return {
            "close_ret": 0.0,
            "abs_close_ret": 1e-4,
            "body_abs_pct": 1e-4,
            "range_pct": 2e-4,
        }
    nearest = min(profile.keys(), key=lambda idx: abs(idx - slot_idx))
    return profile[nearest]


def _safe_stats(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0005
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if not np.isfinite(mu):
        mu = 0.0
    if not np.isfinite(sigma) or sigma < 1e-6:
        sigma = 0.0005
    return mu, sigma


def _stabilize_return(raw_ret: float, step: int, mu: float, sigma: float, interval: str) -> float:
    horizon_scale = 90.0 if interval == "1minute" else 10.0
    alpha = float(np.exp(-max(step - 1, 0) / horizon_scale))
    blended = alpha * float(raw_ret) + (1.0 - alpha) * float(mu)
    clip_base = 0.01 if interval == "1minute" else 0.025
    clip_val = max(clip_base, 3.0 * float(sigma))
    return float(np.clip(blended, -clip_val, clip_val))


def _clamp_close_by_volatility(close_candidate: float, anchor_close: float, step: int, sigma_close: float) -> float:
    if anchor_close <= 0:
        return float(close_candidate)
    band = 4.0 * float(sigma_close) * np.sqrt(max(step, 1))
    band = float(np.clip(band, 0.0025, 0.20))
    lower = anchor_close * (1.0 - band)
    upper = anchor_close * (1.0 + band)
    return float(np.clip(close_candidate, lower, upper))


def predict_intraday_horizon(
    db: Session,
    symbol: str,
    interval: str,
    horizon: int,
    lookback_limit: int | None = None,
) -> dict:
    if interval not in {"1minute", "30minute"}:
        raise ValueError(f"Unsupported intraday interval={interval}")
    if horizon <= 0:
        return {"model_version": None, "predictions": []}
    horizon = min(int(horizon), MAX_HORIZON.get(interval, int(horizon)))

    default_lookback = 8000 if interval == "1minute" else 2500
    data = _load_candles(db, symbol, interval, limit=lookback_limit or default_lookback)
    if len(data) < 500:
        raise ValueError(f"Not enough candles for intraday model: symbol={symbol} interval={interval}")

    X, y = _training_frame(data)
    if len(X) < 400:
        raise ValueError(f"Not enough training rows for intraday model: symbol={symbol} interval={interval}")

    reg_params = dict(max_depth=4, learning_rate=0.05, n_estimators=180, random_state=42)
    reg_open = GradientBoostingRegressor(**reg_params).fit(X, y["t_open"])
    reg_high = GradientBoostingRegressor(**reg_params).fit(X, y["t_high"])
    reg_low = GradientBoostingRegressor(**reg_params).fit(X, y["t_low"])
    reg_close = GradientBoostingRegressor(**reg_params).fit(X, y["t_close"])

    thr = float(max(0.0002, np.percentile(np.abs(y["t_close"].values), 35)))
    y_dir = np.where(y["t_close"].values > thr, 2, np.where(y["t_close"].values < -thr, 0, 1))
    clf = GradientBoostingClassifier(max_depth=3, learning_rate=0.05, n_estimators=180, random_state=42)
    clf.fit(X, y_dir)
    class_to_idx = {int(c): i for i, c in enumerate(clf.classes_)}

    preds: list[IntradayPrediction] = []
    context = data[["ts", "open", "high", "low", "close", "volume"]].copy()
    feature_cols = list(X.columns)
    avg_volume = float(max(1.0, context["volume"].tail(50).mean()))
    anchor_close = float(context.iloc[-1]["close"])
    seasonal_profile, recent_abs_floor = _build_seasonal_profile(data, interval)
    stats = {
        "open": _safe_stats(y["t_open"].to_numpy(dtype=float)),
        "high": _safe_stats(y["t_high"].to_numpy(dtype=float)),
        "low": _safe_stats(y["t_low"].to_numpy(dtype=float)),
        "close": _safe_stats(y["t_close"].to_numpy(dtype=float)),
    }
    model_version = f"intraday_v1_{interval}_{datetime.now(IST_ZONE).strftime('%Y%m%d_%H%M%S')}"

    for step in range(1, horizon + 1):
        feat_ctx = _feature_frame(context).dropna().reset_index(drop=True)
        if feat_ctx.empty:
            break
        x_last = feat_ctx[feature_cols].tail(1).astype(float)
        prev_close = float(context.iloc[-1]["close"])

        r_open = float(reg_open.predict(x_last)[0])
        r_high = float(reg_high.predict(x_last)[0])
        r_low = float(reg_low.predict(x_last)[0])
        r_close = float(reg_close.predict(x_last)[0])
        r_open = _stabilize_return(r_open, step, stats["open"][0], stats["open"][1], interval)
        r_high = _stabilize_return(r_high, step, stats["high"][0], stats["high"][1], interval)
        r_low = _stabilize_return(r_low, step, stats["low"][0], stats["low"][1], interval)
        r_close = _stabilize_return(r_close, step, stats["close"][0], stats["close"][1], interval)
        probs = clf.predict_proba(x_last)[0]
        p_down = float(probs[class_to_idx.get(0, 0)]) if 0 in class_to_idx else 0.0
        p_flat = float(probs[class_to_idx.get(1, 0)]) if 1 in class_to_idx else 0.0
        p_up = float(probs[class_to_idx.get(2, len(probs) - 1)]) if 2 in class_to_idx else 0.0

        next_ts = _next_bar_ts(context.iloc[-1]["ts"], interval)
        slot_idx = _session_slot_index(next_ts, interval)
        slot_profile = _profile_for_slot(seasonal_profile, slot_idx)
        seasonal_ret = float(slot_profile["close_ret"])
        seasonal_abs_ret = float(max(slot_profile["abs_close_ret"], recent_abs_floor))
        classifier_edge = float(np.clip(p_up - p_down, -1.0, 1.0))

        close_signal = 0.55 * r_close + 0.45 * seasonal_ret
        if abs(close_signal) < 0.15 * seasonal_abs_ret:
            close_signal += 0.35 * classifier_edge * seasonal_abs_ret
        body_floor = max(0.65 * seasonal_abs_ret, 0.45 * recent_abs_floor)
        body_cap = max(4.0 * body_floor, 0.012 if interval == "1minute" else 0.03)
        target_body = float(np.clip(abs(close_signal), body_floor, body_cap))
        sign = 1.0 if close_signal > 0 else (-1.0 if close_signal < 0 else (1.0 if classifier_edge >= 0 else -1.0))

        open_shift = float(np.clip(0.25 * r_open, -0.5 * target_body, 0.5 * target_body))
        pred_open = prev_close * (1.0 + open_shift)
        pred_close = pred_open * (1.0 + sign * target_body)
        pred_close = _clamp_close_by_volatility(
            close_candidate=pred_close,
            anchor_close=anchor_close,
            step=step,
            sigma_close=max(stats["close"][1], recent_abs_floor),
        )
        target_range_pct = max(
            float(slot_profile["range_pct"]),
            abs(r_high - r_low),
            abs(float(pred_close - pred_open)) / max(prev_close, 1e-9) * 1.25,
            0.75 * seasonal_abs_ret,
        )
        wick_total = max(
            target_range_pct - abs(float(pred_close - pred_open)) / max(prev_close, 1e-9),
            0.0,
        )
        upper_share = float(np.clip(0.5 + 0.25 * classifier_edge, 0.2, 0.8))
        upper_wick = wick_total * upper_share
        lower_wick = wick_total - upper_wick
        pred_high = max(pred_open, pred_close) * (1.0 + upper_wick)
        pred_low = min(pred_open, pred_close) * (1.0 - lower_wick)
        projected_move_pct = (float(pred_close) - prev_close) / max(prev_close, 1e-9)
        move_floor = max(
            0.00025 if interval == "1minute" else 0.00100,
            0.35 * seasonal_abs_ret,
            0.20 * recent_abs_floor,
        )
        if projected_move_pct > move_floor and (p_up >= 0.40 or classifier_edge > 0.08):
            direction = DIRECTION_BUY
        elif projected_move_pct < -move_floor and (p_down >= 0.40 or classifier_edge < -0.08):
            direction = DIRECTION_SELL
        else:
            direction = _direction_from_probs(p_down, p_flat, p_up)
        horizon_scale = 180.0 if interval == "1minute" else 18.0
        horizon_decay = float(np.exp(-max(step - 1, 0) / horizon_scale))
        dominant_prob = float(max(p_up, p_down, p_flat))
        forecast_edge = float(np.clip(abs(projected_move_pct) / max(move_floor * 2.0, 1e-9), 0.0, 1.0))
        agreement_bonus = 0.0
        if (direction == DIRECTION_BUY and classifier_edge >= 0) or (
            direction == DIRECTION_SELL and classifier_edge <= 0
        ):
            agreement_bonus = 0.08
        confidence = float(
            np.clip((dominant_prob * 0.72 + forecast_edge * 0.28 + agreement_bonus) * horizon_decay, 0.05, 0.99)
        )

        preds.append(
            IntradayPrediction(
                target_ts=next_ts,
                pred_open=float(pred_open),
                pred_high=float(pred_high),
                pred_low=float(pred_low),
                pred_close=float(pred_close),
                direction=direction,
                confidence=confidence,
                direction_prob_up=p_up,
                direction_prob_down=p_down,
                direction_prob_flat=p_flat,
            )
        )

        new_row = {
            "ts": next_ts,
            "open": pred_open,
            "high": pred_high,
            "low": pred_low,
            "close": pred_close,
            "volume": avg_volume,
        }
        context = pd.concat([context, pd.DataFrame([new_row])], ignore_index=True)

    return {
        "model_version": model_version,
        "predictions": preds,
        "source_candle_ts": context.iloc[-len(preds) - 1]["ts"] if preds else context.iloc[-1]["ts"],
        "source_close": anchor_close,
    }
