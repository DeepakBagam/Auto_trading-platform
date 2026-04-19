from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from api.deps import get_db
from api.routes.predict import predict as predict_single
from api.schemas import SignalResponse
from db.models import RawCandle
from feature_engine.price_features import build_price_features
from prediction_engine.signal_engine import build_trade_signal
from utils.intervals import INTERVAL_QUERY_PATTERN, normalize_interval
from utils.pine_strategy import attach_short_interval_states
from utils.symbols import instrument_key_filter

router = APIRouter(prefix="/signal", tags=["signal"])

TECHNICAL_CONTEXT_KEYS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "atr_14",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_hist_delta_1",
    "ema_9",
    "ema_21",
    "ema_21_slope_3",
    "ema_50",
    "ema_200",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "kc_upper",
    "kc_mid",
    "kc_lower",
    "volume_sma_20",
    "volume_ratio_20",
    "breakout_high_20",
    "breakout_low_20",
    "candle_green",
    "candle_red",
    "volatility_20d",
    "body_size",
    "body_pct_range",
    "upper_wick_pct",
    "lower_wick_pct",
    "pattern_hanging_man",
    "pattern_shooting_star",
    "pattern_marubozu",
    "pattern_spinning_top",
    "pattern_engulfing",
    "mtf_3m_action",
    "mtf_3m_buy_score",
    "mtf_3m_sell_score",
    "mtf_3m_market_regime",
    "mtf_5m_action",
    "mtf_5m_buy_score",
    "mtf_5m_sell_score",
    "mtf_5m_market_regime",
}


def latest_price_for_symbol(db: Session, symbol: str, interval: str) -> float | None:
    price = db.scalar(
        select(RawCandle.close)
        .where(
            and_(
                instrument_key_filter(RawCandle.instrument_key, symbol),
                RawCandle.interval == interval,
            )
        )
        .order_by(RawCandle.ts.desc())
        .limit(1)
    )
    return float(price) if price is not None else None


def latest_candle_ts_for_symbol(db: Session, symbol: str, interval: str) -> pd.Timestamp | None:
    candle_ts = db.scalar(
        select(RawCandle.ts)
        .where(
            and_(
                instrument_key_filter(RawCandle.instrument_key, symbol),
                RawCandle.interval == interval,
            )
        )
        .order_by(RawCandle.ts.desc())
        .limit(1)
    )
    return pd.Timestamp(candle_ts) if candle_ts is not None else None


def technical_context_for_symbol(db: Session, symbol: str, interval: str) -> dict | None:
    try:
        rows = (
            db.execute(
                select(RawCandle)
                .where(
                    and_(
                        instrument_key_filter(RawCandle.instrument_key, symbol),
                        RawCandle.interval == interval,
                    )
                )
                .order_by(RawCandle.ts.desc())
                .limit(260)
            )
            .scalars()
            .all()
        )
    except Exception:
        return None

    if len(rows) < 30:
        return None

    rows = list(reversed(rows))
    raw_frame = pd.DataFrame(
        {
            "ts": [r.ts for r in rows],
            "open": [float(r.open) for r in rows],
            "high": [float(r.high) for r in rows],
            "low": [float(r.low) for r in rows],
            "close": [float(r.close) for r in rows],
            "volume": [float(r.volume) for r in rows],
        }
    )
    frame = build_price_features(raw_frame)
    frame = attach_short_interval_states(frame, symbol=symbol)
    latest = frame.iloc[-1].to_dict()
    return {k: latest.get(k) for k in TECHNICAL_CONTEXT_KEYS if k in latest}


def build_signal_snapshot(
    *,
    symbol: str,
    interval: str,
    prediction_mode: str,
    direction: str,
    confidence: float,
    confidence_bucket: str | None,
    latest_price: float,
    predicted_price: float,
    pred_high: float,
    pred_low: float,
    pred_interval: dict | None,
    technical_context: dict | None,
    target_session_date,
    target_ts,
    generated_at,
    model_version: str,
) -> SignalResponse:
    pred_interval = pred_interval if isinstance(pred_interval, dict) else {}
    signal_out = build_trade_signal(
        symbol=symbol,
        interval=interval,
        direction=direction,
        confidence=float(confidence),
        latest_price=latest_price,
        predicted_price=float(predicted_price),
        pred_high=float(pred_high),
        pred_low=float(pred_low),
        pred_interval_close=pred_interval.get("close") if isinstance(pred_interval, dict) else None,
        technical_context=technical_context,
    )
    return SignalResponse(
        symbol=symbol,
        interval=interval,
        prediction_mode=prediction_mode,
        action=signal_out.action,
        conviction=signal_out.conviction,
        latest_price=float(latest_price),
        predicted_price=float(predicted_price),
        expected_return_pct=signal_out.expected_return_pct,
        expected_move_points=signal_out.expected_move_points,
        confidence=float(confidence),
        confidence_bucket=confidence_bucket,
        stop_loss=signal_out.stop_loss,
        take_profit=signal_out.take_profit,
        risk_reward_ratio=signal_out.risk_reward_ratio,
        technical_score=signal_out.technical_score,
        target_session_date=target_session_date,
        target_ts=target_ts,
        generated_at=generated_at,
        model_version=model_version,
        reasons=signal_out.reasons,
    )


@router.get("", response_model=SignalResponse)
def signal(
    symbol: str = Query(..., description="Trading symbol"),
    interval: str = Query("1minute", pattern=INTERVAL_QUERY_PATTERN, description="Signal interval"),
    prediction_mode: str = Query("standard", pattern="^(standard|session_close)$"),
    db: Session = Depends(get_db),
) -> SignalResponse:
    try:
        interval = normalize_interval(interval)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    prediction = predict_single(
        symbol=symbol,
        interval=interval,
        prediction_mode=prediction_mode,
        target_date=None,
        db=db,
    )
    latest_price = latest_price_for_symbol(db, symbol, interval)
    if latest_price is None:
        raise HTTPException(status_code=404, detail=f"No market price available for {symbol}")

    pred_interval = prediction.pred_interval.model_dump() if prediction.pred_interval is not None else {}
    technical_context = technical_context_for_symbol(db, symbol, interval)
    return build_signal_snapshot(
        symbol=prediction.symbol,
        interval=prediction.interval,
        prediction_mode=prediction.prediction_mode,
        direction=prediction.direction,
        confidence=float(prediction.confidence_score or prediction.confidence),
        confidence_bucket=prediction.confidence_bucket,
        latest_price=float(latest_price),
        predicted_price=float(prediction.pred_close),
        pred_high=float(prediction.pred_high),
        pred_low=float(prediction.pred_low),
        pred_interval=pred_interval,
        technical_context=technical_context,
        target_session_date=prediction.target_session_date,
        target_ts=prediction.target_ts,
        generated_at=prediction.generated_at,
        model_version=prediction.model_version,
    )
