from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time

from sqlalchemy import and_, desc, func, select
from sqlalchemy.orm import Session

from api.routes.predict import predict as predict_single
from api.routes.signal import latest_price_for_symbol, technical_context_for_symbol
from db.models import ExecutionExternalSignal, ExecutionPosition, SignalLog
from execution_engine.ai_intelligence import score_trade_intelligence
from prediction_engine.adaptive_thresholds import compute_adaptive_thresholds
from prediction_engine.consensus_scoring import (
    EffectiveThresholds,
    derive_candidate_directions,
    derive_ml_direction,
    evaluate_candidate,
    load_threshold_profile,
    select_best_candidate,
    technical_confirmation_direction,
)
from prediction_engine.signal_engine import actionable_direction_from_forecast, build_trade_signal
from utils.config import Settings, get_settings
from utils.constants import IST_ZONE
from utils.market_context import latest_vix_level, recent_news_sentiment_for_symbol
from utils.pine_strategy import local_pine_signal
from utils.symbols import normalize_symbol_key, symbol_value_filter


@dataclass(slots=True)
class ConsensusResult:
    symbol: str
    interval: str
    timestamp: datetime
    ml_signal: str
    ml_confidence: float
    ml_expected_move: float
    ml_reasons: list[str]
    pine_signal: str
    pine_age_seconds: int | None
    ai_score: float
    ai_reasons: list[str]
    news_sentiment: float
    combined_score: float
    consensus: str
    skip_reason: str | None
    trade_placed: bool = False
    details: dict = field(default_factory=dict)


def _parse_time(value: str, fallback: time) -> time:
    try:
        hour, minute = str(value).split(":", 1)
        return time(int(hour), int(minute))
    except Exception:
        return fallback


def _now_ist(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(IST_ZONE)
    return now if now.tzinfo is not None else now.replace(tzinfo=IST_ZONE)


def _latest_pine_signal(db: Session, *, symbol: str, interval: str) -> ExecutionExternalSignal | None:
    return db.scalar(
        select(ExecutionExternalSignal)
        .where(
            and_(
                symbol_value_filter(ExecutionExternalSignal.symbol, symbol),
                ExecutionExternalSignal.interval == interval,
                ExecutionExternalSignal.source.in_(["pine", "pine_v2"]),
            )
        )
        .order_by(ExecutionExternalSignal.signal_ts.desc())
        .limit(1)
    )


def _recent_prediction_accuracy(db: Session, *, symbol: str, lookback: int = 20) -> float:
    rows = (
        db.execute(
            select(ExecutionPosition)
            .where(
                and_(
                    symbol_value_filter(ExecutionPosition.symbol, symbol),
                    ExecutionPosition.status == "CLOSED",
                )
            )
            .order_by(ExecutionPosition.closed_at.desc())
            .limit(max(1, int(lookback)))
        )
        .scalars()
        .all()
    )
    if not rows:
        return 0.5
    wins = sum(1 for row in rows if float(row.realized_pnl or row.pnl_value or 0.0) > 0)
    return wins / max(1, len(rows))


def _daily_realized_pnl(db: Session, trade_date: date) -> float:
    value = db.scalar(
        select(func.sum(ExecutionPosition.realized_pnl)).where(
            and_(
                ExecutionPosition.trade_date == trade_date,
                ExecutionPosition.status == "CLOSED",
            )
        )
    )
    return float(value or 0.0)


def _open_positions_count(db: Session) -> int:
    count = db.scalar(select(func.count(ExecutionPosition.id)).where(ExecutionPosition.status == "OPEN"))
    return int(count or 0)


def _daily_trades_count(db: Session, trade_date: date) -> int:
    count = db.scalar(select(func.count(ExecutionPosition.id)).where(ExecutionPosition.trade_date == trade_date))
    return int(count or 0)


def _build_skip_reason(failures: list[str]) -> str | None:
    return failures[0] if failures else None


def _recent_trade_activity(
    db: Session,
    *,
    symbol: str,
    interval: str,
    lookback: int,
) -> tuple[int, int]:
    try:
        rows = (
            db.execute(
                select(SignalLog)
                .where(
                    and_(
                        symbol_value_filter(SignalLog.symbol, symbol),
                        SignalLog.interval == interval,
                    )
                )
                .order_by(desc(SignalLog.timestamp))
                .limit(max(1, int(lookback)))
            )
            .scalars()
            .all()
        )
    except Exception:
        return 0, int(lookback)
    if not rows:
        return 0, int(lookback)

    recent_trade_count = 0
    bars_since_trade = int(lookback)
    for idx, row in enumerate(rows):
        if str(row.consensus).upper() in {"BUY", "SELL"} or bool(row.trade_placed):
            recent_trade_count += 1
            bars_since_trade = min(bars_since_trade, idx)
    return recent_trade_count, bars_since_trade


def get_consensus_signal(
    db: Session,
    *,
    symbol: str,
    interval: str = "1minute",
    now: datetime | None = None,
    settings: Settings | None = None,
    persist: bool = False,
) -> ConsensusResult:
    settings = settings or get_settings()
    now_ist = _now_ist(now)
    # Disable India VIX trading - no directional edge
    if normalize_symbol_key(symbol) == "INDIAVIX":
        return ConsensusResult(
            symbol=symbol,
            interval=interval,
            timestamp=now_ist,
            ml_signal="HOLD",
            ml_confidence=0.0,
            ml_expected_move=0.0,
            ml_reasons=["India VIX disabled - no directional edge"],
            pine_signal="NEUTRAL",
            pine_age_seconds=None,
            ai_score=0.0,
            ai_reasons=["India VIX disabled"],
            news_sentiment=0.0,
            combined_score=0.0,
            consensus="non_trade_signal",
            skip_reason="India VIX is not suitable for directional trading",
            details={"disabled": True},
        )

    profile = load_threshold_profile(symbol)
    trades_today = _daily_trades_count(db, now_ist.date())
    recent_trade_count, bars_since_trade = _recent_trade_activity(
        db,
        symbol=symbol,
        interval=interval,
        lookback=profile.trade_drought_lookback_bars,
    )

    adaptive = compute_adaptive_thresholds(
        db,
        symbol=symbol,
        base_ml_threshold=float(settings.ml_buy_threshold),
        base_combined_threshold=float(settings.combined_score_threshold),
        base_ai_minimum=float(settings.ai_quality_minimum),
        base_expected_move=float(settings.ml_min_expected_move),
        now=now_ist,
        trades_today=trades_today,
        bars_since_trade=bars_since_trade,
        recent_trade_count=recent_trade_count,
    )
    thresholds = EffectiveThresholds(
        entry_score_threshold=float(adaptive.entry_score_threshold),
        pine_led_score_threshold=float(adaptive.pine_led_score_threshold),
        ml_confidence_floor=float(adaptive.ml_buy_threshold),
        combined_score_floor=float(adaptive.combined_score_threshold),
        ai_score_floor=float(adaptive.ai_quality_minimum),
        pine_led_ai_floor=float(adaptive.pine_led_ai_minimum),
        expected_move_floor=float(adaptive.min_expected_move),
        relaxation_pct=float(adaptive.relaxation_pct),
        reasons=list(adaptive.adjustments.get("activity", {}).get("reasons", [])),
    )

    prediction = predict_single(symbol=symbol, interval=interval, prediction_mode="standard", target_date=None, db=db)
    latest_price = latest_price_for_symbol(db, symbol, interval)
    if latest_price is None:
        raise ValueError(f"No latest price available for {symbol}")

    technical_context = technical_context_for_symbol(db, symbol, interval) or {}
    news_sentiment = recent_news_sentiment_for_symbol(db, symbol=symbol, now=now_ist, lookback_hours=2)
    ml_signal = build_trade_signal(
        symbol=symbol,
        interval=interval,
        direction=prediction.direction,
        confidence=float(prediction.confidence_score or prediction.confidence),
        latest_price=float(latest_price),
        predicted_price=float(prediction.pred_close),
        pred_high=float(prediction.pred_high),
        pred_low=float(prediction.pred_low),
        pred_interval_close=(
            prediction.pred_interval.model_dump().get("close")
            if prediction.pred_interval is not None
            else None
        ),
        technical_context=technical_context,
        buy_threshold=adaptive.ml_buy_threshold,
        sell_threshold=adaptive.ml_sell_threshold,
        minimum_expected_move=adaptive.min_expected_move,
        news_sentiment=news_sentiment,
    )
    expected_move_points = float(prediction.pred_close) - float(latest_price)
    forecast_action = actionable_direction_from_forecast(
        direction=prediction.direction,
        latest_price=float(latest_price),
        predicted_price=float(prediction.pred_close),
        technical_context=technical_context,
    )
    ml_direction = derive_ml_direction(direction=prediction.direction, expected_move_points=expected_move_points)

    pine_row = _latest_pine_signal(db, symbol=symbol, interval=interval)
    pine_local = local_pine_signal(technical_context, symbol=symbol)
    pine_signal = "NEUTRAL"
    pine_age_seconds = None
    pine_source = "none"
    if pine_row is not None:
        pine_signal = str(pine_row.signal_action or "NEUTRAL").upper()
        pine_age_seconds = max(0, int((now_ist - (pine_row.signal_ts if pine_row.signal_ts.tzinfo else pine_row.signal_ts.replace(tzinfo=IST_ZONE))).total_seconds()))
        pine_source = "external"
    if (
        pine_source != "external"
        or pine_signal not in {"BUY", "SELL"}
        or (pine_age_seconds is not None and pine_age_seconds >= int(profile.pine_max_age_seconds))
    ) and pine_local["action"] in {"BUY", "SELL"}:
        pine_signal = str(pine_local["action"])
        pine_age_seconds = 0
        pine_source = "local_fallback"
    if pine_signal not in {"BUY", "SELL"}:
        technical_fallback = technical_confirmation_direction(technical_context)
        if technical_fallback in {"BUY", "SELL"}:
            pine_signal = technical_fallback
            pine_age_seconds = 0
            pine_source = "scorecard_fallback"

    vix_level = latest_vix_level(db, interval=interval)
    recent_accuracy = _recent_prediction_accuracy(db, symbol=symbol)
    expected_return_pct = 0.0 if float(latest_price) == 0 else expected_move_points / float(latest_price)
    ai_by_direction: dict[str, Any] = {}

    def _ai_for_direction(direction: str):
        normalized = str(direction or "HOLD").upper()
        cached = ai_by_direction.get(normalized)
        if cached is not None:
            return cached
        ai_by_direction[normalized] = score_trade_intelligence(
            signal_action=normalized,
            confidence=float(prediction.confidence_score or prediction.confidence),
            expected_return_pct=float(expected_return_pct),
            technical_context=technical_context,
            now=now_ist,
            news_sentiment=news_sentiment,
            recent_prediction_accuracy=recent_accuracy,
            vix_level=vix_level,
        )
        return ai_by_direction[normalized]

    candidate_evaluations = []
    for candidate_direction, source in derive_candidate_directions(
        ml_direction=ml_direction,
        pine_signal=pine_signal,
    ):
        ai = _ai_for_direction(candidate_direction)
        candidate_evaluations.append(
            evaluate_candidate(
                direction=candidate_direction,
                source=source,
                ml_direction=ml_direction,
                ml_trade_action=ml_signal.action,
                ml_confidence=float(prediction.confidence_score or prediction.confidence),
                expected_move_points=float(expected_move_points),
                ai_score=float(ai.score),
                pine_signal=pine_signal,
                pine_age_seconds=pine_age_seconds,
                technical_context=technical_context,
                thresholds=thresholds,
                pine_max_age_seconds=int(profile.pine_max_age_seconds),
            )
        )
    best_candidate = select_best_candidate(candidate_evaluations)
    fallback_ai = _ai_for_direction(ml_direction if ml_direction in {"BUY", "SELL"} else "HOLD")
    ai_signal_action = best_candidate.direction if best_candidate is not None else (
        ml_direction if ml_direction in {"BUY", "SELL"} else "HOLD"
    )
    combined_score = float(best_candidate.combined_score) if best_candidate is not None else 0.0
    ai_score = float(best_candidate.ai_score) if best_candidate is not None else float(fallback_ai.score)

    failures: list[str] = []
    start = _parse_time(settings.entry_window_start, time(9, 20))
    end = _parse_time(settings.entry_window_end, time(13, 30))
    current_time = now_ist.timetz().replace(tzinfo=None)
    if not (start <= current_time <= end):
        failures.append(f"Outside entry window ({current_time.strftime('%H:%M:%S')} not in {start.strftime('%H:%M')}-{end.strftime('%H:%M')})")
    loss_limit = -abs(float(settings.execution_capital) * float(settings.execution_max_daily_loss_pct))
    daily_realized = _daily_realized_pnl(db, now_ist.date())
    if daily_realized <= loss_limit:
        failures.append(f"Daily loss limit hit ({daily_realized:.2f} <= {loss_limit:.2f})")
    if _open_positions_count(db) >= int(settings.execution_max_simultaneous_trades):
        failures.append(f"Max simultaneous trades reached ({_open_positions_count(db)} >= {settings.execution_max_simultaneous_trades})")
    if trades_today >= int(settings.execution_max_daily_trades):
        failures.append(f"Max daily trades reached ({trades_today} >= {settings.execution_max_daily_trades})")
    if best_candidate is None:
        failures.append("No directional candidate from ML or Pine")
    elif not best_candidate.qualifies:
        failures.append(
            f"Score {best_candidate.weighted_score:.0f} < {thresholds.entry_score_threshold:.0f}"
            if best_candidate.weighted_score < thresholds.entry_score_threshold and not best_candidate.pine_led_eligible
            else f"Combined score {best_candidate.combined_score:.2f} < {thresholds.combined_score_floor:.2f}"
        )

    consensus = (
        best_candidate.direction
        if best_candidate is not None and best_candidate.qualifies and not failures
        else "non_trade_signal"
    )
    
    # Apply quality filters with slightly relaxed thresholds
    if consensus in {"BUY", "SELL"} and best_candidate is not None:
        skip_reasons = []
        ai_score_val = float(best_candidate.ai_score)
        weighted_score_val = float(best_candidate.weighted_score)
        
        # Filter 1: Minimum AI score (40, with secondary condition for 38-40)
        if ai_score_val < 38.0:
            skip_reasons.append(f"AI score too low ({ai_score_val:.1f} < 38)")
        elif 38.0 <= ai_score_val < 40.0 and weighted_score_val < 72.0:
            skip_reasons.append(f"AI score {ai_score_val:.1f} requires weighted score >= 72 (got {weighted_score_val:.1f})")
        
        # Filter 2: Minimum weighted score (70, relaxed from 72)
        if weighted_score_val < 70.0:
            skip_reasons.append(f"Weighted score too low ({weighted_score_val:.1f} < 70)")
        
        # Filter 3: Stricter filter for SENSEX
        if normalize_symbol_key(symbol) == "SENSEX" and ai_score_val < 45.0:
            skip_reasons.append(f"SENSEX requires AI score >= 45 (got {ai_score_val:.1f})")
        
        # Filter 3: Avoid range-bound markets (relaxed - only skip strong range)
        market_regime = technical_context.get("mtf_5m_market_regime") or technical_context.get("mtf_3m_market_regime") or "trend"
        adx = technical_context.get("adx_14", 25)
        if market_regime == "range" and adx < 15:
            skip_reasons.append("Strong range-bound market with low ADX - no directional edge")
        
        # Filter 4: Momentum filter - require strong candle
        atr = technical_context.get("atr_14")
        candle_range = abs(float(technical_context.get("high", 0)) - float(technical_context.get("low", 0)))
        if atr and candle_range < float(atr) * 0.6:
            skip_reasons.append(f"Weak candle momentum (range {candle_range:.1f} < 0.6*ATR)")
        
        # Filter 5: Require breakout or strong continuation
        close = technical_context.get("close")
        breakout_high = technical_context.get("breakout_high_20")
        breakout_low = technical_context.get("breakout_low_20")
        ema21 = technical_context.get("ema_21")
        ema21_slope = technical_context.get("ema_21_slope_3")
        
        has_breakout = False
        has_continuation = False
        
        if close and breakout_high and breakout_low:
            if consensus == "BUY" and close > breakout_high:
                has_breakout = True
            elif consensus == "SELL" and close < breakout_low:
                has_breakout = True
        
        if close and ema21 and ema21_slope:
            if consensus == "BUY" and close > ema21 and ema21_slope > 0:
                has_continuation = True
            elif consensus == "SELL" and close < ema21 and ema21_slope < 0:
                has_continuation = True
        
        if not has_breakout and not has_continuation:
            skip_reasons.append("No breakout or strong continuation pattern")
        
        # Apply filters
        if skip_reasons:
            consensus = "non_trade_signal"
            failures.extend(skip_reasons)
    
    result = ConsensusResult(
        symbol=symbol,
        interval=interval,
        timestamp=now_ist,
        ml_signal=str(ml_direction),
        ml_confidence=float(prediction.confidence_score or prediction.confidence),
        ml_expected_move=float(ml_signal.expected_move_points),
        ml_reasons=list(ml_signal.reasons),
        pine_signal=pine_signal,
        pine_age_seconds=pine_age_seconds,
        ai_score=ai_score,
        ai_reasons=list((_ai_for_direction(ai_signal_action)).reasons),
        news_sentiment=float(news_sentiment),
        combined_score=float(combined_score),
        consensus=consensus,
        skip_reason=_build_skip_reason(failures),
        details={
            "prediction_direction": prediction.direction,
            "forecast_action": forecast_action,
            "ai_signal_action": ai_signal_action,
            "model_trade_signal": ml_signal.action,
            "candidate_direction": best_candidate.direction if best_candidate is not None else "HOLD",
            "candidate_source": best_candidate.source if best_candidate is not None else "none",
            "predicted_close": float(prediction.pred_close),
            "latest_price": float(latest_price),
            "vix_level": vix_level,
            "pine_max_age_seconds": int(profile.pine_max_age_seconds),
            "failures": failures,
            "pine_score": float(best_candidate.pine_score) if best_candidate is not None else 0.0,
            "pine_source": pine_source,
            "local_pine_signal": pine_local["action"],
            "local_buy_strategies": pine_local["buy_strategy_names"],
            "local_sell_strategies": pine_local["sell_strategy_names"],
            "market_regime": technical_context.get("mtf_5m_market_regime") or technical_context.get("mtf_3m_market_regime") or pine_local.get("market_regime"),
            "mtf_3m_action": technical_context.get("mtf_3m_action"),
            "mtf_5m_action": technical_context.get("mtf_5m_action"),
            "mtf_3m_buy_score": technical_context.get("mtf_3m_buy_score"),
            "mtf_5m_buy_score": technical_context.get("mtf_5m_buy_score"),
            "mtf_3m_sell_score": technical_context.get("mtf_3m_sell_score"),
            "mtf_5m_sell_score": technical_context.get("mtf_5m_sell_score"),
            "mtf_score": float(best_candidate.mtf_score) if best_candidate is not None else 0.0,
            "weighted_score": float(best_candidate.weighted_score) if best_candidate is not None else 0.0,
            "component_flags": (
                {
                    "ml": best_candidate.ml_component,
                    "ai": best_candidate.ai_component,
                    "pine": best_candidate.pine_component,
                    "mtf": best_candidate.mtf_component,
                }
                if best_candidate is not None
                else {"ml": False, "ai": False, "pine": False, "mtf": False}
            ),
            "candidate_evaluations": [
                {
                    "direction": candidate.direction,
                    "source": candidate.source,
                    "ai_score": candidate.ai_score,
                    "weighted_score": candidate.weighted_score,
                    "combined_score": candidate.combined_score,
                    "pine_score": candidate.pine_score,
                    "mtf_score": candidate.mtf_score,
                    "qualifies": candidate.qualifies,
                    "pine_led_eligible": candidate.pine_led_eligible,
                }
                for candidate in candidate_evaluations
            ],
            "recent_trade_activity": {
                "trades_today": trades_today,
                "recent_trade_count": recent_trade_count,
                "bars_since_trade": bars_since_trade,
            },
            "adaptive_thresholds": {
                "ml_threshold": adaptive.ml_buy_threshold,
                "combined_threshold": adaptive.combined_score_threshold,
                "ai_minimum": adaptive.ai_quality_minimum,
                "expected_move": adaptive.min_expected_move,
                "entry_score_threshold": adaptive.entry_score_threshold,
                "pine_led_score_threshold": adaptive.pine_led_score_threshold,
                "pine_led_ai_minimum": adaptive.pine_led_ai_minimum,
                "relaxation_pct": adaptive.relaxation_pct,
                "reason": adaptive.reason,
            },
        },
    )
    if persist:
        log_consensus_result(db, result)
    return result


def log_consensus_result(db: Session, result: ConsensusResult) -> SignalLog:
    row = SignalLog(
        timestamp=result.timestamp,
        trade_date=result.timestamp.date(),
        symbol=result.symbol,
        interval=result.interval,
        ml_signal=result.ml_signal,
        ml_confidence=result.ml_confidence,
        ml_expected_move=result.ml_expected_move,
        pine_signal=result.pine_signal,
        pine_age_seconds=result.pine_age_seconds,
        ai_score=result.ai_score,
        news_sentiment=result.news_sentiment,
        combined_score=result.combined_score,
        consensus=result.consensus,
        skip_reason=result.skip_reason,
        trade_placed=bool(result.trade_placed),
        details=result.details,
    )
    db.add(row)
    db.flush()
    return row
