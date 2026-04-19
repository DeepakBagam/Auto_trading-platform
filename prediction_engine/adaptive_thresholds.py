"""Adaptive threshold adjustment for score-driven consensus decisions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from db.models import ExecutionPosition
from prediction_engine.consensus_scoring import (
    compute_effective_thresholds,
    load_threshold_profile,
)
from utils.constants import IST_ZONE
from utils.logger import get_logger
from utils.symbols import symbol_value_filter

logger = get_logger(__name__)


@dataclass
class AdaptiveThresholds:
    ml_buy_threshold: float
    ml_sell_threshold: float
    combined_score_threshold: float
    ai_quality_minimum: float
    min_expected_move: float
    entry_score_threshold: float
    pine_led_score_threshold: float
    pine_led_ai_minimum: float
    relaxation_pct: float
    reason: str
    adjustments: dict


def get_recent_performance(
    db: Session, 
    symbol: str | None = None, 
    lookback_days: int = 30
) -> dict:
    """Get recent trading performance metrics."""
    cutoff = datetime.now(IST_ZONE) - timedelta(days=lookback_days)
    
    query = select(ExecutionPosition).where(
        and_(
            ExecutionPosition.status == "CLOSED",
            ExecutionPosition.closed_at >= cutoff,
        )
    )
    
    if symbol:
        query = query.where(symbol_value_filter(ExecutionPosition.symbol, symbol))
    
    positions = db.execute(query).scalars().all()
    
    if not positions:
        return {
            "total_trades": 0,
            "win_rate": 0.5,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 1.0,
            "max_consecutive_losses": 0,
        }
    
    wins = [p for p in positions if float(p.realized_pnl or p.pnl_value or 0.0) > 0]
    losses = [p for p in positions if float(p.realized_pnl or p.pnl_value or 0.0) <= 0]
    
    win_rate = len(wins) / len(positions) if positions else 0.5
    avg_profit = (
        sum(float(p.realized_pnl or p.pnl_value or 0.0) for p in wins) / len(wins)
        if wins else 0.0
    )
    avg_loss = (
        sum(abs(float(p.realized_pnl or p.pnl_value or 0.0)) for p in losses) / len(losses)
        if losses else 0.0
    )
    profit_factor = (
        (avg_profit * len(wins)) / (avg_loss * len(losses))
        if losses and avg_loss > 0 else 1.0
    )
    
    # Calculate max consecutive losses
    max_consecutive = 0
    current_consecutive = 0
    for p in sorted(positions, key=lambda x: x.closed_at):
        if float(p.realized_pnl or p.pnl_value or 0.0) <= 0:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return {
        "total_trades": len(positions),
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_consecutive_losses": max_consecutive,
    }


def get_time_of_day_adjustment(now: datetime) -> dict:
    """Expose time-of-day context for diagnostics."""
    current_time = now.time()
    if time(11, 0) <= current_time < time(13, 0):
        return {
            "multiplier": 0.90,
            "reason": "lunch_relax_10pct",
            "expected_move_multiplier": 0.90,
        }
    return {
        "multiplier": 1.0,
        "reason": "normal_hours",
        "expected_move_multiplier": 1.0,
    }


def get_day_of_week_adjustment(now: datetime) -> dict:
    """Expose day-of-week context for diagnostics."""
    weekday = now.weekday()  # 0=Monday, 4=Friday
    if weekday == 4:  # Friday
        return {
            "multiplier": 0.95,
            "reason": "friday_relax_5pct",
        }
    return {
        "multiplier": 1.0,
        "reason": "mid_week",
    }


def compute_adaptive_thresholds(
    db: Session,
    symbol: str,
    base_ml_threshold: float = 0.62,
    base_combined_threshold: float = 0.65,
    base_ai_minimum: float = 65.0,
    base_expected_move: float = 80.0,
    now: datetime | None = None,
    trades_today: int = 0,
    bars_since_trade: int | None = None,
    recent_trade_count: int | None = None,
) -> AdaptiveThresholds:
    """Compute scorecard thresholds with activity-aware relaxation."""
    now = now or datetime.now(IST_ZONE)

    try:
        perf = get_recent_performance(db, symbol=symbol, lookback_days=30)
    except Exception as exc:
        logger.debug("Adaptive performance fallback for %s: %s", symbol, exc)
        perf = {
            "total_trades": 0,
            "win_rate": 0.5,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 1.0,
            "max_consecutive_losses": 0,
        }
    profile = load_threshold_profile(symbol)
    if base_ml_threshold > 0:
        profile = profile.__class__(
            symbol_key=profile.symbol_key,
            entry_score_threshold=profile.entry_score_threshold,
            pine_led_score_threshold=profile.pine_led_score_threshold,
            ml_confidence_floor=min(profile.ml_confidence_floor, float(base_ml_threshold)),
            combined_score_floor=min(profile.combined_score_floor, float(base_combined_threshold)),
            ai_score_floor=min(profile.ai_score_floor, float(base_ai_minimum)),
            pine_led_ai_floor=profile.pine_led_ai_floor,
            expected_move_floor=min(profile.expected_move_floor, float(base_expected_move)),
            pine_max_age_seconds=profile.pine_max_age_seconds,
            trade_drought_lookback_bars=profile.trade_drought_lookback_bars,
            trade_drought_relax_pct=profile.trade_drought_relax_pct,
            lunch_relax_pct=profile.lunch_relax_pct,
            friday_relax_pct=profile.friday_relax_pct,
            low_activity_trade_target=profile.low_activity_trade_target,
            low_activity_relax_pct=profile.low_activity_relax_pct,
            max_total_relax_pct=profile.max_total_relax_pct,
        )

    time_adj = get_time_of_day_adjustment(now)
    day_adj = get_day_of_week_adjustment(now)
    effective = compute_effective_thresholds(
        profile,
        now=now,
        trades_today=trades_today,
        bars_since_trade=bars_since_trade,
        recent_trade_count=recent_trade_count,
    )

    adjustments = {
        "performance": {},
        "time": time_adj,
        "day": day_adj,
        "activity": {
            "trades_today": trades_today,
            "bars_since_trade": bars_since_trade,
            "recent_trade_count": recent_trade_count,
            "relaxation_pct": effective.relaxation_pct,
            "reasons": effective.reasons,
        },
    }

    ml_threshold = effective.ml_confidence_floor
    combined_threshold = effective.combined_score_floor
    ai_minimum = effective.ai_score_floor
    expected_move = effective.expected_move_floor

    if perf["total_trades"] >= 10:
        win_rate = perf["win_rate"]
        profit_factor = perf["profit_factor"]
        if win_rate > 0.78 and profit_factor > 1.2:
            ml_threshold *= 0.96
            combined_threshold *= 0.95
            ai_minimum *= 0.94
            adjustments["performance"] = {
                "adjustment": "relax",
                "reason": f"overfiltered_recent_wr_{win_rate:.2f}_pf_{profit_factor:.2f}",
                "multiplier": 0.95,
            }
        elif win_rate < 0.52 and profit_factor < 0.95:
            ml_threshold *= 1.04
            combined_threshold *= 1.04
            ai_minimum *= 1.03
            expected_move *= 1.05
            adjustments["performance"] = {
                "adjustment": "tighten",
                "reason": f"low_performance_wr_{win_rate:.2f}_pf_{profit_factor:.2f}",
                "multiplier": 1.04,
            }
        if perf["max_consecutive_losses"] >= 3:
            ml_threshold *= 1.03
            combined_threshold *= 1.03
            ai_minimum *= 1.02
            adjustments["performance"]["consecutive_losses"] = perf["max_consecutive_losses"]

    minimum_expected_move = 0.05 if profile.symbol_key == "INDIAVIX" else 10.0
    ml_threshold = max(0.45, min(0.75, ml_threshold))
    combined_threshold = max(0.40, min(0.70, combined_threshold))
    ai_minimum = max(25.0, min(60.0, ai_minimum))
    expected_move = max(minimum_expected_move, expected_move)

    reason_parts = []
    if adjustments["performance"]:
        reason_parts.append(adjustments["performance"].get("reason", "performance"))
    reason_parts.append(time_adj["reason"])
    reason_parts.append(day_adj["reason"])
    reason_parts.extend(effective.reasons)

    return AdaptiveThresholds(
        ml_buy_threshold=round(ml_threshold, 3),
        ml_sell_threshold=round(ml_threshold, 3),
        combined_score_threshold=round(combined_threshold, 3),
        ai_quality_minimum=round(ai_minimum, 1),
        min_expected_move=round(expected_move, 1),
        entry_score_threshold=round(effective.entry_score_threshold, 1),
        pine_led_score_threshold=round(effective.pine_led_score_threshold, 1),
        pine_led_ai_minimum=round(effective.pine_led_ai_floor, 1),
        relaxation_pct=round(effective.relaxation_pct, 4),
        reason=" | ".join(reason_parts),
        adjustments=adjustments,
    )
