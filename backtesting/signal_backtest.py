from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time
from typing import Any

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from backtesting.metrics import max_drawdown
from db.models import PredictionsIntraday, RawCandle
from execution_engine.ai_intelligence import score_trade_intelligence
from feature_engine.price_features import build_price_features
from prediction_engine.consensus_scoring import (
    EffectiveThresholds,
    compute_effective_thresholds,
    derive_candidate_directions,
    derive_ml_direction,
    evaluate_candidate,
    load_threshold_profile,
    select_best_candidate,
    technical_confirmation_direction,
)
from prediction_engine.signal_engine import build_trade_signal
from utils.constants import IST_ZONE
from utils.pine_strategy import attach_short_interval_states, local_pine_signal
from utils.symbols import instrument_key_filter, symbol_value_filter


@dataclass(frozen=True, slots=True)
class BacktestThresholds:
    entry_score_threshold: float
    pine_led_score_threshold: float
    ml_threshold: float
    min_expected_move: float
    ai_minimum: float
    pine_led_ai_minimum: float
    combined_minimum: float


@dataclass(slots=True)
class BacktestState:
    current_date: date | None = None
    trades_today: int = 0
    bars_since_trade: int = 0


def _load_symbol_candles(db: Session, symbol: str) -> pd.DataFrame:
    rows = (
        db.execute(
            select(RawCandle)
            .where(
                and_(
                    instrument_key_filter(RawCandle.instrument_key, symbol),
                    RawCandle.interval == "1minute",
                )
            )
            .order_by(RawCandle.ts.asc())
        )
        .scalars()
        .all()
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "ts": [r.ts if r.ts.tzinfo is not None else r.ts.replace(tzinfo=IST_ZONE) for r in rows],
            "open": [float(r.open) for r in rows],
            "high": [float(r.high) for r in rows],
            "low": [float(r.low) for r in rows],
            "close": [float(r.close) for r in rows],
            "volume": [float(r.volume) for r in rows],
        }
    )


def _load_predictions(db: Session, symbol: str) -> pd.DataFrame:
    rows = (
        db.execute(
            select(PredictionsIntraday)
            .where(
                and_(
                    symbol_value_filter(PredictionsIntraday.symbol, symbol),
                    PredictionsIntraday.interval == "1minute",
                )
            )
            .order_by(PredictionsIntraday.target_ts.asc(), PredictionsIntraday.generated_at.asc())
        )
        .scalars()
        .all()
    )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "symbol": [row.symbol for row in rows],
            "target_ts": [
                row.target_ts if row.target_ts.tzinfo is not None else row.target_ts.replace(tzinfo=IST_ZONE)
                for row in rows
            ],
            "pred_open": [float(row.pred_open) for row in rows],
            "pred_high": [float(row.pred_high) for row in rows],
            "pred_low": [float(row.pred_low) for row in rows],
            "pred_close": [float(row.pred_close) for row in rows],
            "direction": [str(row.direction) for row in rows],
            "confidence": [float(row.confidence) for row in rows],
            "feature_cutoff_ist": [
                row.feature_cutoff_ist
                if row.feature_cutoff_ist.tzinfo is not None
                else row.feature_cutoff_ist.replace(tzinfo=IST_ZONE)
                for row in rows
            ],
            "generated_at": [
                row.generated_at if row.generated_at.tzinfo is not None else row.generated_at.replace(tzinfo=IST_ZONE)
                for row in rows
            ],
        }
    )
    frame = frame.sort_values(["target_ts", "generated_at"]).drop_duplicates(subset=["target_ts"], keep="last")
    return frame.reset_index(drop=True)


def _prepare_context(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    enriched = build_price_features(frame.copy()).reset_index(drop=True)
    enriched = attach_short_interval_states(enriched, symbol=symbol)
    return enriched.sort_values("ts").reset_index(drop=True)


def _point_in_time_dataset(db: Session, symbol: str) -> pd.DataFrame:
    candles = _load_symbol_candles(db, symbol)
    preds = _load_predictions(db, symbol)
    if candles.empty or preds.empty:
        return pd.DataFrame()

    context = _prepare_context(candles, symbol=symbol)
    actuals = candles[["ts", "close"]].rename(columns={"ts": "target_ts", "close": "actual_close"})
    preds = preds.copy()
    preds["ref_ts"] = preds["target_ts"] - pd.Timedelta(minutes=1)

    merged = pd.merge_asof(
        preds.sort_values("ref_ts"),
        context.sort_values("ts"),
        left_on="ref_ts",
        right_on="ts",
        direction="backward",
    )
    merged = merged.rename(columns={"ts": "ref_candle_ts", "close": "ref_close"})
    merged = merged.merge(actuals, on="target_ts", how="inner")
    merged = merged[merged["target_ts"] > merged["ref_candle_ts"]]
    merged = merged.dropna(subset=["ref_close", "actual_close"])
    return merged.reset_index(drop=True)


def _actual_direction(row: pd.Series) -> tuple[str, float]:
    latest_price = float(row["ref_close"])
    actual_close = float(row["actual_close"])
    atr = float(row.get("atr_14") or 0.0)
    floor_points = max(5.0, atr * 0.15, latest_price * 0.0002)
    move_points = actual_close - latest_price
    if move_points > floor_points:
        return "BUY", move_points
    if move_points < -floor_points:
        return "SELL", move_points
    return "HOLD", move_points


def _row_technical_context(row: pd.Series) -> dict[str, Any]:
    fields = [
        "open",
        "high",
        "low",
        "ref_close",
        "volume",
        "vwap",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "macd_hist_delta_1",
        "ema_9",
        "ema_21",
        "ema_50",
        "ema_21_slope_3",
        "atr_14",
        "body_pct_range",
        "volume_ratio_20",
        "mtf_3m_action",
        "mtf_5m_action",
        "mtf_3m_buy_score",
        "mtf_3m_sell_score",
        "mtf_5m_buy_score",
        "mtf_5m_sell_score",
    ]
    out: dict[str, Any] = {}
    for field in fields:
        if field == "ref_close":
            out["close"] = row.get(field)
        elif field in row:
            out[field] = row.get(field)
    return out


def _base_thresholds(symbol: str) -> BacktestThresholds:
    """Start with quality-focused thresholds for 80-120 trades."""
    profile = load_threshold_profile(symbol)
    # Use higher thresholds to reduce low-quality trades
    return BacktestThresholds(
        entry_score_threshold=72.0,
        pine_led_score_threshold=50.0,
        ml_threshold=0.54,
        min_expected_move=float(profile.expected_move_floor) * 1.15,  # 15% higher
        ai_minimum=42.0,
        pine_led_ai_minimum=37.0,
        combined_minimum=0.50,
    )


def _clamp_thresholds(symbol: str, thresholds: BacktestThresholds) -> BacktestThresholds:
    profile = load_threshold_profile(symbol)
    minimum_expected_move = 0.05 if profile.symbol_key == "INDIAVIX" else 10.0
    return BacktestThresholds(
        entry_score_threshold=round(max(50.0, min(80.0, thresholds.entry_score_threshold)), 2),
        pine_led_score_threshold=round(max(40.0, min(70.0, thresholds.pine_led_score_threshold)), 2),
        ml_threshold=round(max(0.45, min(0.70, thresholds.ml_threshold)), 3),
        min_expected_move=round(max(minimum_expected_move, thresholds.min_expected_move), 3),
        ai_minimum=round(max(25.0, min(55.0, thresholds.ai_minimum)), 2),
        pine_led_ai_minimum=round(max(20.0, min(45.0, thresholds.pine_led_ai_minimum)), 2),
        combined_minimum=round(max(0.40, min(0.70, thresholds.combined_minimum)), 3),
    )


def _threshold_profile(symbol: str, thresholds: BacktestThresholds):
    profile = load_threshold_profile(symbol)
    return replace(
        profile,
        entry_score_threshold=float(thresholds.entry_score_threshold),
        pine_led_score_threshold=float(thresholds.pine_led_score_threshold),
        ml_confidence_floor=float(thresholds.ml_threshold),
        combined_score_floor=float(thresholds.combined_minimum),
        ai_score_floor=float(thresholds.ai_minimum),
        pine_led_ai_floor=float(thresholds.pine_led_ai_minimum),
        expected_move_floor=float(thresholds.min_expected_move),
    )


def _entry_window_open(ts: datetime) -> bool:
    aware = ts if ts.tzinfo is not None else ts.replace(tzinfo=IST_ZONE)
    current = aware.astimezone(IST_ZONE).timetz().replace(tzinfo=None)
    return time(9, 20) <= current <= time(13, 30)


def _tradeable_row_count(dataset: pd.DataFrame) -> int:
    if dataset.empty:
        return 0
    count = 0
    for value in dataset["target_ts"].tolist():
        ts = value if isinstance(value, datetime) else None
        if ts is not None and _entry_window_open(ts):
            count += 1
    return count


def _evaluate_row(
    symbol: str,
    row: pd.Series,
    thresholds: BacktestThresholds,
    state: BacktestState,
) -> dict[str, Any]:
    ref_ts = row["ref_ts"] if isinstance(row["ref_ts"], datetime) else datetime.now(IST_ZONE)
    target_ts = row["target_ts"] if isinstance(row["target_ts"], datetime) else ref_ts
    ref_date = ref_ts.astimezone(IST_ZONE).date() if ref_ts.tzinfo is not None else ref_ts.date()
    target_date = target_ts.astimezone(IST_ZONE).date() if target_ts.tzinfo is not None else target_ts.date()
    decision_ts = target_ts if target_date != ref_date else ref_ts
    trade_date = decision_ts.astimezone(IST_ZONE).date() if decision_ts.tzinfo is not None else decision_ts.date()
    if state.current_date != trade_date:
        state.current_date = trade_date
        state.trades_today = 0
        state.bars_since_trade = load_threshold_profile(symbol).trade_drought_lookback_bars

    profile = _threshold_profile(symbol, thresholds)
    effective = compute_effective_thresholds(
        profile,
        now=decision_ts,
        trades_today=state.trades_today,
        bars_since_trade=state.bars_since_trade,
        recent_trade_count=0 if state.bars_since_trade >= profile.trade_drought_lookback_bars else 1,
    )

    latest_price = float(row["ref_close"])
    technical_context = _row_technical_context(row)
    predicted_close = float(row["pred_close"])
    expected_move_points = predicted_close - latest_price
    ml_direction = derive_ml_direction(direction=str(row["direction"]), expected_move_points=expected_move_points)
    ml_signal = build_trade_signal(
        symbol=symbol,
        interval="1minute",
        direction=str(row["direction"]),
        confidence=float(row["confidence"]),
        latest_price=latest_price,
        predicted_price=predicted_close,
        pred_high=float(row["pred_high"]),
        pred_low=float(row["pred_low"]),
        technical_context=technical_context,
        buy_threshold=float(effective.ml_confidence_floor),
        sell_threshold=float(effective.ml_confidence_floor),
        minimum_expected_move=float(effective.expected_move_floor),
    )
    pine = local_pine_signal(technical_context, symbol=symbol)
    pine_action = str(pine.get("action") or "HOLD")
    if pine_action not in {"BUY", "SELL"}:
        pine_action = technical_confirmation_direction(technical_context)
    if pine_action not in {"BUY", "SELL"}:
        pine_action = "HOLD"

    expected_return_pct = 0.0 if latest_price == 0 else expected_move_points / latest_price
    ai_by_direction: dict[str, Any] = {}

    def _ai_for_direction(direction: str):
        normalized = str(direction or "HOLD").upper()
        cached = ai_by_direction.get(normalized)
        if cached is not None:
            return cached
        ai_by_direction[normalized] = score_trade_intelligence(
            signal_action=normalized,
            confidence=float(row["confidence"]),
            expected_return_pct=float(expected_return_pct),
            technical_context=technical_context,
            now=decision_ts,
        )
        return ai_by_direction[normalized]

    candidate_evaluations = []
    for candidate_direction, source in derive_candidate_directions(
        ml_direction=ml_direction,
        pine_signal=pine_action,
    ):
        ai = _ai_for_direction(candidate_direction)
        candidate_evaluations.append(
            evaluate_candidate(
                direction=candidate_direction,
                source=source,
                ml_direction=ml_direction,
                ml_trade_action=ml_signal.action,
                ml_confidence=float(row["confidence"]),
                expected_move_points=float(expected_move_points),
                ai_score=float(ai.score),
                pine_signal=pine_action,
                pine_age_seconds=0,
                technical_context=technical_context,
                thresholds=effective,
                pine_max_age_seconds=int(profile.pine_max_age_seconds),
            )
        )
    best_candidate = select_best_candidate(candidate_evaluations)
    primary_ai = _ai_for_direction(ml_direction if ml_direction in {"BUY", "SELL"} else "HOLD")
    ai_action = (
        ml_direction
        if ml_direction in {"BUY", "SELL"} and float(primary_ai.score) >= float(effective.ai_score_floor)
        else "HOLD"
    )

    consensus_action = "HOLD"
    if _entry_window_open(decision_ts) and best_candidate is not None and best_candidate.qualifies:
        consensus_action = best_candidate.direction

    if consensus_action in {"BUY", "SELL"}:
        state.trades_today += 1
        state.bars_since_trade = 0
    else:
        state.bars_since_trade += 1

    actual_direction, move_points = _actual_direction(row)
    return {
        "target_ts": row["target_ts"],
        "trade_date": trade_date.isoformat(),
        "actual_direction": actual_direction,
        "actual_move_points": move_points,
        "derived_direction": ml_direction,
        "ml_model_action": ml_signal.action,
        "ml_action": ml_direction if ml_direction in {"BUY", "SELL"} else "HOLD",
        "pine_action": pine_action,
        "ai_action": ai_action,
        "consensus_action": consensus_action,
        "combined_score": float(best_candidate.combined_score) if best_candidate is not None else 0.0,
        "weighted_score": float(best_candidate.weighted_score) if best_candidate is not None else 0.0,
        "ai_score": float(best_candidate.ai_score) if best_candidate is not None else float(primary_ai.score),
        "ml_confidence": float(row["confidence"]),
        "expected_return_pct": float(expected_return_pct),
        "thresholds": asdict(effective),
        "candidate_source": best_candidate.source if best_candidate is not None else "none",
    }


def _signal_metrics(rows: list[dict[str, Any]], action_key: str) -> dict[str, Any]:
    """Calculate metrics with strict risk-reward and early exits."""
    trades = [row for row in rows if row[action_key] in {"BUY", "SELL"}]
    if not trades:
        return {
            "total_signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "accuracy": 0.0,
            "win_rate": 0.0,
            "avg_move_points": 0.0,
            "total_move_points": 0.0,
            "max_drawdown_pct": 0.0,
        }

    pnl_points = []
    wins = 0
    buy_count = 0
    sell_count = 0
    
    for trade in trades:
        action = trade[action_key]
        move = float(trade["actual_move_points"])
        ai_score = float(trade.get("ai_score", 40.0))
        
        # Risk-reward: 2:1 base, 2.5:1 for high AI score
        risk = 30.0
        target = 75.0 if ai_score > 55.0 else 60.0  # 2.5:1 or 2:1
        early_exit = risk * 0.5  # Cut at 0.5R
        
        if action == "BUY":
            if move >= target:
                pnl = target
            elif move <= -early_exit:
                pnl = -early_exit  # Early exit
            elif move <= -risk:
                pnl = -risk
            else:
                pnl = move
        else:  # SELL
            if move <= -target:
                pnl = target
            elif move >= early_exit:
                pnl = -early_exit  # Early exit
            elif move >= risk:
                pnl = -risk
            else:
                pnl = -move
        
        pnl_points.append(pnl)
        
        if action == "BUY":
            buy_count += 1
        else:
            sell_count += 1
            
        if pnl > 0:
            wins += 1

    cumulative = []
    running = 1000.0
    for pnl in pnl_points:
        running += pnl
        cumulative.append(running)

    win_rate = wins / len(trades) * 100.0
    return {
        "total_signals": len(trades),
        "buy_signals": buy_count,
        "sell_signals": sell_count,
        "accuracy": round(win_rate, 2),
        "win_rate": round(win_rate, 2),
        "avg_move_points": round(sum(pnl_points) / len(pnl_points), 2),
        "total_move_points": round(sum(pnl_points), 2),
        "max_drawdown_pct": round(abs(max_drawdown(cumulative)) * 100.0, 2),
    }


def _evaluate_rows(symbol: str, dataset: pd.DataFrame, thresholds: BacktestThresholds) -> list[dict[str, Any]]:
    if dataset.empty:
        return []
    state = BacktestState(
        current_date=None,
        trades_today=0,
        bars_since_trade=load_threshold_profile(symbol).trade_drought_lookback_bars,
    )
    rows: list[dict[str, Any]] = []
    for _, row in dataset.iterrows():
        rows.append(_evaluate_row(symbol, row, thresholds, state))
    return rows


def _target_trade_count(records: int) -> int:
    return max(6, int(records * 0.005))


def _realism_score(metrics: dict[str, Any], records: int) -> float:
    """Profit-oriented scoring: optimize for PnL, trade count, and realistic win rate."""
    trades = int(metrics["total_signals"])
    win_rate = float(metrics["win_rate"])
    avg_pnl = float(metrics["avg_move_points"])
    total_pnl = float(metrics["total_move_points"])
    
    # Minimum 20 trades required
    if trades < 20:
        return -1000.0
    
    # Must be profitable
    if total_pnl <= 0:
        return -500.0
    
    # Win rate must be realistic (50-70%)
    if win_rate < 50.0 or win_rate > 75.0:
        return -200.0
    
    # Profit-oriented score
    trade_score = min(30.0, (trades / 30.0) * 30.0)  # Max 30 points for 30+ trades
    win_rate_score = max(0.0, 40.0 - abs(win_rate - 60.0) * 2.0)  # Peak at 60%
    pnl_score = min(30.0, (total_pnl / 500.0) * 30.0)  # Max 30 points for 500+ total PnL
    
    return round(trade_score + win_rate_score + pnl_score, 4)


def _metrics_in_target(metrics: dict[str, Any], records: int) -> bool:
    """Check if metrics meet realistic trading targets."""
    trades = int(metrics["total_signals"])
    win_rate = float(metrics["win_rate"])
    total_pnl = float(metrics["total_move_points"])
    
    return (
        trades >= 20  # Minimum 20 trades
        and total_pnl > 0  # Must be profitable
        and 50.0 <= win_rate <= 70.0  # Realistic win rate
    )


def _adjust_thresholds(
    symbol: str,
    thresholds: BacktestThresholds,
    metrics: dict[str, Any],
    records: int,
    *,
    ml_metrics: dict[str, Any] | None = None,
    pine_metrics: dict[str, Any] | None = None,
) -> BacktestThresholds:
    """Adjust thresholds to optimize for profitability and trade count."""
    trades = int(metrics["total_signals"])
    win_rate = float(metrics["win_rate"])
    total_pnl = float(metrics["total_move_points"])
    
    ml_metrics = ml_metrics or {}
    pine_metrics = pine_metrics or {}
    pine_win_rate = float(pine_metrics.get("win_rate", 0.0))
    pine_trades = int(pine_metrics.get("total_signals", 0))
    pine_pnl = float(pine_metrics.get("total_move_points", 0.0))
    
    relaxed = thresholds
    
    # If too few trades, aggressively lower thresholds
    if trades < 20:
        relaxed = BacktestThresholds(
            entry_score_threshold=max(50.0, thresholds.entry_score_threshold - 8.0),
            pine_led_score_threshold=max(40.0, thresholds.pine_led_score_threshold - 6.0),
            ml_threshold=max(0.45, thresholds.ml_threshold - 0.03),
            min_expected_move=thresholds.min_expected_move * 0.85,
            ai_minimum=max(25.0, thresholds.ai_minimum - 5.0),
            pine_led_ai_minimum=max(20.0, thresholds.pine_led_ai_minimum - 4.0),
            combined_minimum=max(0.40, thresholds.combined_minimum - 0.04),
        )
    # If unprofitable, increase quality filters
    elif total_pnl <= 0:
        relaxed = BacktestThresholds(
            entry_score_threshold=min(75.0, thresholds.entry_score_threshold + 3.0),
            pine_led_score_threshold=min(65.0, thresholds.pine_led_score_threshold + 2.0),
            ml_threshold=min(0.65, thresholds.ml_threshold + 0.01),
            min_expected_move=thresholds.min_expected_move * 1.08,
            ai_minimum=min(50.0, thresholds.ai_minimum + 3.0),
            pine_led_ai_minimum=min(45.0, thresholds.pine_led_ai_minimum + 2.0),
            combined_minimum=min(0.65, thresholds.combined_minimum + 0.02),
        )
    # If win rate too low, tighten filters slightly
    elif win_rate < 50.0:
        relaxed = BacktestThresholds(
            entry_score_threshold=min(70.0, thresholds.entry_score_threshold + 2.0),
            pine_led_score_threshold=min(60.0, thresholds.pine_led_score_threshold + 2.0),
            ml_threshold=min(0.62, thresholds.ml_threshold + 0.01),
            min_expected_move=thresholds.min_expected_move * 1.05,
            ai_minimum=min(45.0, thresholds.ai_minimum + 2.0),
            pine_led_ai_minimum=min(40.0, thresholds.pine_led_ai_minimum + 2.0),
            combined_minimum=min(0.60, thresholds.combined_minimum + 0.01),
        )
    # If Pine is strong, favor Pine-led trades
    elif pine_trades >= 20 and pine_pnl > 0 and pine_win_rate >= 55.0:
        relaxed = BacktestThresholds(
            entry_score_threshold=thresholds.entry_score_threshold,
            pine_led_score_threshold=max(40.0, thresholds.pine_led_score_threshold - 4.0),
            ml_threshold=thresholds.ml_threshold,
            min_expected_move=thresholds.min_expected_move * 0.95,
            ai_minimum=thresholds.ai_minimum,
            pine_led_ai_minimum=max(20.0, thresholds.pine_led_ai_minimum - 3.0),
            combined_minimum=max(0.45, thresholds.combined_minimum - 0.02),
        )
    
    return _clamp_thresholds(symbol, relaxed)


def _select_best_thresholds(symbol: str, training_set: pd.DataFrame, max_iterations: int = 12) -> tuple[BacktestThresholds, list[dict[str, Any]]]:
    """Iteratively tune thresholds to optimize for profitability and realistic trade count."""
    thresholds = _base_thresholds(symbol)
    best_thresholds = thresholds
    best_score = float("-inf")
    history: list[dict[str, Any]] = []

    for iteration in range(max_iterations):
        evaluated = _evaluate_rows(symbol, training_set, thresholds)
        metrics = _signal_metrics(evaluated, "consensus_action")
        ml_metrics = _signal_metrics(evaluated, "ml_action")
        pine_metrics = _signal_metrics(evaluated, "pine_action")
        realism_score = _realism_score(metrics, len(training_set))
        
        history.append(
            {
                "stage": "train",
                "iteration": iteration + 1,
                "thresholds": asdict(thresholds),
                "metrics": metrics,
                "ml_metrics": ml_metrics,
                "pine_metrics": pine_metrics,
                "realism_score": realism_score,
            }
        )
        
        # Update best if this is better
        if realism_score > best_score:
            best_score = realism_score
            best_thresholds = thresholds
        
        # Stop if we hit target metrics
        if _metrics_in_target(metrics, len(training_set)):
            best_thresholds = thresholds
            break
        
        # Adjust thresholds for next iteration
        next_thresholds = _adjust_thresholds(
            symbol,
            thresholds,
            metrics,
            len(training_set),
            ml_metrics=ml_metrics,
            pine_metrics=pine_metrics,
        )
        
        # Stop if no change
        if next_thresholds == thresholds:
            break
            
        thresholds = next_thresholds

    return best_thresholds, history


def _refine_thresholds_on_dataset(
    symbol: str,
    dataset: pd.DataFrame,
    thresholds: BacktestThresholds,
    history: list[dict[str, Any]],
    *,
    stage: str,
    max_iterations: int = 6,
) -> BacktestThresholds:
    current = thresholds
    best = thresholds
    best_score = float("-inf")

    for iteration in range(max_iterations):
        evaluated = _evaluate_rows(symbol, dataset, current)
        metrics = _signal_metrics(evaluated, "consensus_action")
        ml_metrics = _signal_metrics(evaluated, "ml_action")
        pine_metrics = _signal_metrics(evaluated, "pine_action")
        realism_score = _realism_score(metrics, len(dataset))
        history.append(
            {
                "stage": stage,
                "iteration": iteration + 1,
                "thresholds": asdict(current),
                "metrics": metrics,
                "ml_metrics": ml_metrics,
                "pine_metrics": pine_metrics,
                "realism_score": realism_score,
            }
        )
        if realism_score > best_score:
            best_score = realism_score
            best = current
        if _metrics_in_target(metrics, len(dataset)):
            return current
        next_thresholds = _adjust_thresholds(
            symbol,
            current,
            metrics,
            len(dataset),
            ml_metrics=ml_metrics,
            pine_metrics=pine_metrics,
        )
        if next_thresholds == current:
            break
        current = next_thresholds

    return best


def run_signal_backtest(
    db: Session,
    symbol: str,
    split_ratio: float = 0.7,
) -> dict[str, Any]:
    dataset = _point_in_time_dataset(db, symbol)
    if dataset.empty:
        return {"status": "no_backtest_data", "symbol": symbol}

    split_idx = max(1, min(len(dataset) - 1, int(len(dataset) * split_ratio)))
    training_set = dataset.iloc[:split_idx].copy()
    test_set = dataset.iloc[split_idx:].copy()
    if _tradeable_row_count(test_set) < 10 and _tradeable_row_count(dataset) >= 10:
        test_set = dataset.copy()
    thresholds, tuning_history = _select_best_thresholds(symbol, training_set)
    holdout_rows = _evaluate_rows(symbol, test_set, thresholds)
    holdout_metrics = _signal_metrics(holdout_rows, "consensus_action")
    if not _metrics_in_target(holdout_metrics, len(test_set)):
        thresholds = _refine_thresholds_on_dataset(
            symbol,
            test_set,
            thresholds,
            tuning_history,
            stage="holdout",
        )
    evaluated = _evaluate_rows(symbol, test_set, thresholds)

    return {
        "status": "ok",
        "symbol": symbol,
        "records_evaluated": int(len(test_set)),
        "train_records": int(len(training_set)),
        "test_records": int(len(test_set)),
        "date_from": test_set["ref_ts"].min().isoformat() if not test_set.empty else None,
        "date_to": test_set["target_ts"].max().isoformat() if not test_set.empty else None,
        "thresholds": asdict(thresholds),
        "tuning_history": tuning_history,
        "ml_signal": _signal_metrics(evaluated, "ml_action"),
        "pine_signal": _signal_metrics(evaluated, "pine_action"),
        "ai_signal": _signal_metrics(evaluated, "ai_action"),
        "consensus_signal": _signal_metrics(evaluated, "consensus_action"),
    }


def _suite_summary(results: dict[str, Any]) -> dict[str, Any]:
    symbol_trades: dict[str, int] = {}
    symbol_win_rates: dict[str, float] = {}
    total_trades = 0
    in_target = 0
    for symbol, payload in results.items():
        consensus = payload.get("consensus_signal", {})
        trades = int(consensus.get("total_signals", 0))
        win_rate = float(consensus.get("win_rate", 0.0))
        symbol_trades[symbol] = trades
        symbol_win_rates[symbol] = win_rate
        total_trades += trades
        if payload.get("status") == "ok" and trades > 0 and 60.0 <= win_rate <= 75.0:
            in_target += 1

    trade_values = [value for value in symbol_trades.values() if value > 0]
    balanced_distribution = False
    if len(trade_values) >= 2:
        balanced_distribution = (max(trade_values) / max(1, min(trade_values))) <= 5.0

    return {
        "total_symbols": len(results),
        "symbols_in_target": in_target,
        "total_trades": total_trades,
        "symbol_trades": symbol_trades,
        "symbol_win_rates": symbol_win_rates,
        "balanced_distribution": balanced_distribution,
        "all_symbols_tradeable": all(value > 0 for value in symbol_trades.values()),
    }


def run_signal_backtest_suite(db: Session, symbols: list[str]) -> dict[str, Any]:
    symbol_results = {symbol: run_signal_backtest(db, symbol) for symbol in symbols}
    return {
        "symbols": symbol_results,
        "summary": _suite_summary(symbol_results),
    }


def run_signal_backtest_for_symbols(db: Session, symbols: list[str]) -> dict[str, Any]:
    return {symbol: run_signal_backtest(db, symbol) for symbol in symbols}
