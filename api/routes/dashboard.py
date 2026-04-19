from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime
from functools import lru_cache

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from api.deps import get_db
from api.routes.signal import latest_candle_ts_for_symbol
from backtesting.engine import run_pine_signal_backtest
from db.models import DailySummary, ExecutionPosition, SignalLog
from db.models import ExecutionOrder
from execution_engine.engine import IntradayOptionsExecutionEngine
from prediction_engine.consensus_engine import get_consensus_signal
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.symbols import symbol_value_filter

router = APIRouter(prefix="/api", tags=["dashboard-api"])


@lru_cache(maxsize=1)
def get_execution_engine() -> IntradayOptionsExecutionEngine:
    return IntradayOptionsExecutionEngine(settings=get_settings())


def _default_symbol() -> str:
    settings = get_settings()
    return settings.execution_symbol_list[0] if settings.execution_symbol_list else "Nifty 50"


def _ensure_ist(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.astimezone(IST_ZONE) if dt.tzinfo is not None else dt.replace(tzinfo=IST_ZONE)


def _serialize_position(row: ExecutionPosition) -> dict:
    metadata = row.metadata_json or {}
    return {
        "order_id": row.entry_order_id,
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
        "tsl_active": bool(row.tsl_active),
        "target_premium": row.target_premium or row.take_profit,
        "peak_premium": row.peak_premium,
        "status": row.status,
        "exit_reason": row.exit_reason,
        "ml_confidence": row.ml_confidence,
        "ai_score": row.ai_score,
        "pine_signal": row.pine_signal,
        "consensus_reason": row.consensus_reason,
        "premium_history": metadata.get("premium_history") or [],
    }


def _serialize_order(row: ExecutionOrder) -> dict:
    return {
        "id": row.id,
        "position_id": row.position_id,
        "trade_date": row.trade_date.isoformat() if row.trade_date else None,
        "symbol": row.symbol,
        "strike_price": row.strike_price,
        "option_type": row.option_type,
        "expiry_date": row.expiry_date.isoformat() if row.expiry_date else None,
        "order_kind": row.order_kind,
        "side": row.side,
        "quantity": row.quantity,
        "price": row.price,
        "trigger_price": row.trigger_price,
        "entry_premium": row.entry_premium,
        "current_sl": row.current_sl,
        "target_premium": row.target_premium,
        "status": row.status,
        "broker_name": row.broker_name,
        "broker_order_id": row.broker_order_id,
        "exit_reason": row.exit_reason,
        "realized_pnl": row.realized_pnl,
        "unrealized_pnl": row.unrealized_pnl,
        "consensus_reason": row.consensus_reason,
        "created_at": _ensure_ist(row.created_at).isoformat() if row.created_at else None,
    }


def _stats_payload(db: Session) -> dict:
    settings = get_settings()
    today = datetime.now(IST_ZONE).date()
    today_summary = db.get(DailySummary, today)

    open_positions = (
        db.execute(select(ExecutionPosition).where(ExecutionPosition.status == "OPEN"))
        .scalars()
        .all()
    )
    total_unrealized = sum(float(row.unrealized_pnl or 0.0) for row in open_positions)

    closed = (
        db.execute(
            select(ExecutionPosition)
            .where(ExecutionPosition.status == "CLOSED")
            .order_by(ExecutionPosition.closed_at.asc())
        )
        .scalars()
        .all()
    )
    gross_profit = sum(max(0.0, float(row.realized_pnl or row.pnl_value or 0.0)) for row in closed)
    gross_loss = sum(abs(min(0.0, float(row.realized_pnl or row.pnl_value or 0.0))) for row in closed)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    durations = []
    equity = float(settings.execution_capital)
    peak = equity
    max_drawdown_percent = 0.0
    for row in closed:
        if row.opened_at and row.closed_at:
            durations.append((row.closed_at - row.opened_at).total_seconds() / 60.0)
        equity += float(row.realized_pnl or row.pnl_value or 0.0)
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown_percent = max(max_drawdown_percent, ((peak - equity) / peak) * 100.0)

    return {
        "win_rate": float(today_summary.win_rate if today_summary is not None else 0.0),
        "profit_factor": round(float(profit_factor), 2),
        "avg_duration_minutes": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "max_drawdown_percent": round(float(max_drawdown_percent), 2),
        "total_pnl_today": float(today_summary.total_pnl if today_summary is not None else 0.0),
        "open_positions_count": len(open_positions),
        "open_positions_unrealized_pnl": round(total_unrealized, 2),
        "wins_today": int(today_summary.winning_trades if today_summary is not None else 0),
        "total_trades_today": int(today_summary.total_trades if today_summary is not None else 0),
    }


@router.get("/dashboard/stats")
def dashboard_stats(db: Session = Depends(get_db)) -> dict:
    return _stats_payload(db)


@router.get("/positions/open")
def positions_open(db: Session = Depends(get_db)) -> list[dict]:
    rows = (
        db.execute(
            select(ExecutionPosition)
            .where(ExecutionPosition.status == "OPEN")
            .order_by(ExecutionPosition.opened_at.desc())
        )
        .scalars()
        .all()
    )
    return [_serialize_position(row) for row in rows]


@router.get("/positions/history")
def positions_history(
    trade_date: date | None = Query(None, alias="date"),
    db: Session = Depends(get_db),
) -> list[dict]:
    query = select(ExecutionPosition).where(ExecutionPosition.status == "CLOSED")
    if trade_date is not None:
        query = query.where(ExecutionPosition.trade_date == trade_date)
    rows = db.execute(query.order_by(ExecutionPosition.closed_at.desc())).scalars().all()
    return [_serialize_position(row) for row in rows]


@router.get("/orders/recent")
def orders_recent(
    symbol: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[dict]:
    query = select(ExecutionOrder)
    if symbol:
        query = query.where(symbol_value_filter(ExecutionOrder.symbol, symbol))
    rows = db.execute(query.order_by(ExecutionOrder.created_at.desc()).limit(limit)).scalars().all()
    return [_serialize_order(row) for row in rows]


@router.post("/positions/{position_id}/close")
def close_position(position_id: int, db: Session = Depends(get_db)) -> dict:
    return get_execution_engine().close_position_by_id(db, position_id)


@router.get("/calendar/monthly")
def calendar_monthly(
    year: int = Query(..., ge=2000, le=2100),
    month: int = Query(..., ge=1, le=12),
    db: Session = Depends(get_db),
) -> list[dict]:
    start = date(year, month, 1)
    end = date(year, month, monthrange(year, month)[1])
    rows = (
        db.execute(
            select(DailySummary)
            .where(and_(DailySummary.date >= start, DailySummary.date <= end))
            .order_by(DailySummary.date.asc())
        )
        .scalars()
        .all()
    )
    return [
        {
            "date": row.date.isoformat(),
            "total_pnl": row.total_pnl,
            "total_trades": row.total_trades,
            "win_rate": row.win_rate,
            "is_green": row.is_green,
        }
        for row in rows
    ]


@router.get("/calendar/day/{trade_date}")
def calendar_day(trade_date: date, db: Session = Depends(get_db)) -> dict:
    summary = db.get(DailySummary, trade_date)
    trades = (
        db.execute(
            select(ExecutionPosition)
            .where(ExecutionPosition.trade_date == trade_date)
            .order_by(ExecutionPosition.opened_at.asc())
        )
        .scalars()
        .all()
    )
    return {
        "date": trade_date.isoformat(),
        "daily_summary": {
            "date": trade_date.isoformat(),
            "total_trades": int(summary.total_trades if summary is not None else 0),
            "winning_trades": int(summary.winning_trades if summary is not None else 0),
            "losing_trades": int(summary.losing_trades if summary is not None else 0),
            "total_pnl": float(summary.total_pnl if summary is not None else 0.0),
            "win_rate": float(summary.win_rate if summary is not None else 0.0),
            "is_green": bool(summary.is_green if summary is not None else False),
        },
        "trades": [_serialize_position(row) for row in trades],
    }


@router.get("/signals/log")
def signals_log(
    symbol: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[dict]:
    symbol = symbol or _default_symbol()
    rows = (
        db.execute(
            select(SignalLog)
            .where(symbol_value_filter(SignalLog.symbol, symbol))
            .order_by(SignalLog.timestamp.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return [
        {
            "timestamp": _ensure_ist(row.timestamp).isoformat(),
            "symbol": row.symbol,
            "interval": row.interval,
            "ml_signal": row.ml_signal,
            "ml_confidence": row.ml_confidence,
            "pine_signal": row.pine_signal,
            "pine_age_seconds": row.pine_age_seconds,
            "ai_score": row.ai_score,
            "news_sentiment": row.news_sentiment,
            "combined_score": row.combined_score,
            "consensus": row.consensus,
            "skip_reason": row.skip_reason,
            "trade_placed": row.trade_placed,
            "details": row.details,
        }
        for row in rows
    ]


@router.get("/consensus/live")
def consensus_live(
    symbol: str | None = Query(None),
    db: Session = Depends(get_db),
) -> dict:
    symbol = symbol or _default_symbol()
    reference_ts = latest_candle_ts_for_symbol(db, symbol, "1minute")
    result = get_consensus_signal(
        db,
        symbol=symbol,
        interval="1minute",
        now=_ensure_ist(reference_ts) if reference_ts is not None else None,
        settings=get_settings(),
        persist=False,
    )
    return {
        "timestamp": _ensure_ist(result.timestamp).isoformat(),
        "market_data_timestamp": _ensure_ist(reference_ts).isoformat() if reference_ts is not None else None,
        "ml_signal": result.ml_signal,
        "ml_confidence": result.ml_confidence,
        "ml_expected_move": result.ml_expected_move,
        "pine_signal": result.pine_signal,
        "pine_age_seconds": result.pine_age_seconds,
        "ai_score": result.ai_score,
        "news_sentiment": result.news_sentiment,
        "combined_score": result.combined_score,
        "consensus": result.consensus,
        "skip_reason": result.skip_reason,
        "details": result.details,
        "ml_reasons": result.ml_reasons,
        "ai_reasons": result.ai_reasons,
    }


@router.get("/backtest/pine")
def backtest_pine(
    symbol: str | None = Query(None),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    initial_capital: float = Query(100000.0, ge=1000.0),
    db: Session = Depends(get_db),
) -> dict:
    symbol = symbol or _default_symbol()
    return run_pine_signal_backtest(
        db,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )


@router.post("/emergency-stop")
def emergency_stop(db: Session = Depends(get_db)) -> dict:
    return get_execution_engine().emergency_exit_all(db)
