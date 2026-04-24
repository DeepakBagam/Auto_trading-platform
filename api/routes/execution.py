from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from api.deps import get_db
from api.schemas import ExecutionReportResponse, ExecutionRunResponse
from db.models import ExecutionOrder, ExecutionPosition
from execution_engine.engine import IntradayOptionsExecutionEngine
from utils.config import get_settings
from utils.constants import IST_ZONE

router = APIRouter(prefix="/execution", tags=["execution"])


@lru_cache(maxsize=1)
def get_execution_engine() -> IntradayOptionsExecutionEngine:
    return IntradayOptionsExecutionEngine(settings=get_settings())


@router.post("/run-once", response_model=ExecutionRunResponse)
def run_once(db: Session = Depends(get_db)) -> ExecutionRunResponse:
    out = get_execution_engine().run_once(db)
    at_raw = out.get("at")
    at = datetime.fromisoformat(str(at_raw)) if at_raw else None
    return ExecutionRunResponse(status=str(out.get("status")), at=at, details=out)


@router.post("/emergency-exit", response_model=ExecutionRunResponse)
def emergency_exit(db: Session = Depends(get_db)) -> ExecutionRunResponse:
    out = get_execution_engine().emergency_exit_all(db)
    at_raw = out.get("at")
    at = datetime.fromisoformat(str(at_raw)) if at_raw else None
    return ExecutionRunResponse(status=str(out.get("status")), at=at, details=out)


@router.post("/positions/{position_id}/close")
def close_position(position_id: int, db: Session = Depends(get_db)) -> dict:
    return get_execution_engine().close_position_by_id(db, position_id)


@router.get("/report", response_model=ExecutionReportResponse)
def report(
    trade_date: date | None = Query(None),
    db: Session = Depends(get_db),
) -> ExecutionReportResponse:
    out = get_execution_engine().daily_report(db, trade_date=trade_date)
    return ExecutionReportResponse(
        trade_date=date.fromisoformat(str(out["trade_date"])),
        total_trades=int(out["total_trades"]),
        win_rate=float(out["win_rate"]),
        max_drawdown=float(out["max_drawdown"]),
        total_profit=float(out["total_profit"]),
        missed_signals=int(out["missed_signals"]),
        executed_signals=int(out["executed_signals"]),
        total_signal_events=int(out["total_signal_events"]),
    )


@router.get("/status")
def status(db: Session = Depends(get_db)) -> dict:
    engine = get_execution_engine()
    open_positions = (
        db.execute(select(ExecutionPosition).where(ExecutionPosition.status == "OPEN"))
        .scalars()
        .all()
    )
    today = datetime.now(IST_ZONE).date()
    today_closed = db.scalar(
        select(func.count(ExecutionPosition.id)).where(
            and_(ExecutionPosition.trade_date == today, ExecutionPosition.status == "CLOSED")
        )
    )
    today_orders = db.scalar(
        select(func.count(ExecutionOrder.id)).where(ExecutionOrder.trade_date == today)
    )
    return {
        "execution_enabled": bool(engine.settings.execution_enabled),
        "execution_mode": str(engine.settings.execution_mode),
        "execution_interval": "1minute",
        "open_positions": int(len(open_positions)),
        "today_closed_positions": int(today_closed or 0),
        "today_orders": int(today_orders or 0),
        "symbols": list(engine.settings.execution_symbol_list),
    }
