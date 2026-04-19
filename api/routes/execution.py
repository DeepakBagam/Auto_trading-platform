from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from api.deps import get_db
from api.schemas import ExecutionReportResponse, ExecutionRunResponse, ExternalSignalRequest
from db.models import ExecutionExternalSignal, ExecutionOrder, ExecutionPosition, RawCandle
from execution_engine.engine import IntradayOptionsExecutionEngine
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.symbols import instrument_key_filter, symbol_value_filter

router = APIRouter(prefix="/execution", tags=["execution"])


@lru_cache(maxsize=1)
def get_execution_engine() -> IntradayOptionsExecutionEngine:
    return IntradayOptionsExecutionEngine(settings=get_settings())


def _ensure_ist(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.astimezone(IST_ZONE) if dt.tzinfo is not None else dt.replace(tzinfo=IST_ZONE)


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


@router.post("/webhook/pine", response_model=ExecutionRunResponse)
def pine_webhook(
    payload: ExternalSignalRequest,
    db: Session = Depends(get_db),
    x_webhook_secret: str | None = Header(default=None),
) -> ExecutionRunResponse:
    settings = get_settings()
    if not settings.execution_accept_external_webhook:
        raise HTTPException(status_code=403, detail="External webhook execution is disabled")
    if settings.pine_webhook_secret.strip():
        if str(x_webhook_secret or "").strip() != settings.pine_webhook_secret.strip():
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    action_raw = str(payload.signal_action).strip().upper()
    action = {"LONG": "BUY", "SHORT": "SELL"}.get(action_raw, action_raw)
    if action not in {"BUY", "SELL"}:
        raise HTTPException(status_code=422, detail="signal_action must be BUY/SELL/LONG/SHORT")

    latest_candle_ts = db.scalar(
        select(func.max(RawCandle.ts)).where(
            and_(
                instrument_key_filter(RawCandle.instrument_key, payload.symbol),
                RawCandle.interval == "1minute",
            )
        )
    )
    bucket_anchor = (
        latest_candle_ts if latest_candle_ts is not None and latest_candle_ts.tzinfo is not None
        else (latest_candle_ts.replace(tzinfo=IST_ZONE) if latest_candle_ts is not None else datetime.now(IST_ZONE))
    )
    bucket_start = bucket_anchor.replace(second=0, microsecond=0)
    bucket_end = bucket_start + timedelta(minutes=1)
    duplicate = db.scalar(
        select(ExecutionExternalSignal.id)
        .where(
            and_(
                symbol_value_filter(ExecutionExternalSignal.symbol, payload.symbol),
                ExecutionExternalSignal.interval == "1minute",
                ExecutionExternalSignal.source == str(payload.source or "pine"),
                ExecutionExternalSignal.signal_action == action,
                ExecutionExternalSignal.signal_ts >= bucket_start,
                ExecutionExternalSignal.signal_ts < bucket_end,
            )
        )
        .limit(1)
    )
    if duplicate is not None:
        return ExecutionRunResponse(
            status="duplicate_ignored",
            details={
                "source": str(payload.source or "pine"),
                "symbol": payload.symbol,
                "interval": "1minute",
                "signal_action": action,
                "bucket_start": _ensure_ist(bucket_start).isoformat(),
                "bucket_end": _ensure_ist(bucket_end).isoformat(),
            },
        )
    signal_row = ExecutionExternalSignal(
        source=str(payload.source or "pine"),
        symbol=str(payload.symbol),
        interval="1minute",
        signal_action=action,
        signal_ts=bucket_anchor if latest_candle_ts is not None else datetime.now(IST_ZONE),
        confidence=float(payload.confidence),
        metadata_json={
            **(payload.metadata_json or {}),
            "candle_ts": _ensure_ist(latest_candle_ts).isoformat() if latest_candle_ts is not None else None,
            "received_at": datetime.now(IST_ZONE).isoformat(),
        },
    )
    db.add(signal_row)
    signal_row.processed = False
    db.commit()
    return ExecutionRunResponse(
        status="accepted",
        details={
            "source": signal_row.source,
            "symbol": signal_row.symbol,
            "interval": signal_row.interval,
            "signal_action": signal_row.signal_action,
            "signal_ts": _ensure_ist(signal_row.signal_ts).isoformat(),
            "processed": signal_row.processed,
            "candle_ts": signal_row.metadata_json.get("candle_ts"),
        },
    )


@router.post("/webhook/pine/v2", response_model=ExecutionRunResponse)
def pine_webhook_v2(
    payload: ExternalSignalRequest,
    db: Session = Depends(get_db),
    x_webhook_secret: str | None = Header(default=None),
) -> ExecutionRunResponse:
    return pine_webhook(payload=payload, db=db, x_webhook_secret=x_webhook_secret)
