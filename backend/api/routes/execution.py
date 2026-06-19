from __future__ import annotations

from datetime import date, datetime
from email.message import EmailMessage
from functools import lru_cache

import requests as _requests
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.api.market_stream_runtime import ensure_market_stream_started, get_market_stream_runtime_status
from backend.api.schemas import ExecutionReportResponse, ExecutionRunResponse
from backend.db.models import AuditLog, ExecutionOrder, ExecutionPosition
from backend.execution_engine.engine import IntradayOptionsExecutionEngine
from backend.execution_engine.live_service import _serialize_position, compute_paper_portfolio_metrics, list_symbols
from backend.utils.app_state import (
    apply_runtime_execution_settings,
    create_audit_log,
    get_runtime_execution_settings,
    get_runtime_trading_mode,
    mask_secret,
    reset_paper_account,
    set_runtime_execution_settings,
    set_runtime_trading_mode,
)
from backend.utils.config import Settings, get_settings, read_runtime_upstox_access_token
from backend.utils.constants import IST_ZONE
from backend.utils.notifications import send_email_message_result, smtp_ready

router = APIRouter(prefix="/execution", tags=["execution"])


class UpdateSLTargetRequest(BaseModel):
    position_id: int
    new_sl: float | None = None
    new_target: float | None = None


class TradingModeRequest(BaseModel):
    mode: str


class PaperResetRequest(BaseModel):
    starting_balance: float | None = None
    clear_open_positions: bool = True


class RuntimeSettingsRequest(BaseModel):
    execution_enabled: bool | None = None
    execution_symbols: list[str] | None = None
    execution_capital: float | None = None
    execution_per_trade_risk_pct: float | None = None
    execution_max_daily_loss_pct: float | None = None
    execution_max_simultaneous_trades: int | None = None
    execution_max_daily_trades: int | None = None
    execution_lot_size: int | None = None
    execution_premium_min: float | None = None
    execution_premium_max: float | None = None
    execution_stop_loss_pct: float | None = None
    tsl_activation_percent: float | None = None
    tsl_trail_percent: float | None = None
    tsl_immediate: bool | None = None
    target_profit_percent: float | None = None
    entry_window_start: str | None = None
    entry_window_end: str | None = None
    force_squareoff_time: str | None = None
    signal_min_score: float | None = None
    signal_cooldown_minutes: int | None = None
    signal_max_per_day: int | None = None
    signal_require_volume_confirmation: bool | None = None
    signal_min_volume_ratio: float | None = None
    signal_require_breakout: bool | None = None
    signal_rsi_buy_min: float | None = None
    signal_rsi_sell_max: float | None = None
    signal_vix_min: float | None = None
    signal_vix_max: float | None = None
    signal_atr_min_points: float | None = None
    signal_atr_max_points: float | None = None
    option_min_volume: float | None = None
    option_min_oi: float | None = None
    option_max_spread_pct: float | None = None
    upstox_access_token: str | None = None
    smtp_enabled: bool | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from_email: str | None = None
    smtp_to_emails: str | None = None
    smtp_use_tls: bool | None = None
    smtp_use_ssl: bool | None = None


@lru_cache(maxsize=1)
def get_execution_engine() -> IntradayOptionsExecutionEngine:
    return IntradayOptionsExecutionEngine(settings=get_settings())


def get_runtime_engine(db: Session) -> IntradayOptionsExecutionEngine:
    engine = get_execution_engine()
    apply_runtime_execution_settings(db, engine.settings)
    return engine


def _serializable_settings(settings: Settings, *, runtime: dict | None = None, token: str = "") -> dict:
    runtime = runtime or {}
    return {
        "execution_enabled": bool(settings.execution_enabled),
        "execution_symbols": list(settings.execution_symbol_list),
        "execution_capital": float(settings.execution_capital),
        "execution_per_trade_risk_pct": float(settings.execution_per_trade_risk_pct),
        "execution_max_daily_loss_pct": float(settings.execution_max_daily_loss_pct),
        "execution_max_simultaneous_trades": int(settings.execution_max_simultaneous_trades),
        "execution_max_daily_trades": int(settings.execution_max_daily_trades),
        "execution_lot_size": int(settings.execution_lot_size),
        "execution_premium_min": float(settings.execution_premium_min),
        "execution_premium_max": float(settings.execution_premium_max),
        "execution_stop_loss_pct": float(settings.execution_stop_loss_pct),
        "tsl_activation_percent": float(settings.tsl_activation_percent),
        "tsl_trail_percent": float(settings.tsl_trail_percent),
        "tsl_immediate": bool(settings.tsl_immediate),
        "target_profit_percent": float(settings.target_profit_percent),
        "entry_window_start": str(settings.entry_window_start),
        "entry_window_end": str(settings.entry_window_end),
        "force_squareoff_time": str(settings.force_squareoff_time),
        "signal_min_score": float(settings.signal_min_score),
        "signal_cooldown_minutes": int(settings.signal_cooldown_minutes),
        "signal_max_per_day": int(settings.signal_max_per_day),
        "signal_require_volume_confirmation": bool(settings.signal_require_volume_confirmation),
        "signal_min_volume_ratio": float(settings.signal_min_volume_ratio),
        "signal_require_breakout": bool(settings.signal_require_breakout),
        "signal_rsi_buy_min": float(settings.signal_rsi_buy_min),
        "signal_rsi_sell_max": float(settings.signal_rsi_sell_max),
        "signal_vix_min": float(settings.signal_vix_min),
        "signal_vix_max": float(settings.signal_vix_max),
        "signal_atr_min_points": float(settings.signal_atr_min_points),
        "signal_atr_max_points": float(settings.signal_atr_max_points),
        "option_min_volume": float(settings.option_min_volume),
        "option_min_oi": float(settings.option_min_oi),
        "option_max_spread_pct": float(settings.option_max_spread_pct),
        "upstox_token_present": bool(token),
        "upstox_token_masked": mask_secret(token),
        "smtp_enabled": bool(settings.smtp_enabled),
        "smtp_host": str(settings.smtp_host),
        "smtp_port": int(settings.smtp_port),
        "smtp_username": str(settings.smtp_username),
        "smtp_password_present": bool(runtime.get("smtp_password") or settings.smtp_password),
        "smtp_from_email": str(settings.smtp_from_email),
        "smtp_to_emails": str(settings.smtp_to_emails),
        "smtp_use_tls": bool(settings.smtp_use_tls),
        "smtp_use_ssl": bool(settings.smtp_use_ssl),
        "smtp_ready": smtp_ready(settings),
    }


def _settings_payload(db: Session) -> dict:
    settings = get_settings().model_copy()
    apply_runtime_execution_settings(db, settings)
    runtime = get_runtime_execution_settings(db)
    token = read_runtime_upstox_access_token(settings)
    defaults = Settings(_env_file=None)
    return {
        "settings": _serializable_settings(settings, runtime=runtime, token=token),
        "defaults": _serializable_settings(defaults),
        "available_symbols": list_symbols(db, settings=settings),
        "runtime_keys": sorted(runtime.keys()),
    }


def broker_health(engine: IntradayOptionsExecutionEngine) -> dict:
    settings = engine.settings
    token_present = bool(read_runtime_upstox_access_token(settings))
    out = {
        "broker": "upstox" if str(settings.execution_mode).lower() == "live" else "paper",
        "token_present": token_present,
        "status": "ok",
        "errors": [],
    }
    if str(settings.execution_mode).lower() != "live":
        return out
    if not token_present:
        out["status"] = "error"
        out["errors"] = [{"source": "token", "message": "UPSTOX_ACCESS_TOKEN is missing"}]
        return out
    data = engine.broker.get_portfolio()
    errors = data.get("errors") or []
    out["broker"] = data.get("broker") or out["broker"]
    out["status"] = data.get("status") or ("error" if errors else "ok")
    out["errors"] = errors
    out["funds_available"] = bool(data.get("funds"))
    return out


def _check_upstox_profile(settings: Settings, token: str) -> dict:
    if not token:
        return {
            "status": "error",
            "token_present": False,
            "detail": "UPSTOX_ACCESS_TOKEN is missing",
        }
    try:
        resp = _requests.get(
            f"{settings.upstox_base_url}/v2/user/profile",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=5,
        )
    except _requests.exceptions.Timeout:
        return {"status": "warn", "token_present": True, "detail": "Upstox API timed out"}
    except Exception as exc:
        return {"status": "error", "token_present": True, "detail": str(exc)}

    if resp.status_code == 200:
        profile = resp.json().get("data", {})
        return {
            "status": "ok",
            "token_present": True,
            "user_name": profile.get("user_name", ""),
            "email": profile.get("email", ""),
            "broker": profile.get("broker", ""),
        }
    if resp.status_code == 401:
        return {
            "status": "error",
            "token_present": True,
            "detail": "token expired or invalid (HTTP 401)",
        }
    return {
        "status": "warn",
        "token_present": True,
        "detail": f"unexpected HTTP {resp.status_code} from Upstox profile API",
    }


@router.get("/settings")
def get_runtime_settings(db: Session = Depends(get_db)) -> dict:
    engine = get_runtime_engine(db)
    return {**_settings_payload(db), "mode": get_runtime_trading_mode(db, settings=engine.settings), "broker": broker_health(engine)}


@router.put("/settings")
def update_runtime_settings(request: RuntimeSettingsRequest, db: Session = Depends(get_db)) -> dict:
    values = request.model_dump(exclude_unset=True)
    token_updated = bool(str(values.get("upstox_access_token") or "").strip())
    symbols = values.pop("execution_symbols", None)
    if symbols is not None:
        cleaned = []
        for symbol in symbols:
            text = str(symbol or "").strip()
            if text and text not in cleaned:
                cleaned.append(text)
        values["execution_symbols"] = ",".join(cleaned)
    for key, value in list(values.items()):
        if isinstance(value, str):
            values[key] = value.strip()
        if value is None:
            values.pop(key)
    if values.get("execution_capital") is not None and float(values["execution_capital"]) < 0:
        raise HTTPException(status_code=400, detail="Execution capital cannot be negative")
    if values.get("execution_premium_min") is not None and float(values["execution_premium_min"]) < 0:
        raise HTTPException(status_code=400, detail="Minimum premium cannot be negative")
    if values.get("execution_premium_max") is not None and float(values["execution_premium_max"]) <= 0:
        raise HTTPException(status_code=400, detail="Maximum premium must be positive")
    if (
        values.get("execution_premium_min") is not None
        and values.get("execution_premium_max") is not None
        and float(values["execution_premium_min"]) > float(values["execution_premium_max"])
    ):
        raise HTTPException(status_code=400, detail="Minimum premium cannot exceed maximum premium")
    for key in ("execution_per_trade_risk_pct", "execution_max_daily_loss_pct", "execution_stop_loss_pct", "tsl_activation_percent", "tsl_trail_percent", "target_profit_percent"):
        if values.get(key) is not None and float(values[key]) < 0:
            raise HTTPException(status_code=400, detail=f"{key} cannot be negative")
    for key in ("signal_min_score", "signal_min_volume_ratio", "signal_vix_min", "signal_vix_max", "signal_atr_min_points", "signal_atr_max_points", "option_min_volume", "option_min_oi", "option_max_spread_pct"):
        if values.get(key) is not None and float(values[key]) < 0:
            raise HTTPException(status_code=400, detail=f"{key} cannot be negative")
    if values.get("signal_max_per_day") is not None and int(values["signal_max_per_day"]) < 1:
        raise HTTPException(status_code=400, detail="Successful trades per symbol must be at least 1")
    if (
        values.get("signal_vix_min") is not None
        and values.get("signal_vix_max") is not None
        and float(values["signal_vix_min"]) > float(values["signal_vix_max"])
    ):
        raise HTTPException(status_code=400, detail="signal_vix_min cannot exceed signal_vix_max")
    if (
        values.get("signal_atr_min_points") is not None
        and values.get("signal_atr_max_points") is not None
        and float(values["signal_atr_min_points"]) > float(values["signal_atr_max_points"])
    ):
        raise HTTPException(status_code=400, detail="signal_atr_min_points cannot exceed signal_atr_max_points")
    set_runtime_execution_settings(db, values)
    engine = get_runtime_engine(db)
    engine.broker = engine._build_broker()
    if token_updated:
        ensure_market_stream_started(engine.settings)
    create_audit_log(
        db,
        action="runtime_settings_updated",
        resource="settings",
        status="INFO",
        message="Runtime execution settings updated",
        details={"updated_keys": sorted(key for key in values.keys() if key not in {"upstox_access_token", "smtp_password"})},
    )
    db.commit()
    return {**_settings_payload(db), "mode": get_runtime_trading_mode(db, settings=engine.settings), "broker": broker_health(engine)}


@router.post("/settings/test-upstox-token")
def test_upstox_token_settings(db: Session = Depends(get_db)) -> dict:
    settings = get_settings().model_copy()
    apply_runtime_execution_settings(db, settings)
    token = read_runtime_upstox_access_token(settings)
    broker_check = _check_upstox_profile(settings, token)
    stream_started = False
    if broker_check["status"] == "ok":
        stream_started = ensure_market_stream_started(settings)
    stream_status = get_market_stream_runtime_status(settings)
    websocket_open = bool(stream_status.get("running") or stream_status.get("thread_alive"))
    test_status = "ok" if broker_check["status"] == "ok" and websocket_open else (
        "warn" if broker_check["status"] == "ok" else "error"
    )
    token_test = {
        "status": test_status,
        "broker": broker_check,
        "market_stream": stream_status,
        "websocket_open": websocket_open,
        "start_attempted": broker_check["status"] == "ok",
        "started_now": stream_started,
    }
    create_audit_log(
        db,
        action="upstox_token_test",
        resource="settings",
        status=test_status.upper(),
        message="Upstox token tested from settings",
        details={
            "broker_status": broker_check["status"],
            "websocket_open": websocket_open,
            "started_now": stream_started,
        },
    )
    db.commit()
    engine = get_runtime_engine(db)
    return {
        **_settings_payload(db),
        "mode": get_runtime_trading_mode(db, settings=engine.settings),
        "broker": broker_health(engine),
        "token_test": token_test,
    }


@router.post("/settings/test-smtp")
def test_smtp_settings(db: Session = Depends(get_db)) -> dict:
    settings = get_settings().model_copy()
    apply_runtime_execution_settings(db, settings)
    if not smtp_ready(settings):
        raise HTTPException(status_code=400, detail="SMTP settings are incomplete")
    message = EmailMessage()
    message["Subject"] = f"[{str(settings.env).upper()}] Alpha Terminal SMTP test"
    message["From"] = settings.smtp_from_email
    message["To"] = ", ".join(settings.smtp_recipients)
    message.set_content("SMTP test from Alpha Terminal settings.")
    result = send_email_message_result(message, settings=settings)
    if not result.sent:
        detail = f"SMTP test failed: {result.detail}" if result.detail else "SMTP test failed"
        raise HTTPException(status_code=502, detail=detail)
    create_audit_log(
        db,
        action="smtp_test_sent",
        resource="settings",
        status="INFO",
        message="SMTP test email sent",
        details={"recipient_count": len(settings.smtp_recipients)},
    )
    db.commit()
    return {"status": "sent", "recipient_count": len(settings.smtp_recipients)}


@router.post("/run-once", response_model=ExecutionRunResponse)
def run_once(db: Session = Depends(get_db)) -> ExecutionRunResponse:
    out = get_runtime_engine(db).run_once(db)
    at_raw = out.get("at")
    at = datetime.fromisoformat(str(at_raw)) if at_raw else None
    return ExecutionRunResponse(status=str(out.get("status")), at=at, details=out)


@router.post("/emergency-exit", response_model=ExecutionRunResponse)
def emergency_exit(db: Session = Depends(get_db)) -> ExecutionRunResponse:
    out = get_runtime_engine(db).emergency_exit_all(db)
    at_raw = out.get("at")
    at = datetime.fromisoformat(str(at_raw)) if at_raw else None
    return ExecutionRunResponse(status=str(out.get("status")), at=at, details=out)


@router.post("/positions/{position_id}/close")
def close_position(position_id: int, db: Session = Depends(get_db)) -> dict:
    return get_runtime_engine(db).close_position_by_id(db, position_id)


@router.delete("/positions/{position_id}")
def delete_position(position_id: int, db: Session = Depends(get_db)) -> dict:
    position = db.get(ExecutionPosition, position_id)
    if position is None:
        raise HTTPException(status_code=404, detail="Position not found")
    if str(position.status).upper() in {"ENTRY_PENDING", "EXIT_SUBMITTING", "EXIT_PENDING"}:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a position while broker reconciliation is pending",
        )
    orders = (
        db.execute(select(ExecutionOrder).where(ExecutionOrder.position_id == position_id))
        .scalars()
        .all()
    )
    for order in orders:
        order.position_id = None
    db.delete(position)
    create_audit_log(
        db,
        action="position_deleted",
        resource="position",
        resource_id=str(position_id),
        status="WARN",
        message="Position row deleted from execution_positions",
        details={"symbol": position.symbol, "status": position.status, "orders_detached": len(orders)},
    )
    db.commit()
    return {"status": "deleted", "position_id": position_id, "orders_detached": len(orders)}


@router.get("/mode")
def get_mode(db: Session = Depends(get_db)) -> dict:
    engine = get_runtime_engine(db)
    mode = get_runtime_trading_mode(db, settings=engine.settings)
    engine.settings.execution_mode = mode
    engine.broker = engine._build_broker()
    return {"mode": mode, "broker": broker_health(engine)}


@router.post("/mode")
def set_mode(request: TradingModeRequest, db: Session = Depends(get_db)) -> dict:
    engine = get_runtime_engine(db)
    mode = set_runtime_trading_mode(db, request.mode)
    engine.settings.execution_mode = mode
    engine.broker = engine._build_broker()
    create_audit_log(
        db,
        action="trading_mode_changed",
        resource="execution",
        resource_id="runtime_mode",
        status="INFO",
        message=f"Trading mode changed to {mode}",
        details={"mode": mode},
    )
    db.commit()
    return {"status": "success", "mode": mode, "broker": broker_health(engine)}


@router.post("/paper/reset")
def reset_paper(request: PaperResetRequest, db: Session = Depends(get_db)) -> dict:
    engine = get_runtime_engine(db)
    starting_balance = float(request.starting_balance or engine.settings.execution_capital)
    if bool(engine.settings.paper_reset_requires_flat_positions) and request.clear_open_positions is False:
        open_paper_positions = [
            row for row in db.execute(select(ExecutionPosition).where(ExecutionPosition.status == "OPEN")).scalars().all()
            if str((row.metadata_json or {}).get("execution_mode") or "paper").lower() == "paper"
        ]
        if open_paper_positions:
            raise HTTPException(status_code=400, detail="Flat paper positions before reset or allow clear_open_positions")
    out = reset_paper_account(
        db,
        starting_balance=starting_balance,
        clear_open_positions=bool(request.clear_open_positions),
    )
    create_audit_log(
        db,
        action="paper_reset",
        resource="paper_account",
        resource_id="default",
        status="WARN",
        message="Paper account reset",
        details=out,
    )
    db.commit()
    return {"status": "success", **out}


@router.get("/report", response_model=ExecutionReportResponse)
def report(
    trade_date: date | None = Query(None),
    db: Session = Depends(get_db),
) -> ExecutionReportResponse:
    out = get_runtime_engine(db).daily_report(db, trade_date=trade_date)
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
    engine = get_runtime_engine(db)
    runtime_mode = get_runtime_trading_mode(db, settings=engine.settings)
    engine.settings.execution_mode = runtime_mode
    engine.broker = engine._build_broker()
    open_positions = (
        db.execute(
            select(ExecutionPosition).where(
                ExecutionPosition.status.in_(["OPEN", "ENTRY_PENDING", "EXIT_SUBMITTING", "EXIT_PENDING"])
            )
        )
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
        "execution_mode": runtime_mode,
        "execution_interval": "1minute",
        "broker": broker_health(engine),
        "open_positions": int(len(open_positions)),
        "today_closed_positions": int(today_closed or 0),
        "today_orders": int(today_orders or 0),
        "symbols": list(engine.settings.execution_symbol_list),
    }


@router.get("/portfolio")
def portfolio(db: Session = Depends(get_db)) -> dict:
    engine = get_runtime_engine(db)
    mode = get_runtime_trading_mode(db, settings=engine.settings)
    engine.settings.execution_mode = mode
    engine.broker = engine._build_broker()
    if mode == "live":
        data = engine.broker.get_portfolio()
        create_audit_log(
            db,
            action="portfolio_fetch",
            resource="broker",
            resource_id="upstox",
            status="INFO",
            message="Fetched live portfolio",
            details={"mode": mode},
        )
        db.commit()
        return {"mode": mode, **data}
    paper = compute_paper_portfolio_metrics(db, settings=engine.settings)
    positions = (
        db.execute(
            select(ExecutionPosition).where(
                ExecutionPosition.status.in_(["OPEN", "ENTRY_PENDING", "EXIT_SUBMITTING", "EXIT_PENDING"])
            )
        )
        .scalars()
        .all()
    )
    return {
        "mode": "paper",
        "broker": "paper",
        "summary": paper,
        "positions": [_serialize_position(row) for row in positions],
    }


@router.get("/trade-history")
def trade_history(
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    strategy: str | None = Query(None),
    db: Session = Depends(get_db),
) -> dict:
    query = select(ExecutionPosition).where(ExecutionPosition.status == "CLOSED")
    if date_from is not None:
        query = query.where(ExecutionPosition.trade_date >= date_from)
    if date_to is not None:
        query = query.where(ExecutionPosition.trade_date <= date_to)
    if strategy:
        query = query.where(ExecutionPosition.strategy_name == strategy)
    rows = db.execute(query.order_by(ExecutionPosition.closed_at.desc()).limit(300)).scalars().all()
    pnl_values = [float(row.realized_pnl or row.pnl_value or 0.0) for row in rows]
    return {
        "rows": [_serialize_position(row) for row in rows],
        "summary": {
            "trades": len(rows),
            "realized_pnl": round(sum(pnl_values), 2),
            "wins": sum(1 for value in pnl_values if value > 0),
            "losses": sum(1 for value in pnl_values if value <= 0),
        },
    }


@router.get("/strategy-performance")
def strategy_performance(db: Session = Depends(get_db)) -> dict:
    rows = db.execute(
        select(ExecutionPosition).where(ExecutionPosition.status == "CLOSED").order_by(ExecutionPosition.closed_at.asc())
    ).scalars().all()
    grouped: dict[str, dict] = {}
    for row in rows:
        key = str(row.strategy_name or "unknown")
        item = grouped.setdefault(
            key,
            {"strategy": key, "trades": 0, "wins": 0, "realized_pnl": 0.0, "equity": 0.0, "peak": 0.0, "max_drawdown": 0.0},
        )
        pnl = float(row.realized_pnl or row.pnl_value or 0.0)
        item["trades"] += 1
        item["wins"] += int(pnl > 0)
        item["realized_pnl"] += pnl
        item["equity"] += pnl
        item["peak"] = max(item["peak"], item["equity"])
        peak = item["peak"] or 1.0
        item["max_drawdown"] = max(item["max_drawdown"], round(((item["peak"] - item["equity"]) / peak) * 100.0, 2))
    return {
        "rows": [
            {
                **value,
                "win_rate": round((value["wins"] / value["trades"] * 100.0) if value["trades"] else 0.0, 2),
                "realized_pnl": round(value["realized_pnl"], 2),
            }
            for value in grouped.values()
        ]
    }


@router.get("/audit-logs")
def audit_logs(limit: int = Query(200, ge=1, le=1000), db: Session = Depends(get_db)) -> dict:
    rows = db.execute(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)).scalars().all()
    return {
        "rows": [
            {
                "id": row.id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "action": row.action,
                "resource": row.resource,
                "resource_id": row.resource_id,
                "status": row.status,
                "message": row.message,
                "details": row.details or {},
            }
            for row in rows
        ]
    }


@router.post("/update-sl-target")
def update_sl_target(request: UpdateSLTargetRequest, db: Session = Depends(get_db)) -> dict:
    """Update stop loss and/or target for an open position with risk validation."""
    position = db.get(ExecutionPosition, request.position_id)
    
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")
    
    if position.status != "OPEN":
        raise HTTPException(status_code=400, detail="Can only modify open positions")
    
    entry_premium = position.entry_premium or position.entry_price or 0
    current_premium = position.current_premium or position.current_price or entry_premium
    initial_sl = position.initial_sl or position.stop_loss
    
    # Validate new SL (can only tighten, not loosen)
    if request.new_sl is not None:
        if request.new_sl <= 0:
            raise HTTPException(status_code=400, detail="Stop loss must be positive")
        
        # For long positions: new SL must be >= current SL (tighten only)
        if initial_sl and request.new_sl < initial_sl:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot loosen stop loss. Current SL: {initial_sl}, New SL: {request.new_sl}"
            )
        
        # SL cannot be above current premium (would trigger immediately)
        if request.new_sl >= current_premium:
            raise HTTPException(
                status_code=400,
                detail=f"Stop loss ({request.new_sl}) cannot be >= current premium ({current_premium})"
            )
        
        position.current_sl = request.new_sl
        if not position.initial_sl:
            position.initial_sl = request.new_sl
    
    # Validate new target
    if request.new_target is not None:
        if request.new_target <= 0:
            raise HTTPException(status_code=400, detail="Target must be positive")
        
        # Target should be above entry (otherwise it's a loss target)
        if request.new_target <= entry_premium:
            raise HTTPException(
                status_code=400,
                detail=f"Target ({request.new_target}) should be above entry ({entry_premium})"
            )
        
        position.target_premium = request.new_target
        if not position.take_profit:
            position.take_profit = request.new_target
    
    # Update metadata
    metadata = position.metadata_json or {}
    modifications = metadata.get("sl_target_modifications", [])
    modifications.append({
        "timestamp": datetime.now(IST_ZONE).isoformat(),
        "old_sl": position.current_sl,
        "new_sl": request.new_sl,
        "old_target": position.target_premium,
        "new_target": request.new_target,
        "current_premium": current_premium,
    })
    metadata["sl_target_modifications"] = modifications
    position.metadata_json = metadata
    
    db.commit()
    db.refresh(position)
    
    return {
        "status": "success",
        "position_id": position.id,
        "current_sl": position.current_sl,
        "target_premium": position.target_premium,
        "message": "Stop loss and target updated successfully",
    }
