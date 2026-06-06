from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from backend.db.connection import SessionLocal, get_db_session
from backend.execution_engine.live_service import (
    build_live_price_update,
    default_symbol,
    list_symbols,
    load_candles_payload,
)
from backend.utils.app_state import apply_runtime_execution_settings
from backend.utils.config import get_settings

router = APIRouter(prefix="/api", tags=["candles"])
_ALERT_EVENTS: list[dict] = []


def _runtime_settings(db: Session):
    settings = get_settings().model_copy()
    apply_runtime_execution_settings(db, settings)
    return settings


@router.get("/symbols")
def symbols(db: Session = Depends(get_db_session)) -> dict:
    return {"symbols": list_symbols(db, settings=_runtime_settings(db))}


@router.get("/candles")
def candles(
    symbol: str | None = Query(None),
    interval: str = Query("1m"),
    before: str | None = Query(None),
    after: str | None = Query(None),
    limit: int = Query(500, ge=1, le=500000),
    db: Session = Depends(get_db_session),
) -> dict:
    settings = _runtime_settings(db)
    try:
        return load_candles_payload(
            db,
            symbol=symbol or default_symbol(settings),
            interval=interval,
            before=before,
            after=after,
            limit=limit,
            settings=settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/candles/history")
def candle_history(
    symbol: str | None = Query(None),
    interval: str = Query("1m"),
    before: str | None = Query(None),
    limit: int = Query(5000, ge=1, le=5000),
    db: Session = Depends(get_db_session),
) -> dict:
    settings = _runtime_settings(db)
    try:
        return load_candles_payload(
            db,
            symbol=symbol or default_symbol(settings),
            interval=interval,
            before=before,
            limit=limit,
            settings=settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/candles/latest")
def latest_candles(
    symbol: str | None = Query(None),
    interval: str = Query("1m"),
    limit: int = Query(1, ge=1, le=5000),
    db: Session = Depends(get_db_session),
) -> dict:
    settings = _runtime_settings(db)
    try:
        return load_candles_payload(
            db,
            symbol=symbol or default_symbol(settings),
            interval=interval,
            limit=limit,
            settings=settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.websocket("/ws/market")
async def market_websocket(websocket: WebSocket) -> None:
    settings = get_settings()
    await websocket.accept()
    symbol = default_symbol(settings)
    interval = "1minute"
    tick_interval = max(0.10, float(getattr(settings, "ui_tick_interval_ms", 150)) / 1000.0)

    try:
        try:
            message = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
            payload = json.loads(message)
            symbol = str(payload.get("symbol") or symbol)
            interval = str(payload.get("interval") or interval)
        except asyncio.TimeoutError:
            pass

        last_digest: tuple | None = None
        while True:
            db = SessionLocal()
            try:
                if interval in {"1m", "1min", "1minute"}:
                    quick = build_live_price_update(db, symbol=symbol, settings=settings)
                    candle = quick.get("candle") or {}
                    digest = (
                        candle.get("x"),
                        candle.get("open"),
                        candle.get("high"),
                        candle.get("low"),
                        candle.get("close"),
                        candle.get("volume"),
                    )
                    if digest != last_digest:
                        await websocket.send_text(json.dumps({"type": "candle", "payload": quick}))
                        last_digest = digest
                else:
                    latest = load_candles_payload(db, symbol=symbol, interval=interval, limit=1, settings=settings)
                    candle = (latest.get("candles") or [None])[-1]
                    digest = tuple(candle.items()) if isinstance(candle, dict) else None
                    if candle is not None and digest != last_digest:
                        await websocket.send_text(json.dumps({"type": "candle", "payload": latest}))
                        last_digest = digest
            finally:
                db.close()
            await asyncio.sleep(tick_interval)
    except WebSocketDisconnect:
        return
    except ValueError as exc:
        await websocket.send_text(json.dumps({"type": "error", "payload": {"detail": str(exc)}}))


@router.post("/alerts/event")
async def alert_event(request: Request) -> dict:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid alert payload")
    _ALERT_EVENTS.insert(0, payload)
    del _ALERT_EVENTS[200:]
    return {"status": "accepted", "count": len(_ALERT_EVENTS)}


@router.get("/alerts/events")
def alert_events() -> dict:
    return {"events": list(_ALERT_EVENTS)}
