from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db.connection import SessionLocal, get_db_session
from execution_engine.live_service import (
    build_chart_payload,
    build_live_price_update,
    build_live_snapshot,
    default_symbol,
    list_symbols,
)
from utils.config import get_settings

router = APIRouter(prefix="/api/live", tags=["live"])


@router.get("/symbols")
def symbols(db: Session = Depends(get_db_session)) -> dict:
    return {"symbols": list_symbols(db, settings=get_settings())}


@router.get("/snapshot")
def snapshot(
    symbol: str | None = Query(None),
    db: Session = Depends(get_db_session),
) -> dict:
    try:
        target = symbol or default_symbol(get_settings())
        return build_live_snapshot(db, symbol=target, settings=get_settings())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/chart")
def chart(
    symbol: str | None = Query(None),
    range_key: str | None = Query(None, alias="range"),
    db: Session = Depends(get_db_session),
) -> dict:
    settings = get_settings()
    try:
        target = symbol or default_symbol(settings)
        return build_chart_payload(
            db,
            symbol=target,
            range_key=range_key or "1d",
            settings=settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/stream")
async def stream(symbol: str | None = Query(None)) -> StreamingResponse:
    settings = get_settings()
    target = symbol or default_symbol(settings)
    interval_seconds = max(0.25, float(getattr(settings, "ui_stream_interval_ms", 800)) / 1000.0)

    async def event_generator():
        while True:
            db = SessionLocal()
            try:
                payload = build_live_snapshot(db, symbol=target, settings=settings)
                yield f"event: snapshot\ndata: {json.dumps(payload)}\n\n"
            except ValueError as exc:
                yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            finally:
                db.close()
            await asyncio.sleep(interval_seconds)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.websocket("/ws")
async def websocket_stream(websocket: WebSocket, symbol: str | None = Query(None)) -> None:
    settings = get_settings()
    target = symbol or default_symbol(settings)
    tick_interval = max(0.10, float(getattr(settings, "ui_tick_interval_ms", 150)) / 1000.0)
    snapshot_interval = max(tick_interval, float(getattr(settings, "ui_stream_interval_ms", 800)) / 1000.0)
    await websocket.accept()

    last_digest: tuple | None = None
    last_snapshot_at = 0.0
    loop = asyncio.get_running_loop()

    try:
        db = SessionLocal()
        try:
            initial = build_live_snapshot(
                db,
                symbol=target,
                settings=settings,
                include_static=False,
                include_chart=False,
            )
            await websocket.send_text(json.dumps({"type": "snapshot", "payload": initial}))
            last_snapshot_at = loop.time()
        finally:
            db.close()

        while True:
            now = loop.time()
            db = SessionLocal()
            try:
                quick = build_live_price_update(db, symbol=target, settings=settings)
                candle = quick.get("candle") or {}
                price = quick.get("price") or {}
                digest = (
                    candle.get("x"),
                    candle.get("open"),
                    candle.get("high"),
                    candle.get("low"),
                    candle.get("close"),
                    price.get("last"),
                )
                if digest != last_digest:
                    await websocket.send_text(json.dumps({"type": "price", "payload": quick}))
                    last_digest = digest

                if (now - last_snapshot_at) >= snapshot_interval:
                    snapshot = build_live_snapshot(
                        db,
                        symbol=target,
                        settings=settings,
                        include_static=False,
                        include_chart=False,
                    )
                    await websocket.send_text(json.dumps({"type": "snapshot", "payload": snapshot}))
                    last_snapshot_at = now
            except ValueError as exc:
                await websocket.send_text(json.dumps({"type": "error", "payload": {"detail": str(exc)}}))
            finally:
                db.close()
            await asyncio.sleep(tick_interval)
    except WebSocketDisconnect:
        return
