from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.db.connection import SessionLocal, get_db_session
from backend.execution_engine.live_service import (
    build_chart_payload,
    build_live_price_update,
    build_live_snapshot,
    compute_paper_portfolio_metrics,
    default_symbol,
    list_symbols,
    load_market_context,
    _load_option_quotes,
    _latest_option_quote_ts,
    _maybe_refresh_option_chain,
    resolve_underlying_key,
)
from backend.db.models import ExecutionPosition, OptionQuote
from sqlalchemy import and_, select
from backend.utils.app_state import apply_runtime_execution_settings, get_runtime_trading_mode
from backend.utils.config import get_settings
from backend.utils.symbols import symbol_value_filter

router = APIRouter(prefix="/api/live", tags=["live"])


def _runtime_settings(db: Session):
    settings = get_settings().model_copy()
    apply_runtime_execution_settings(db, settings)
    return settings


@router.get("/symbols")
def symbols(db: Session = Depends(get_db_session)) -> dict:
    return {"symbols": list_symbols(db, settings=_runtime_settings(db))}


@router.get("/snapshot")
def snapshot(
    symbol: str | None = Query(None),
    include_chart: bool = Query(False),
    include_static: bool = Query(True),
    include_option: bool = Query(True),
    db: Session = Depends(get_db_session),
) -> dict:
    try:
        settings = _runtime_settings(db)
        target = symbol or default_symbol(settings)
        return build_live_snapshot(
            db,
            symbol=target,
            settings=settings,
            include_chart=include_chart,
            include_static=include_static,
            include_option=include_option,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/chart")
def chart(
    symbol: str | None = Query(None),
    range_key: str | None = Query(None, alias="range"),
    interval: str | None = Query(None),
    db: Session = Depends(get_db_session),
) -> dict:
    settings = _runtime_settings(db)
    try:
        target = symbol or default_symbol(settings)
        return build_chart_payload(
            db,
            symbol=target,
            range_key=range_key or "1d",
            interval_key=interval,
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


@router.get("/option-chain")
def option_chain(
    symbol: str | None = Query(None),
    expiry: str | None = Query(None),
    refresh: bool = Query(False),
    strikes_each_side: int = Query(6, ge=1, le=20),
    db: Session = Depends(get_db_session),
) -> dict:
    """Get full option chain with live premiums, Greeks, OI, and volume."""
    from datetime import date as date_type, datetime
    from backend.prediction_engine.options_engine import (
        build_chain_rows,
        nearest_strike,
        next_weekly_expiries,
        strike_step_for_symbol,
        synthetic_option_chain,
    )
    from backend.utils.constants import IST_ZONE
    
    settings = _runtime_settings(db)
    try:
        target = symbol or default_symbol(settings)
        context = load_market_context(db, symbol=target, settings=settings)
        
        # Resolve expiry
        underlying_key = resolve_underlying_key(db, target, settings=settings)
        if expiry:
            try:
                expiry_date = date_type.fromisoformat(expiry)
            except ValueError:
                expiry_date = next_weekly_expiries(symbol=target, count=1)[0]
        else:
            expiry_date = next_weekly_expiries(symbol=target, count=1)[0]
        
        available_expiries = next_weekly_expiries(symbol=target, count=6)
        
        if refresh:
            _maybe_refresh_option_chain(
                db,
                symbol=target,
                underlying_key=underlying_key,
                expiry_date=expiry_date,
                settings=settings,
            )
        
        quotes = _load_option_quotes(db, symbol=target, expiry_date=expiry_date)
        chain_generated_at = _latest_option_quote_ts(db, target, expiry_date)
        chain_source = next((str(item.source or "db") for item in quotes if item.source), "synthetic")
        
        # If no quotes, generate synthetic
        strike_step = strike_step_for_symbol(target)
        if not quotes:
            quotes = synthetic_option_chain(
                symbol=target,
                underlying_price=context.latest_price,
                expiry_date=expiry_date,
                strike_step=strike_step,
            )
            chain_source = "synthetic"
            chain_generated_at = datetime.now(IST_ZONE)
        
        chain_rows = build_chain_rows(quotes)
        atm_strike = nearest_strike(context.latest_price, strike_step)
        
        # Filter around ATM for fast first paint.
        filtered_chain = [
            row for row in chain_rows
            if abs(row.get("strike", 0) - atm_strike) <= (strike_step * strikes_each_side)
        ]
        
        return {
            "symbol": target,
            "spot_price": context.latest_price,
            "atm_strike": atm_strike,
            "strike_step": strike_step,
            "expiry_date": expiry_date.isoformat(),
            "available_expiries": [exp.isoformat() for exp in available_expiries],
            "chain_source": chain_source,
            "chain_generated_at": chain_generated_at.isoformat() if chain_generated_at else None,
            "chain": filtered_chain,
            "total_strikes": len(filtered_chain),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.websocket("/ws")
async def websocket_stream(websocket: WebSocket, symbol: str | None = Query(None)) -> None:
    settings = get_settings().model_copy()
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
            apply_runtime_execution_settings(db, settings)
            initial = build_live_snapshot(
                db,
                symbol=target,
                settings=settings,
                include_static=False,
                include_chart=False,
                include_option=False,
            )
            await websocket.send_text(json.dumps({"type": "snapshot", "payload": initial}))
            last_snapshot_at = loop.time()
        finally:
            db.close()

        while True:
            now = loop.time()
            db = SessionLocal()
            try:
                apply_runtime_execution_settings(db, settings)
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
                        include_option=False,
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


@router.get("/option-contract-chart")
def option_contract_chart(
    symbol: str = Query(...),
    expiry: str = Query(...),
    strike: float = Query(...),
    option_type: str = Query(...),
    position_id: int | None = Query(None),
    db: Session = Depends(get_db_session),
) -> dict:
    from datetime import date as date_type

    try:
        expiry_date = date_type.fromisoformat(expiry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid expiry date") from exc

    rows = (
        db.execute(
            select(OptionQuote)
            .where(
                and_(
                    symbol_value_filter(OptionQuote.underlying_symbol, symbol),
                    OptionQuote.expiry_date == expiry_date,
                    OptionQuote.strike == float(strike),
                    OptionQuote.option_type == str(option_type).upper(),
                )
            )
            .order_by(OptionQuote.ts.asc())
            .limit(500)
        )
        .scalars()
        .all()
    )
    position = db.get(ExecutionPosition, position_id) if position_id is not None else None
    entry_price = float(position.entry_premium or position.entry_price) if position is not None else None
    quantity = int(position.quantity) if position is not None else 1
    points = []
    for row in rows:
        ltp = float(row.ltp)
        pnl = ((ltp - entry_price) * quantity) if entry_price is not None else None
        points.append(
            {
                "x": row.ts.isoformat() if row.ts is not None else None,
                "ltp": ltp,
                "pnl": round(pnl, 2) if pnl is not None else None,
                "volume": float(row.volume or 0.0),
            }
        )
    return {
        "symbol": symbol,
        "expiry_date": expiry_date.isoformat(),
        "strike": float(strike),
        "option_type": str(option_type).upper(),
        "entry_price": entry_price,
        "quantity": quantity,
        "points": points,
    }


@router.get("/dashboard-state")
def dashboard_state(db: Session = Depends(get_db_session)) -> dict:
    settings = _runtime_settings(db)
    return {
        "mode": get_runtime_trading_mode(db, settings=settings),
        "paper": compute_paper_portfolio_metrics(db, settings=settings),
    }
