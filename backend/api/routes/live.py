from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.data_layer.collectors.upstox_collector import UpstoxCollector
from backend.db.connection import SessionLocal, get_db_session
from backend.data_layer.collectors.upstox_option_chain import UpstoxOptionChainCollector
from backend.execution_engine.live_service import (
    _build_pine_chart_overlay,
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
from backend.utils.config import get_settings, read_runtime_upstox_access_token
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
    db: Session = Depends(get_db_session),
) -> dict:
    settings = _runtime_settings(db)
    try:
        target = symbol or default_symbol(settings)
        return build_chart_payload(
            db,
            symbol=target,
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
    from backend.execution_engine.options_engine import (
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
        
        # Resolve expiry from broker contracts when available. SENSEX weekly
        # expiries do not always match the generic fallback calendar.
        underlying_key = resolve_underlying_key(db, target, settings=settings)
        available_expiries = []
        if underlying_key and settings.has_market_data_access:
            try:
                available_expiries = UpstoxOptionChainCollector(settings).list_expiries(underlying_key, max_items=6)
            except Exception:
                available_expiries = []
        if not available_expiries:
            available_expiries = next_weekly_expiries(symbol=target, count=6)

        if expiry:
            try:
                expiry_date = date_type.fromisoformat(expiry)
            except ValueError:
                expiry_date = available_expiries[0]
            if expiry_date not in available_expiries:
                expiry_date = available_expiries[0]
        else:
            expiry_date = available_expiries[0]
        
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
    limit: int = Query(2000, ge=50, le=10000),
    include_previous: bool = Query(False),
    refresh: bool = Query(False),
    include_live: bool = Query(False),
    ltp: float | None = Query(None),
    entry_price: float | None = Query(None),
    stop_loss: float | None = Query(None),
    take_profit: float | None = Query(None),
    trailing_stop_loss: float | None = Query(None),
    db: Session = Depends(get_db_session),
) -> dict:
    from datetime import date as date_type

    try:
        expiry_date = date_type.fromisoformat(expiry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid expiry date") from exc

    settings = _runtime_settings(db)
    normalized_type = str(option_type).upper()
    base_filters = (
        symbol_value_filter(OptionQuote.underlying_symbol, symbol),
        OptionQuote.strike == float(strike),
        OptionQuote.option_type == normalized_type,
    )
    underlying_key = resolve_underlying_key(db, symbol, settings=settings)
    if refresh:
        _maybe_refresh_option_chain(
            db,
            symbol=symbol,
            underlying_key=underlying_key,
            expiry_date=expiry_date,
            settings=settings,
        )

    rows = list(
        db.execute(
            select(OptionQuote)
            .where(
                and_(
                    *base_filters,
                    OptionQuote.expiry_date == expiry_date,
                )
            )
            .order_by(OptionQuote.ts.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    rows.reverse()
    instrument_key = next((row.instrument_key for row in reversed(rows) if row.instrument_key), None)

    previous_expiry = None
    previous_rows = []
    if include_previous:
        previous_expiry = db.scalar(
            select(OptionQuote.expiry_date)
            .where(and_(*base_filters, OptionQuote.expiry_date < expiry_date))
            .distinct()
            .order_by(OptionQuote.expiry_date.desc())
            .limit(1)
        )
        if previous_expiry is not None:
            previous_rows = list(
                db.execute(
                    select(OptionQuote)
                    .where(and_(*base_filters, OptionQuote.expiry_date == previous_expiry))
                    .order_by(OptionQuote.ts.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            previous_rows.reverse()

    position = db.get(ExecutionPosition, position_id) if position_id is not None else None
    entry = float(position.entry_premium or position.entry_price) if position is not None else entry_price
    stop = float(position.stop_loss or position.initial_sl or 0.0) if position is not None else stop_loss
    target = float(position.take_profit or position.target_premium or 0.0) if position is not None else take_profit
    trailing_stop = float(position.current_sl or position.trailing_stop or 0.0) if position is not None else trailing_stop_loss
    quantity = int(position.quantity) if position is not None else 1

    def serialize_points(quote_rows: list[OptionQuote], *, with_pnl: bool) -> list[dict]:
        out = []
        for row in quote_rows:
            row_ltp = float(row.ltp)
            pnl = ((row_ltp - entry) * quantity) if with_pnl and entry is not None else None
            out.append(
                {
                    "x": row.ts.isoformat() if row.ts is not None else None,
                    "ltp": row_ltp,
                    "pnl": round(pnl, 2) if pnl is not None else None,
                    "volume": float(row.volume or 0.0),
                }
            )
        return out

    def quote_chart(quote_rows: list[OptionQuote], *, label: str, expiry_value) -> dict:
        frame_rows = [
            {
                "ts": row.ts,
                "ltp": float(row.ltp),
                "volume": float(row.volume or 0.0),
            }
            for row in quote_rows
            if row.ts is not None
        ]
        if not frame_rows:
            return {
                "label": label,
                "expiry_date": expiry_value.isoformat() if expiry_value is not None else None,
                "candles": [],
                "markers": [],
                "pine_levels": [],
            }
        import pandas as pd

        frame = pd.DataFrame(frame_rows)
        frame["ts"] = pd.to_datetime(frame["ts"])
        candles_frame = (
            frame.set_index("ts")
            .resample("1min", label="right", closed="right")
            .agg({"ltp": ["first", "max", "min", "last"], "volume": "sum"})
            .dropna()
        )
        candles_frame.columns = ["open", "high", "low", "close", "volume"]
        candles_frame = candles_frame.reset_index()
        candles = [
            {
                "x": row.ts.isoformat() if row.ts is not None else None,
                "open": round(float(row.open), 2),
                "high": round(float(row.high), 2),
                "low": round(float(row.low), 2),
                "close": round(float(row.close), 2),
                "volume": round(float(row.volume or 0.0), 2),
            }
            for row in candles_frame.itertuples(index=False)
        ]
        overlay = _build_pine_chart_overlay(
            [
                {
                    "ts": item["x"],
                    "open": item["open"],
                    "high": item["high"],
                    "low": item["low"],
                    "close": item["close"],
                    "volume": item["volume"],
                }
                for item in candles
            ],
            interval="1minute",
            settings=settings,
            range_key="all",
        )
        return {
            "label": label,
            "expiry_date": expiry_value.isoformat() if expiry_value is not None else None,
            "source": "option_quote_snapshots",
            "candles": candles,
            "markers": list(overlay.get("markers") or []),
            "pine_levels": list(overlay.get("levels") or []),
        }

    def candle_record_chart(records, *, label: str, expiry_value) -> dict:
        candles = [
            {
                "x": row.ts.isoformat() if row.ts is not None else None,
                "open": round(float(row.open), 2),
                "high": round(float(row.high), 2),
                "low": round(float(row.low), 2),
                "close": round(float(row.close), 2),
                "volume": round(float(row.volume or 0.0), 2),
            }
            for row in sorted(records, key=lambda item: item.ts)[-limit:]
            if row.ts is not None
        ]
        overlay = _build_pine_chart_overlay(
            [
                {
                    "ts": item["x"],
                    "open": item["open"],
                    "high": item["high"],
                    "low": item["low"],
                    "close": item["close"],
                    "volume": item["volume"],
                }
                for item in candles
            ],
            interval="1minute",
            settings=settings,
            range_key="all",
        )
        return {
            "label": label,
            "expiry_date": expiry_value.isoformat() if expiry_value is not None else None,
            "source": "upstox_intraday_candles",
            "candles": candles,
            "markers": list(overlay.get("markers") or []),
            "pine_levels": list(overlay.get("levels") or []),
        }

    intraday_candles = []
    if include_live and instrument_key:
        try:
            collector = UpstoxCollector()
            access_token = read_runtime_upstox_access_token(settings)
            if access_token:
                collector.headers["Authorization"] = f"Bearer {access_token}"
            intraday_candles = collector.fetch_intraday_candles(instrument_key, "1minute")
        except Exception:
            intraday_candles = []

    points = serialize_points(rows, with_pnl=True)
    previous_points = serialize_points(previous_rows, with_pnl=False)
    current_chart = (
        candle_record_chart(intraday_candles, label="Current expiry", expiry_value=expiry_date)
        if intraday_candles
        else quote_chart(rows, label="Current expiry", expiry_value=expiry_date)
    )
    previous_chart = quote_chart(previous_rows, label="Previous expiry", expiry_value=previous_expiry)
    if not points and ltp is not None:
        from datetime import datetime

        fallback_ltp = float(ltp)
        pnl = ((fallback_ltp - entry) * quantity) if entry is not None else None
        points.append(
            {
                "x": datetime.now().astimezone().isoformat(),
                "ltp": fallback_ltp,
                "pnl": round(pnl, 2) if pnl is not None else None,
                "volume": 0.0,
                "source": "snapshot",
            }
        )
    return {
        "symbol": symbol,
        "expiry_date": expiry_date.isoformat(),
        "strike": float(strike),
        "option_type": normalized_type,
        "entry_price": entry,
        "stop_loss": stop if stop and stop > 0 else None,
        "take_profit": target if target and target > 0 else None,
        "trailing_stop_loss": trailing_stop if trailing_stop and trailing_stop > 0 else None,
        "levels": [
            item
            for item in [
                {"label": "ENTRY", "price": entry, "color": "#f8fafc"} if entry is not None else None,
                {"label": "STOP LOSS", "price": stop, "color": "#ef4444"} if stop and stop > 0 else None,
                {"label": "TRAIL SL", "price": trailing_stop, "color": "#f7c948"} if trailing_stop and trailing_stop > 0 else None,
                {"label": "TARGET", "price": target, "color": "#22c55e"} if target and target > 0 else None,
            ]
            if item is not None
        ],
        "quantity": quantity,
        "points": points,
        "chart": current_chart,
        "previous": {
            "expiry_date": previous_expiry.isoformat() if previous_expiry is not None else None,
            "points": previous_points,
            "chart": previous_chart,
            "history_available": bool(previous_points),
        },
        "history_available": bool(rows),
    }


@router.get("/dashboard-state")
def dashboard_state(db: Session = Depends(get_db_session)) -> dict:
    settings = _runtime_settings(db)
    return {
        "mode": get_runtime_trading_mode(db, settings=settings),
        "paper": compute_paper_portfolio_metrics(db, settings=settings),
    }
