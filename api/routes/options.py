from __future__ import annotations

from datetime import date, datetime
from threading import Lock

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.deps import get_db
from api.routes.signal import (
    build_signal_snapshot,
    latest_price_for_symbol,
    technical_context_for_symbol,
)
from api.schemas import OptionsSignalResponse
from data_layer.collectors.upstox_option_chain import UpstoxOptionChainCollector
from db.models import DataFreshness, OptionQuote, OptionTradeSignal
from execution_engine.ai_intelligence import score_trade_intelligence
from prediction_engine.options_engine import (
    OptionQuoteView,
    build_chain_rows,
    build_option_signal,
    max_pain_proxy_for_chain,
    next_weekly_expiries,
    nearest_strike,
    strike_step_for_symbol,
    synthetic_option_chain,
)
from utils.config import get_settings
from utils.intervals import INTERVAL_QUERY_PATTERN, normalize_interval
from utils.constants import IST_ZONE
from utils.symbols import instrument_key_filter, symbol_aliases, symbol_value_filter
from api.routes.predict import predict as predict_single

router = APIRouter(prefix="/options", tags=["options"])
_OPTION_REFRESH_TTL_SECONDS = 25
_refresh_lock = Lock()
_last_option_refresh: dict[tuple[str, str], datetime] = {}


def _ensure_ist(dt: datetime) -> datetime:
    return dt.astimezone(IST_ZONE) if dt.tzinfo is not None else dt.replace(tzinfo=IST_ZONE)


def _load_option_quotes(
    db: Session,
    symbol: str,
    expiry_date: date,
    max_rows: int = 5000,
) -> list[OptionQuoteView]:
    rows = (
        db.execute(
            select(OptionQuote)
            .where(
                symbol_value_filter(OptionQuote.underlying_symbol, symbol),
                OptionQuote.expiry_date == expiry_date,
            )
            .order_by(OptionQuote.ts.desc())
            .limit(max_rows)
        )
        .scalars()
        .all()
    )
    by_contract: dict[tuple[float, str], OptionQuoteView] = {}
    for row in rows:
        key = (float(row.strike), str(row.option_type))
        if key in by_contract:
            continue
        by_contract[key] = OptionQuoteView(
            instrument_key=str(row.instrument_key),
            strike=float(row.strike),
            option_type=str(row.option_type),  # type: ignore[arg-type]
            ltp=float(row.ltp),
            bid=(float(row.bid) if row.bid is not None else None),
            ask=(float(row.ask) if row.ask is not None else None),
            volume=(float(row.volume) if row.volume is not None else None),
            oi=(float(row.oi) if row.oi is not None else None),
            close_price=(float(row.close_price) if row.close_price is not None else None),
            bid_qty=(float(row.bid_qty) if row.bid_qty is not None else None),
            ask_qty=(float(row.ask_qty) if row.ask_qty is not None else None),
            prev_oi=(float(row.prev_oi) if row.prev_oi is not None else None),
            iv=(float(row.iv) if row.iv is not None else None),
            delta=(float(row.delta) if row.delta is not None else None),
            gamma=(float(row.gamma) if row.gamma is not None else None),
            theta=(float(row.theta) if row.theta is not None else None),
            vega=(float(row.vega) if row.vega is not None else None),
            pop=(float(row.pop) if row.pop is not None else None),
            pcr=(float(row.pcr) if row.pcr is not None else None),
            underlying_spot_price=(
                float(row.underlying_spot_price) if row.underlying_spot_price is not None else None
            ),
            source=str(row.source or "db"),
        )
    return sorted(by_contract.values(), key=lambda q: (q.strike, q.option_type))


def _resolve_expiry(
    *,
    symbol: str,
    underlying_key: str | None,
    requested_expiry: date | None,
) -> tuple[date, list[date]]:
    expiries: list[date] = []
    if underlying_key and get_settings().has_market_data_access:
        try:
            expiries = UpstoxOptionChainCollector().list_expiries(underlying_key, max_items=8)
        except Exception:
            expiries = []
    if not expiries:
        expiries = next_weekly_expiries(symbol=symbol, count=6)
    today = datetime.now(IST_ZONE).date()
    valid_expiries = [exp for exp in expiries if (exp - today).days >= 1]
    if valid_expiries:
        expiries = valid_expiries
    if requested_expiry is None:
        return expiries[0], expiries
    if requested_expiry in expiries:
        return requested_expiry, expiries
    for exp in expiries:
        if exp >= requested_expiry:
            return exp, expiries
    return expiries[-1], expiries


def _persist_option_signal(db: Session, payload: dict) -> None:
    signal = payload.get("option_signal") or {}
    try:
        row = OptionTradeSignal(
            symbol=str(payload.get("symbol")),
            interval=str(payload.get("interval")),
            expiry_date=date.fromisoformat(str(payload.get("expiry_date"))),
            option_type=str(signal.get("option_type") or ""),
            side=str(signal.get("side") or "BUY"),
            action=str(signal.get("action") or "HOLD"),
            strike=float(signal["strike"]) if signal.get("strike") is not None else 0.0,
            entry_price=(float(signal["entry_price"]) if signal.get("entry_price") is not None else None),
            stop_loss=(float(signal["stop_loss"]) if signal.get("stop_loss") is not None else None),
            take_profit=(float(signal["take_profit"]) if signal.get("take_profit") is not None else None),
            trailing_stop_loss=(
                float(signal["trailing_stop_loss"]) if signal.get("trailing_stop_loss") is not None else None
            ),
            trail_trigger_price=(
                float(signal["trail_trigger_price"]) if signal.get("trail_trigger_price") is not None else None
            ),
            trail_step_pct=(float(signal["trail_step_pct"]) if signal.get("trail_step_pct") is not None else None),
            confidence=float(signal.get("confidence") or 0.0),
            reasons=list(signal.get("reasons") or []),
            metadata_json={
                "underlying_price": payload.get("underlying_price"),
                "underlying_signal_action": payload.get("underlying_signal_action"),
                "underlying_conviction": payload.get("underlying_conviction"),
                "underlying_confidence": payload.get("underlying_confidence"),
                "underlying_expected_return_pct": payload.get("underlying_expected_return_pct"),
                "strike_mode": payload.get("strike_mode"),
                "manual_strike": payload.get("manual_strike"),
                "strategy": signal.get("strategy"),
                "legs": list(signal.get("legs") or []),
            },
        )
        db.add(row)
        db.commit()
    except Exception:
        db.rollback()


def _mark_option_chain_freshness(db: Session, symbol: str, status: str, details: dict) -> None:
    source_name = f"upstox_option_chain:{symbol}"
    row = db.scalar(select(DataFreshness).where(DataFreshness.source_name == source_name))
    if row is None:
        row = DataFreshness(source_name=source_name, last_success_at=datetime.now(IST_ZONE))
        db.add(row)
    row.last_success_at = datetime.now(IST_ZONE)
    row.status = status
    row.details = details


@router.get("/meta")
def options_meta(
    symbol: str = Query(..., description="Underlying display symbol e.g. Nifty 50"),
    db: Session = Depends(get_db),
) -> dict:
    underlying_key = _resolve_underlying_key(db, symbol)
    expiry, expiries = _resolve_expiry(
        symbol=symbol,
        underlying_key=underlying_key,
        requested_expiry=None,
    )
    return {
        "symbol": symbol,
        "strike_step": strike_step_for_symbol(symbol),
        "default_expiry": expiry.isoformat(),
        "expiries": [e.isoformat() for e in expiries],
        "strategies": [
            "auto",
            "iron_condor",
            "bull_put_spread",
            "bear_call_spread",
            "long_straddle",
            "long_strangle",
            "trend_vwap_oi",
            "ema_vwap_rsi_atr",
        ],
    }


def _resolve_underlying_key(db: Session, symbol: str) -> str | None:
    settings = get_settings()
    for instrument_key in settings.instrument_keys:
        if instrument_key.split("|", 1)[-1] in symbol_aliases(symbol):
            return instrument_key
    key = db.scalar(
        select(OptionQuote.underlying_key)
        .where(symbol_value_filter(OptionQuote.underlying_symbol, symbol))
        .order_by(OptionQuote.ts.desc())
        .limit(1)
    )
    if key:
        return str(key)
    from db.models import RawCandle

    key = db.scalar(
        select(RawCandle.instrument_key)
        .where(instrument_key_filter(RawCandle.instrument_key, symbol))
        .order_by(RawCandle.instrument_key.asc())
        .limit(1)
    )
    return str(key) if key else None


def _latest_option_quote_ts(db: Session, symbol: str, expiry_date: date) -> datetime | None:
    return db.scalar(
        select(func.max(OptionQuote.ts)).where(
            symbol_value_filter(OptionQuote.underlying_symbol, symbol),
            OptionQuote.expiry_date == expiry_date,
        )
    )


def _should_refresh_option_chain(symbol: str, expiry_date: date) -> bool:
    now = datetime.now(IST_ZONE)
    key = (symbol, expiry_date.isoformat())
    with _refresh_lock:
        last = _last_option_refresh.get(key)
        if last is not None and (now - last).total_seconds() < _OPTION_REFRESH_TTL_SECONDS:
            return False
        _last_option_refresh[key] = now
        return True


def _maybe_refresh_option_chain(
    db: Session,
    *,
    symbol: str,
    underlying_key: str | None,
    expiry_date: date,
) -> None:
    settings = get_settings()
    if underlying_key is None or not settings.has_market_data_access:
        return
    latest_ts = _latest_option_quote_ts(db, symbol, expiry_date)
    now = datetime.now(IST_ZONE)
    stale = latest_ts is None or (
        now - _ensure_ist(latest_ts)
    ).total_seconds() > max(2, int(settings.option_chain_refresh_seconds))
    if not stale or not _should_refresh_option_chain(symbol, expiry_date):
        return
    try:
        UpstoxOptionChainCollector().sync_option_chain(
            db,
            underlying_key=underlying_key,
            underlying_symbol=symbol,
            expiry_date=expiry_date,
        )
    except Exception as exc:
        db.rollback()
        _mark_option_chain_freshness(
            db,
            symbol,
            "error",
            {
                "underlying_key": underlying_key,
                "expiry_date": expiry_date.isoformat(),
                "error": str(exc),
            },
        )
        db.commit()


@router.get("/signal", response_model=OptionsSignalResponse)
def options_signal(
    symbol: str = Query(..., description="Underlying display symbol e.g. Nifty 50"),
    interval: str = Query("1minute", pattern=INTERVAL_QUERY_PATTERN),
    prediction_mode: str = Query("standard", pattern="^(standard|session_close)$"),
    expiry_date: date | None = Query(None),
    strike_mode: str = Query("auto", pattern="^(auto|manual)$"),
    strategy_mode: str = Query(
        "auto",
        pattern="^(auto|iron_condor|bull_put_spread|bear_call_spread|long_straddle|long_strangle|trend_vwap_oi|ema_vwap_rsi_atr)$",
    ),
    manual_strike: float | None = Query(None, ge=1),
    allow_option_writing: bool = Query(False),
    db: Session = Depends(get_db),
) -> OptionsSignalResponse:
    try:
        interval = normalize_interval(interval)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if interval != "1minute":
        raise HTTPException(status_code=422, detail="Only 1minute interval is supported")

    prediction = predict_single(
        symbol=symbol,
        interval=interval,
        prediction_mode="standard" if prediction_mode == "session_close" else prediction_mode,
        target_date=None,
        db=db,
    )
    latest_price = latest_price_for_symbol(db, symbol, interval)
    if latest_price is None:
        raise HTTPException(status_code=404, detail=f"No market price available for {symbol}")

    pred_interval = prediction.pred_interval.model_dump() if prediction.pred_interval is not None else None
    technical_context = technical_context_for_symbol(db, symbol, interval)
    underlying_signal = build_signal_snapshot(
        symbol=prediction.symbol,
        interval=prediction.interval,
        prediction_mode=prediction.prediction_mode,
        direction=prediction.direction,
        confidence=float(prediction.confidence_score or prediction.confidence),
        confidence_bucket=prediction.confidence_bucket,
        latest_price=float(latest_price),
        predicted_price=float(prediction.pred_close),
        pred_high=float(prediction.pred_high),
        pred_low=float(prediction.pred_low),
        pred_interval=pred_interval,
        technical_context=technical_context,
        target_session_date=prediction.target_session_date,
        target_ts=prediction.target_ts,
        generated_at=prediction.generated_at,
        model_version=prediction.model_version,
    )

    underlying_key = _resolve_underlying_key(db, symbol)
    resolved_expiry, available_expiries = _resolve_expiry(
        symbol=symbol,
        underlying_key=underlying_key,
        requested_expiry=expiry_date,
    )
    strike_step = strike_step_for_symbol(symbol)
    _maybe_refresh_option_chain(
        db,
        symbol=symbol,
        underlying_key=underlying_key,
        expiry_date=resolved_expiry,
    )
    quotes = _load_option_quotes(db=db, symbol=symbol, expiry_date=resolved_expiry)
    synthetic_fallback = False
    if not quotes:
        synthetic_fallback = True
        quotes = synthetic_option_chain(
            symbol=symbol,
            underlying_price=float(latest_price),
            expiry_date=resolved_expiry,
            strike_step=strike_step,
        )
    chain = build_chain_rows(quotes)
    latest_chain_ts = _latest_option_quote_ts(db, symbol, resolved_expiry)
    chain_source = next((str(q.source or "db") for q in quotes if q.source), "synthetic")
    if synthetic_fallback and latest_chain_ts is None:
        latest_chain_ts = datetime.now(IST_ZONE)
    atm_strike = nearest_strike(float(latest_price), strike_step)
    max_pain = max_pain_proxy_for_chain(chain)
    signal_payload = build_option_signal(
        symbol=symbol,
        interval=interval,
        expiry_date=resolved_expiry,
        underlying_price=float(latest_price),
        underlying_signal_action=str(underlying_signal.action),  # type: ignore[arg-type]
        underlying_conviction=str(underlying_signal.conviction),
        underlying_confidence=float(underlying_signal.confidence),
        underlying_expected_return_pct=float(underlying_signal.expected_return_pct),
        chain_rows=chain,
        strike_step=strike_step,
        strike_mode=strike_mode,
        strategy_mode=strategy_mode,
        manual_strike=manual_strike,
        allow_option_writing=allow_option_writing,
        technical_context=technical_context,
    )
    signal_payload["available_expiries"] = [d.isoformat() for d in available_expiries]
    signal_payload["chain"] = chain
    signal_payload["auto_selected_strike"] = signal_payload.get("option_signal", {}).get("strike")
    if synthetic_fallback:
        signal_payload["option_signal"].setdefault("reasons", []).append(
            "Live option chain unavailable; using synthetic chain fallback"
        )
    ai_signal_action = (
        str(signal_payload["underlying_signal_action"]).upper()
        if str(signal_payload["underlying_signal_action"]).upper() in {"BUY", "SELL"}
        else "HOLD"
    )
    ai = score_trade_intelligence(
        signal_action=ai_signal_action,
        confidence=float(underlying_signal.confidence),
        expected_return_pct=float(underlying_signal.expected_return_pct),
        technical_context=technical_context,
        now=datetime.now(IST_ZONE),
    )
    signal_payload["trade_intelligence"] = {
        "score": float(ai.score),
        "trend_continuation_prob": float(ai.trend_continuation_prob),
        "false_breakout_risk": float(ai.false_breakout_risk),
        "premium_expansion_prob": float(ai.premium_expansion_prob),
        "tod_profitability_score": float(ai.tod_profitability_score),
        "reasons": list(ai.reasons),
    }
    signal_payload["atm_strike"] = float(atm_strike)
    signal_payload["max_pain"] = float(max_pain) if max_pain is not None else None
    signal_payload["chain_source"] = chain_source
    signal_payload["chain_generated_at"] = latest_chain_ts.isoformat() if latest_chain_ts is not None else None
    _persist_option_signal(db, signal_payload)
    return OptionsSignalResponse(
        symbol=str(signal_payload["symbol"]),
        interval=str(signal_payload["interval"]),
        expiry_date=date.fromisoformat(str(signal_payload["expiry_date"])),
        available_expiries=available_expiries,
        underlying_price=float(signal_payload["underlying_price"]),
        underlying_signal_action=str(signal_payload["underlying_signal_action"]),
        underlying_conviction=str(signal_payload["underlying_conviction"]),
        underlying_confidence=float(signal_payload["underlying_confidence"]),
        underlying_expected_return_pct=float(signal_payload["underlying_expected_return_pct"]),
        strike_step=int(signal_payload["strike_step"]),
        strike_mode=str(signal_payload["strike_mode"]),
        manual_strike=(
            float(signal_payload["manual_strike"]) if signal_payload.get("manual_strike") is not None else None
        ),
        auto_selected_strike=(
            float(signal_payload["auto_selected_strike"])
            if signal_payload.get("auto_selected_strike") is not None
            else None
        ),
        atm_strike=(float(signal_payload["atm_strike"]) if signal_payload.get("atm_strike") is not None else None),
        max_pain=(float(signal_payload["max_pain"]) if signal_payload.get("max_pain") is not None else None),
        chain_source=(str(signal_payload["chain_source"]) if signal_payload.get("chain_source") is not None else None),
        chain_generated_at=(
            datetime.fromisoformat(str(signal_payload["chain_generated_at"]))
            if signal_payload.get("chain_generated_at") is not None
            else None
        ),
        option_signal=signal_payload["option_signal"],
        trade_intelligence=signal_payload["trade_intelligence"],
        chain=signal_payload["chain"],
    )
