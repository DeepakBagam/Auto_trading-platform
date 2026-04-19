from datetime import datetime, timedelta
from threading import Lock

from fastapi import APIRouter, Depends, HTTPException, Query
import pandas as pd
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from api.deps import get_db
from backtesting.engine import _pine_signals
from api.routes.predict import predict as predict_single
from api.routes.signal import (
    build_signal_snapshot,
    latest_price_for_symbol,
    technical_context_for_symbol,
)
from api.schemas import DataFreshnessResponse
from data_layer.collectors.upstox_collector import UpstoxCollector
from db.models import (
    DataFreshness,
    ExecutionExternalSignal,
    ExecutionPosition,
    ExecutionSignalAudit,
    SignalLog,
    RawCandle,
)
from execution_engine.ai_intelligence import score_trade_intelligence
from feature_engine.price_features import build_price_features
from utils.calendar_utils import is_trading_day, market_session_bounds, previous_trading_day
from utils.config import get_settings
from utils.constants import IST_ZONE, SUPPORTED_INTERVALS
from utils.intervals import INTERVAL_QUERY_PATTERN, normalize_interval
from utils.symbols import (
    display_symbol_from_instrument_key,
    instrument_key_filter,
    sort_display_symbols,
    symbol_value_filter,
)

router = APIRouter(prefix="/data", tags=["data"])
_CANDLE_REFRESH_TTL_SECONDS = 25
_refresh_lock = Lock()
_last_candle_refresh: dict[str, datetime] = {}


def _ensure_ist(dt: datetime) -> datetime:
    return dt.astimezone(IST_ZONE) if dt.tzinfo is not None else dt.replace(tzinfo=IST_ZONE)


def _to_float(value) -> float | None:
    try:
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "-"
    seconds_i = int(round(seconds))
    mins, secs = divmod(seconds_i, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m {secs}s"


def _build_indicator_matrix(technical_context: dict | None) -> list[dict]:
    if not technical_context:
        return []
    ordered_keys = [
        "ema_9",
        "ema_21",
        "ema_50",
        "ema_200",
        "vwap",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "kc_upper",
        "kc_mid",
        "kc_lower",
        "atr_14",
        "volatility_20d",
        "pattern_engulfing",
        "pattern_marubozu",
        "pattern_hanging_man",
        "pattern_shooting_star",
        "pattern_spinning_top",
    ]
    label_map = {
        "ema_9": "EMA 9",
        "ema_21": "EMA 21",
        "ema_50": "EMA 50",
        "ema_200": "EMA 200",
        "vwap": "VWAP",
        "rsi_14": "RSI 14",
        "macd": "MACD",
        "macd_signal": "MACD Signal",
        "macd_hist": "MACD Hist",
        "bb_upper": "BB Upper",
        "bb_mid": "BB Mid",
        "bb_lower": "BB Lower",
        "kc_upper": "KC Upper",
        "kc_mid": "KC Mid",
        "kc_lower": "KC Lower",
        "atr_14": "ATR 14",
        "volatility_20d": "Vol 20",
        "pattern_engulfing": "Engulfing",
        "pattern_marubozu": "Marubozu",
        "pattern_hanging_man": "Hanging Man",
        "pattern_shooting_star": "Shooting Star",
        "pattern_spinning_top": "Spinning Top",
    }
    close = _to_float(technical_context.get("close"))
    ema21 = _to_float(technical_context.get("ema_21"))
    ema50 = _to_float(technical_context.get("ema_50"))
    out: list[dict] = []
    for key in ordered_keys:
        value = technical_context.get(key)
        fv = _to_float(value)
        if fv is None and value is None:
            continue
        bias = "neutral"
        if key.startswith("ema_") and close is not None and fv is not None:
            bias = "bullish" if close >= fv else "bearish"
        elif key == "vwap" and close is not None and fv is not None:
            bias = "bullish" if close >= fv else "bearish"
        elif key == "rsi_14" and fv is not None:
            if fv >= 58:
                bias = "bullish"
            elif fv <= 42:
                bias = "bearish"
        elif key in {"macd", "macd_hist"} and fv is not None:
            bias = "bullish" if fv >= 0 else "bearish"
        elif key == "macd_signal" and fv is not None and _to_float(technical_context.get("macd")) is not None:
            bias = "bullish" if _to_float(technical_context.get("macd")) >= fv else "bearish"
        elif key.startswith("pattern_") and fv is not None and fv >= 1.0:
            if key in {"pattern_hanging_man", "pattern_shooting_star"}:
                bias = "bearish"
            elif key in {"pattern_engulfing", "pattern_marubozu"}:
                bias = "bullish"
        elif key in {"bb_mid", "kc_mid"} and close is not None and fv is not None:
            bias = "bullish" if close >= fv else "bearish"
        elif key in {"atr_14", "volatility_20d"} and fv is not None:
            bias = "risk_on" if fv > 0 else "neutral"

        text_value = "-"
        if fv is not None:
            if key.startswith("pattern_"):
                text_value = "ON" if fv >= 1.0 else "OFF"
            elif abs(fv) >= 100:
                text_value = f"{fv:.2f}"
            elif abs(fv) >= 1:
                text_value = f"{fv:.3f}"
            else:
                text_value = f"{fv:.4f}"
        out.append(
            {
                "name": key,
                "label": label_map.get(key, key),
                "value": text_value,
                "bias": bias,
            }
        )
    if close is not None and ema21 is not None and ema50 is not None:
        out.append(
            {
                "name": "trend_state",
                "label": "Trend State",
                "value": "UP" if close > ema21 > ema50 else ("DOWN" if close < ema21 < ema50 else "RANGE"),
                "bias": "bullish" if close > ema21 > ema50 else ("bearish" if close < ema21 < ema50 else "neutral"),
            }
        )
    return out


def _build_trade_analytics(db: Session, symbol: str, lookback_days: int = 30) -> dict:
    today = datetime.now(IST_ZONE).date()
    start_date = today - timedelta(days=max(1, lookback_days))
    try:
        rows = (
            db.execute(
                select(ExecutionPosition)
                .where(
                    and_(
                        symbol_value_filter(ExecutionPosition.symbol, symbol),
                        ExecutionPosition.trade_date >= start_date,
                    )
                )
                .order_by(ExecutionPosition.opened_at.asc())
            )
            .scalars()
            .all()
        )
    except Exception:
        rows = []

    closed = [r for r in rows if str(getattr(r, "status", "")).upper() == "CLOSED"]
    total = len(closed)
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    total_pnl = 0.0
    duration_seconds: list[float] = []
    equity_curve: list[dict] = []
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for row in closed:
        pnl = _to_float(getattr(row, "pnl_value", None)) or 0.0
        total_pnl += pnl
        if pnl >= 0:
            wins += 1
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)
        opened_at = getattr(row, "opened_at", None)
        closed_at = getattr(row, "closed_at", None)
        if opened_at is not None and closed_at is not None:
            dur = (_ensure_ist(closed_at) - _ensure_ist(opened_at)).total_seconds()
            if dur > 0:
                duration_seconds.append(dur)
        cumulative += pnl
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
        ts = closed_at or opened_at
        if ts is not None:
            equity_curve.append({"x": _ensure_ist(ts).isoformat(), "value": round(cumulative, 2)})

    win_rate_pct = (wins / total * 100.0) if total > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
    avg_duration_sec = (sum(duration_seconds) / len(duration_seconds)) if duration_seconds else None
    max_drawdown_pct = (max_dd / peak * 100.0) if peak > 0 else 0.0

    trade_log = []
    for row in list(reversed(rows))[:40]:
        status = str(getattr(row, "status", "OPEN")).upper()
        entry = _to_float(getattr(row, "entry_price", None))
        current = _to_float(getattr(row, "current_price", None))
        pnl_value = _to_float(getattr(row, "pnl_value", None)) or 0.0
        result = "OPEN" if status != "CLOSED" else ("WIN" if pnl_value >= 0 else "LOSS")
        ts = getattr(row, "opened_at", None)
        trade_log.append(
            {
                "time": _ensure_ist(ts).isoformat() if ts is not None else None,
                "instrument": f"{getattr(row, 'symbol', symbol)} {getattr(row, 'strike', 0):.0f} {getattr(row, 'option_type', '-')}",
                "side": str(getattr(row, "side", "-")).upper(),
                "strategy": str(getattr(row, "strategy_name", "-")),
                "entry_price": entry,
                "exit_price": current if status == "CLOSED" else None,
                "pnl_value": pnl_value,
                "result": result,
            }
        )

    return {
        "lookback_days": lookback_days,
        "win_rate_pct": round(win_rate_pct, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_duration_seconds": round(avg_duration_sec, 2) if avg_duration_sec is not None else None,
        "avg_duration_text": _format_duration(avg_duration_sec),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "gross_pnl": round(total_pnl, 2),
        "total_trades": total,
        "equity_curve": equity_curve[-300:],
        "trade_log": trade_log,
    }


def _build_model_variance(signal_payload: dict, technical_context: dict | None) -> list[dict]:
    confidence = _to_float(signal_payload.get("confidence")) or 0.0
    technical = _to_float(signal_payload.get("technical_score")) or 0.0
    rr = _to_float(signal_payload.get("risk_reward_ratio")) or 1.0
    macd_hist = _to_float((technical_context or {}).get("macd_hist")) or 0.0
    rsi = _to_float((technical_context or {}).get("rsi_14")) or 50.0

    meta_eff = _clip(confidence * 100.0, 25.0, 96.0)
    sequence_eff = _clip(56.0 + (technical * 7.5) + (macd_hist * 4.0), 20.0, 94.0)
    regime_eff = _clip(45.0 + (rr * 10.0) + (abs(rsi - 50.0) * 0.3), 18.0, 92.0)
    return [
        {"model": "Meta Ensemble", "efficiency": round(meta_eff, 1)},
        {"model": "Sequence Trend", "efficiency": round(sequence_eff, 1)},
        {"model": "Regime Volatility", "efficiency": round(regime_eff, 1)},
    ]


def _latest_candle_ts(db: Session, instrument_key: str, interval: str) -> datetime | None:
    return db.scalar(
        select(func.max(RawCandle.ts)).where(
            and_(
                RawCandle.instrument_key == instrument_key,
                RawCandle.interval == interval,
            )
        )
    )


def _should_refresh_candles(instrument_key: str) -> bool:
    now = datetime.now(IST_ZONE)
    with _refresh_lock:
        last = _last_candle_refresh.get(instrument_key)
        if last is not None and (now - last).total_seconds() < _CANDLE_REFRESH_TTL_SECONDS:
            return False
        _last_candle_refresh[instrument_key] = now
        return True


def _maybe_refresh_intraday_candles(db: Session, instrument_key: str, interval: str) -> None:
    settings = get_settings()
    if interval != "1minute" or not settings.upstox_access_token.strip():
        return

    latest_ts = _latest_candle_ts(db, instrument_key, interval)
    now = datetime.now(IST_ZONE)
    session_start, session_end = market_session_bounds(now.date())
    if not is_trading_day(now.date()) or now < session_start or now > session_end:
        return
    stale = latest_ts is None or _ensure_ist(latest_ts).date() < now.date()
    if not stale and latest_ts is not None:
        stale = (now - _ensure_ist(latest_ts)).total_seconds() > 120
    if not stale or not _should_refresh_candles(instrument_key):
        return

    try:
        collector = UpstoxCollector()
        records = collector.fetch_intraday_candles(instrument_key, interval)
        collector.persist(db, records, update_existing=True)
    except Exception:
        db.rollback()


def _build_freshness_payload(
    db: Session,
    *,
    instrument_key: str,
    display_symbol: str,
    interval: str,
    latest_candle_ts: datetime | None,
) -> dict:
    now = datetime.now(IST_ZONE)
    session_start, session_end = market_session_bounds(now.date())
    latest_session_date = _ensure_ist(latest_candle_ts).date() if latest_candle_ts is not None else None
    if not is_trading_day(now.date()) or now < session_start:
        expected_session_date = previous_trading_day(now.date())
    else:
        expected_session_date = now.date()
    age_seconds = None
    if latest_candle_ts is not None:
        age_seconds = max(0.0, (now - _ensure_ist(latest_candle_ts)).total_seconds())
    sources = (
        db.execute(
            select(DataFreshness).where(
                DataFreshness.source_name.in_(
                    [
                        "upstox_candles",
                        "upstox_market_stream",
                        f"upstox_option_chain:{display_symbol}",
                    ]
                )
            )
        )
        .scalars()
        .all()
    )
    is_live = bool(
        latest_session_date == now.date()
        and session_start <= now <= session_end
        and age_seconds is not None
        and age_seconds <= 120.0
    )
    session_aligned = bool(latest_session_date is not None and latest_session_date >= expected_session_date)
    market_status = "live" if is_live else ("complete_previous_session" if session_aligned else "stale")

    return {
        "symbol": display_symbol,
        "instrument_key": instrument_key,
        "interval": interval,
        "latest_candle_ts": _ensure_ist(latest_candle_ts).isoformat() if latest_candle_ts is not None else None,
        "latest_candle_age_seconds": round(age_seconds, 1) if age_seconds is not None else None,
        "latest_session_date": latest_session_date.isoformat() if latest_session_date is not None else None,
        "expected_session_date": expected_session_date.isoformat(),
        "market_status": market_status,
        "is_live": is_live,
        "sources": [
            {
                "source_name": row.source_name,
                "last_success_at": _ensure_ist(row.last_success_at).isoformat(),
                "status": row.status,
                "details": row.details or {},
            }
            for row in sources
            if getattr(row, "source_name", None)
        ],
    }


@router.get("/freshness", response_model=list[DataFreshnessResponse])
def data_freshness(db: Session = Depends(get_db)) -> list[DataFreshnessResponse]:
    rows = db.execute(select(DataFreshness).order_by(DataFreshness.source_name.asc())).scalars().all()
    return [
        DataFreshnessResponse(
            source_name=r.source_name,
            last_success_at=r.last_success_at,
            status=r.status,
            details=r.details or {},
        )
        for r in rows
    ]


@router.get("/chart/options")
def chart_options(db: Session = Depends(get_db)) -> dict:
    instrument_keys = db.scalars(
        select(RawCandle.instrument_key).distinct().order_by(RawCandle.instrument_key.asc())
    ).all()
    configured_keys = get_settings().instrument_keys
    symbols = sort_display_symbols(
        [display_symbol_from_instrument_key(key) for key in [*instrument_keys, *configured_keys]]
    )
    return {"symbols": symbols, "intervals": list(SUPPORTED_INTERVALS)}


def _resolve_instrument_key(db: Session, symbol: str) -> tuple[str, str]:
    if "|" in symbol:
        return symbol, display_symbol_from_instrument_key(symbol)
    key = db.scalar(
        select(RawCandle.instrument_key)
        .where(instrument_key_filter(RawCandle.instrument_key, symbol))
        .order_by(RawCandle.instrument_key.asc())
        .limit(1)
    )
    if key is None:
        key = db.scalar(select(RawCandle.instrument_key).where(instrument_key_filter(RawCandle.instrument_key, symbol)).limit(1))
    if key is None:
        raise HTTPException(status_code=404, detail=f"Symbol not found in candles: {symbol}")
    display_symbol = display_symbol_from_instrument_key(key)
    return key, display_symbol


def _build_actual_payload(candle_rows: list[RawCandle]) -> list[dict]:
    return [
        {
            "x": _ensure_ist(c.ts).isoformat(),
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume),
        }
        for c in candle_rows
    ]


def _build_overlay_payload(candle_rows: list[RawCandle]) -> dict[str, list[dict]]:
    if len(candle_rows) < 25:
        return {"ema21": [], "ema50": []}
    frame = build_price_features(
        pd.DataFrame(
            {
                "ts": [r.ts for r in candle_rows],
                "open": [float(r.open) for r in candle_rows],
                "high": [float(r.high) for r in candle_rows],
                "low": [float(r.low) for r in candle_rows],
                "close": [float(r.close) for r in candle_rows],
                "volume": [float(r.volume) for r in candle_rows],
            }
        )
    )
    overlays: dict[str, list[dict]] = {"ema21": [], "ema50": []}
    for _, row in frame.iterrows():
        x = _ensure_ist(row["ts"]).isoformat()
        for name, column in (("ema21", "ema_21"), ("ema50", "ema_50")):
            value = row.get(column)
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if fv == fv:
                overlays[name].append({"x": x, "value": fv})
    return overlays


def _build_confluence_markers_from_actual(actual_payload: list[dict], *, symbol: str) -> list[dict]:
    if len(actual_payload) < 60:
        return []

    frame = pd.DataFrame(
        {
            "ts": [pd.to_datetime(r.get("x")) for r in actual_payload],
            "open": [float(r.get("open") or 0.0) for r in actual_payload],
            "high": [float(r.get("high") or 0.0) for r in actual_payload],
            "low": [float(r.get("low") or 0.0) for r in actual_payload],
            "close": [float(r.get("close") or 0.0) for r in actual_payload],
            "volume": [float(r.get("volume") or 0.0) for r in actual_payload],
        }
    )
    frame = _pine_signals(frame, symbol=symbol)
    markers: list[dict] = []
    for idx in range(len(frame)):
        row = frame.iloc[idx]
        try:
            ts = _ensure_ist(pd.to_datetime(row["ts"]).to_pydatetime())
            close = float(row["close"])
        except (TypeError, ValueError):
            continue

        if close != close:
            continue
        buy_alignment = int(row.get("buy_alignment_count") or 0)
        sell_alignment = int(row.get("sell_alignment_count") or 0)
        market_regime = str(row.get("market_regime") or "range").upper()
        buy_families = [part for part in str(row.get("buy_strategy_names") or "").split(",") if part]
        sell_families = [part for part in str(row.get("sell_strategy_names") or "").split(",") if part]

        if bool(row.get("buy_signal")):
            markers.append(
                {
                    "time": ts.isoformat(),
                    "position": "belowBar",
                    "color": "#0d9a6b",
                    "shape": "arrowUp",
                    "text": f"PINE BUY {buy_alignment}TF",
                    "price": close,
                    "action": "BUY",
                    "conviction": "medium",
                    "regime": market_regime,
                }
            )
            continue
        if bool(row.get("sell_signal")):
            markers.append(
                {
                    "time": ts.isoformat(),
                    "position": "aboveBar",
                    "color": "#d1533f",
                    "shape": "arrowDown",
                    "text": f"PINE SELL {sell_alignment}TF",
                    "price": close,
                    "action": "SELL",
                    "conviction": "medium",
                    "regime": market_regime,
                }
            )
            continue
        if bool(row.get("base_buy_setup")):
            markers.append(
                {
                    "time": ts.isoformat(),
                    "position": "belowBar",
                    "color": "#53d28c",
                    "shape": "circle",
                    "text": f"SETUP BUY 1M {buy_families[0].replace('_', ' ').upper() if buy_families else market_regime}",
                    "price": close,
                    "action": "BUY_SETUP",
                    "conviction": "low",
                    "regime": market_regime,
                }
            )
        if bool(row.get("base_sell_setup")):
            markers.append(
                {
                    "time": ts.isoformat(),
                    "position": "aboveBar",
                    "color": "#ff8a80",
                    "shape": "circle",
                    "text": f"SETUP SELL 1M {sell_families[0].replace('_', ' ').upper() if sell_families else market_regime}",
                    "price": close,
                    "action": "SELL_SETUP",
                    "conviction": "low",
                    "regime": market_regime,
                }
            )
    return markers


def _build_trade_markers_from_positions(db: Session, symbol: str) -> list[dict]:
    rows = (
        db.execute(
            select(ExecutionPosition)
            .where(symbol_value_filter(ExecutionPosition.symbol, symbol))
            .order_by(ExecutionPosition.opened_at.asc())
            .limit(200)
        )
        .scalars()
        .all()
    )
    markers: list[dict] = []
    for row in rows:
        opened_at = getattr(row, "opened_at", None)
        closed_at = getattr(row, "closed_at", None)
        option_type = str(getattr(row, "option_type", "")).upper()
        if opened_at:
            entry_text = "BUY ENTRY" if option_type == "CE" else "SELL ENTRY"
            markers.append(
                {
                    "time": _ensure_ist(opened_at).isoformat(),
                    "position": "belowBar" if option_type == "CE" else "aboveBar",
                    "color": "#00C853" if option_type == "CE" else "#FF3D00",
                    "shape": "arrowUp" if option_type == "CE" else "arrowDown",
                    "text": entry_text,
                }
            )
        if closed_at:
            markers.append(
                {
                    "time": _ensure_ist(closed_at).isoformat(),
                    "position": "inBar",
                    "color": "#FFD600",
                    "shape": "circle",
                    "text": "EXIT",
                }
            )
    return markers


def _build_signal_markers_from_logs(db: Session, symbol: str, interval: str) -> list[dict]:
    markers: list[dict] = []
    signal_rows = (
        db.execute(
            select(SignalLog)
            .where(and_(symbol_value_filter(SignalLog.symbol, symbol), SignalLog.interval == interval))
            .order_by(SignalLog.timestamp.desc())
            .limit(300)
        )
        .scalars()
        .all()
    )
    for row in reversed(signal_rows):
        action = str(getattr(row, "consensus", "")).upper()
        if action not in {"BUY", "SELL"}:
            continue
        markers.append(
            {
                "time": _ensure_ist(row.timestamp).isoformat(),
                "position": "belowBar" if action == "BUY" else "aboveBar",
                "color": "#1a6fff" if action == "BUY" else "#ff8f00",
                "shape": "arrowUp" if action == "BUY" else "arrowDown",
                "text": f"{action} SIG",
            }
        )

    pine_rows = (
        db.execute(
            select(ExecutionExternalSignal)
            .where(
                and_(
                    symbol_value_filter(ExecutionExternalSignal.symbol, symbol),
                    ExecutionExternalSignal.interval == interval,
                    ExecutionExternalSignal.source.in_(["pine", "pine_v2"]),
                )
            )
            .order_by(ExecutionExternalSignal.signal_ts.desc())
            .limit(300)
        )
        .scalars()
        .all()
    )
    for row in reversed(pine_rows):
        action = str(getattr(row, "signal_action", "")).upper()
        if action not in {"BUY", "SELL"}:
            continue
        markers.append(
            {
                "time": _ensure_ist(row.signal_ts).isoformat(),
                "position": "belowBar" if action == "BUY" else "aboveBar",
                "color": "#6ea8fe" if action == "BUY" else "#ffb86c",
                "shape": "circle",
                "text": f"PINE {action}",
            }
        )
    return markers


def _latest_pine_signal_for_candle(
    db: Session,
    *,
    symbol: str,
    interval: str,
    candle_ts: datetime,
    window_seconds: int = 120,
) -> ExecutionExternalSignal | None:
    lo = candle_ts - timedelta(seconds=window_seconds)
    hi = candle_ts + timedelta(seconds=window_seconds)
    return db.scalar(
        select(ExecutionExternalSignal)
        .where(
            and_(
                symbol_value_filter(ExecutionExternalSignal.symbol, symbol),
                ExecutionExternalSignal.interval == interval,
                ExecutionExternalSignal.source.in_(["pine", "pine_v2"]),
                ExecutionExternalSignal.signal_ts >= lo,
                ExecutionExternalSignal.signal_ts <= hi,
            )
        )
        .order_by(ExecutionExternalSignal.signal_ts.desc())
        .limit(1)
    )


def _build_execution_status(
    db: Session,
    *,
    symbol: str,
    interval: str,
    candle_ts: datetime | None,
    signal_action: str,
) -> dict:
    latest_log = db.scalar(
        select(SignalLog)
        .where(and_(symbol_value_filter(SignalLog.symbol, symbol), SignalLog.interval == interval))
        .order_by(SignalLog.timestamp.desc())
        .limit(1)
    )
    if latest_log is not None:
        log_ts = _ensure_ist(latest_log.timestamp)
        candle_ref = _ensure_ist(candle_ts) if candle_ts is not None else None
        if candle_ref is None or abs((log_ts - candle_ref).total_seconds()) <= 300:
            return {
                "consensus_required": True,
                "consensus_state": latest_log.consensus,
                "pine_action": latest_log.pine_signal,
                "ml_action": latest_log.ml_signal,
                "execution_eligible": bool(latest_log.trade_placed or latest_log.consensus in {"BUY", "SELL"}),
                "skip_reason": latest_log.skip_reason,
                "audit_executed": bool(latest_log.trade_placed),
            }

    action = str(signal_action or "HOLD").upper()
    if candle_ts is None or action not in {"BUY", "SELL"}:
        return {
            "consensus_required": True,
            "consensus_state": "non_trade_signal",
            "execution_eligible": False,
            "skip_reason": "non_trade_signal",
            "audit_executed": False,
        }

    latest_audit = db.scalar(
        select(ExecutionSignalAudit)
        .where(
            and_(
                symbol_value_filter(ExecutionSignalAudit.symbol, symbol),
                ExecutionSignalAudit.interval == interval,
                ExecutionSignalAudit.candle_ts == candle_ts,
            )
        )
        .order_by(ExecutionSignalAudit.created_at.desc())
        .limit(1)
    )
    return {
        "consensus_required": True,
        "consensus_state": "non_trade_signal",
        "pine_action": None,
        "ml_action": action,
        "execution_eligible": False,
        "skip_reason": latest_audit.skip_reason if latest_audit is not None else "non_trade_signal",
        "audit_executed": bool(latest_audit.executed) if latest_audit is not None else False,
    }


@router.get("/chart")
def chart_data(
    symbol: str = Query(..., description="Display symbol like Nifty 50 or instrument key"),
    interval: str = Query("1minute", pattern=INTERVAL_QUERY_PATTERN),
    prediction_target_mode: str = Query("standard", pattern="^(standard|session_close)$"),
    candles_limit: int = Query(1200, ge=20, le=50000),
    predictions_limit: int = Query(300, ge=1, le=5000),
    include_historical_predictions: bool = Query(False),
    db: Session = Depends(get_db),
) -> dict:
    del predictions_limit, include_historical_predictions
    try:
        interval = normalize_interval(interval)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if interval != "1minute":
        raise HTTPException(status_code=422, detail="Only 1minute interval is supported")
    prediction_target_mode = "standard"

    instrument_key, display_symbol = _resolve_instrument_key(db, symbol)
    _maybe_refresh_intraday_candles(db, instrument_key, interval)
    candle_rows = (
        db.execute(
            select(RawCandle)
            .where(and_(RawCandle.instrument_key == instrument_key, RawCandle.interval == interval))
            .order_by(RawCandle.ts.desc())
            .limit(candles_limit)
        )
        .scalars()
        .all()
    )
    candle_rows.reverse()
    if not candle_rows:
        raise HTTPException(status_code=404, detail=f"No 1minute candles found for {display_symbol}")

    actual_payload = _build_actual_payload(candle_rows)
    overlays = _build_overlay_payload(candle_rows)
    technical_context = technical_context_for_symbol(db, display_symbol, interval)

    prediction = predict_single(
        symbol=display_symbol,
        interval=interval,
        prediction_mode=prediction_target_mode,
        target_date=None,
        db=db,
    )
    latest_price = latest_price_for_symbol(db, display_symbol, interval)
    if latest_price is None:
        raise HTTPException(status_code=404, detail=f"No market price available for {display_symbol}")
    pred_interval = prediction.pred_interval.model_dump() if prediction.pred_interval is not None else {}
    signal_payload = build_signal_snapshot(
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
    ).model_dump(mode="json")

    markers = _build_trade_markers_from_positions(db, display_symbol)
    markers.extend(_build_signal_markers_from_logs(db, display_symbol, interval))
    markers.extend(_build_confluence_markers_from_actual(actual_payload, symbol=display_symbol))
    deduped_markers: dict[tuple[str, str, str], dict] = {}
    for marker in markers:
        key = (str(marker.get("time")), str(marker.get("shape")), str(marker.get("text")))
        deduped_markers[key] = marker
    markers = sorted(deduped_markers.values(), key=lambda row: str(row.get("time") or ""))
    latest_candle_ts = _ensure_ist(candle_rows[-1].ts) if candle_rows else None
    execution_status = _build_execution_status(
        db,
        symbol=display_symbol,
        interval=interval,
        candle_ts=latest_candle_ts,
        signal_action=str(signal_payload.get("action") or "HOLD"),
    )
    indicator_matrix = _build_indicator_matrix(technical_context)
    analytics = _build_trade_analytics(db, display_symbol, lookback_days=30)
    model_variance = _build_model_variance(signal_payload, technical_context)
    analysis_now = latest_candle_ts or datetime.now(IST_ZONE)
    ai = score_trade_intelligence(
        signal_action=str(signal_payload.get("action") or "HOLD"),
        confidence=float(signal_payload.get("confidence") or 0.0),
        expected_return_pct=float(signal_payload.get("expected_return_pct") or 0.0),
        technical_context=technical_context,
        now=analysis_now,
    )
    freshness = _build_freshness_payload(
        db,
        instrument_key=instrument_key,
        display_symbol=display_symbol,
        interval=interval,
        latest_candle_ts=latest_candle_ts,
    )
    diagnostics = {
        "latest_actual_ts": actual_payload[-1]["x"] if actual_payload else None,
        "first_pred_ts": None,
        "overlap_count": 0,
        "overlap_mae_close": None,
        "indicator_count": len(indicator_matrix),
        "warning": "Prediction candles are hidden in 1m production desk mode.",
    }
    return {
        "symbol": display_symbol,
        "instrument_key": instrument_key,
        "interval": interval,
        "prediction_target_mode": prediction_target_mode,
        "include_historical_predictions": False,
        "actual": actual_payload,
        "predicted": [],
        "overlays": overlays,
        "markers": markers,
        "signal": signal_payload,
        "indicator_matrix": indicator_matrix,
        "model_variance": model_variance,
        "analytics": analytics,
        "diagnostics": diagnostics,
        "execution_status": execution_status,
        "trade_intelligence": {
            "score": float(ai.score),
            "trend_continuation_prob": float(ai.trend_continuation_prob),
            "false_breakout_risk": float(ai.false_breakout_risk),
            "premium_expansion_prob": float(ai.premium_expansion_prob),
            "tod_profitability_score": float(ai.tod_profitability_score),
            "reasons": list(ai.reasons),
        },
        "freshness": freshness,
    }
