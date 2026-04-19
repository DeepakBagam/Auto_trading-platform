from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from api.deps import get_db
from api.schemas import PredictResponse
from db.models import PredictionsIntraday, RawCandle
from prediction_engine.orchestrator import PredictionOrchestrator
from utils.constants import IST_ZONE
from utils.intervals import INTERVAL_QUERY_PATTERN, normalize_interval
from utils.symbols import instrument_key_filter, symbol_aliases, symbol_value_filter

router = APIRouter(prefix="/predict", tags=["predict"])


def _ensure_ist(dt: datetime) -> datetime:
    return dt.astimezone(IST_ZONE) if dt.tzinfo is not None else dt.replace(tzinfo=IST_ZONE)


def _select_intraday_prediction(
    rows: list[PredictionsIntraday],
    now: datetime,
    target_date: date | None,
) -> PredictionsIntraday | None:
    if not rows:
        return None
    if target_date is not None:
        same_day = [r for r in rows if _ensure_ist(r.target_ts).date() == target_date]
        if not same_day:
            return None
        future_same_day = [r for r in same_day if _ensure_ist(r.target_ts) >= now]
        return future_same_day[0] if future_same_day else same_day[-1]
    future = [r for r in rows if _ensure_ist(r.target_ts) >= now]
    return future[0] if future else rows[-1]


def _latest_intraday_rows(db: Session, symbol: str, interval: str) -> list[PredictionsIntraday]:
    latest_version = db.scalar(
        select(PredictionsIntraday.model_version)
        .where(
            and_(
                symbol_value_filter(PredictionsIntraday.symbol, symbol),
                PredictionsIntraday.interval == interval,
            )
        )
        .order_by(PredictionsIntraday.generated_at.desc())
        .limit(1)
    )
    if latest_version is None:
        return []
    return (
        db.execute(
            select(PredictionsIntraday)
            .where(
                and_(
                    symbol_value_filter(PredictionsIntraday.symbol, symbol),
                    PredictionsIntraday.interval == interval,
                    PredictionsIntraday.model_version == latest_version,
                )
            )
            .order_by(PredictionsIntraday.target_ts.asc(), PredictionsIntraday.generated_at.desc())
        )
        .scalars()
        .all()
    )


def _latest_candle_ts(db: Session, symbol: str, interval: str) -> datetime | None:
    return db.scalar(
        select(func.max(RawCandle.ts)).where(
            and_(
                instrument_key_filter(RawCandle.instrument_key, symbol),
                RawCandle.interval == interval,
            )
        )
    )


def _prediction_source_candle_ts(row: PredictionsIntraday | None) -> datetime | None:
    if row is None:
        return None
    meta = row.metadata_json or {}
    raw = meta.get("source_candle_ts")
    if not raw:
        return None
    try:
        value = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return _ensure_ist(value)


def _prediction_is_stale(
    row: PredictionsIntraday | None,
    *,
    latest_candle_ts: datetime | None,
    now: datetime,
) -> bool:
    if row is None:
        return True
    target_ts = _ensure_ist(row.target_ts)
    if target_ts < now:
        return True
    if latest_candle_ts is None:
        return False
    source_ts = _prediction_source_candle_ts(row)
    if source_ts is not None:
        return _ensure_ist(latest_candle_ts) > source_ts
    generated_at = _ensure_ist(row.generated_at)
    return _ensure_ist(latest_candle_ts) > generated_at


@router.get("", response_model=PredictResponse)
def predict(
    symbol: str = Query(..., description="Trading symbol"),
    interval: str = Query("1minute", pattern=INTERVAL_QUERY_PATTERN, description="Prediction interval"),
    prediction_mode: str = Query(
        "standard",
        pattern="^(standard|session_close)$",
        description="Prediction target mode",
    ),
    target_date: date | None = Query(None, description="Target session date"),
    db: Session = Depends(get_db),
) -> PredictResponse:
    try:
        interval = normalize_interval(interval)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if interval != "1minute":
        raise HTTPException(status_code=422, detail="Only 1minute interval is supported")

    now = datetime.now(IST_ZONE)
    rows = _latest_intraday_rows(db, symbol, interval)
    selected = _select_intraday_prediction(rows, now=now, target_date=target_date)
    latest_candle_ts = _latest_candle_ts(db, symbol, interval)
    if _prediction_is_stale(selected, latest_candle_ts=latest_candle_ts, now=now):
        orchestrator = PredictionOrchestrator()
        out = None
        for candidate in symbol_aliases(symbol):
            out = orchestrator.run_intraday_inference(db, [candidate], interval=interval).get(candidate)
            if out is not None and "error" not in out:
                break
        if out is None or "error" in out:
            raise HTTPException(status_code=404, detail=f"No intraday prediction available for {symbol}")
        rows = _latest_intraday_rows(db, symbol, interval)
        selected = _select_intraday_prediction(rows, now=now, target_date=target_date)
    if selected is None:
        raise HTTPException(status_code=404, detail=f"No prediction found for {symbol} ({interval})")

    meta = selected.metadata_json or {}
    components = meta.get("confidence_components")
    return PredictResponse(
        symbol=selected.symbol,
        interval=selected.interval,
        prediction_mode="standard" if prediction_mode == "session_close" else prediction_mode,
        source_interval=selected.interval,
        target_session_date=_ensure_ist(selected.target_ts).date(),
        target_ts=_ensure_ist(selected.target_ts),
        pred_open=selected.pred_open,
        pred_high=selected.pred_high,
        pred_low=selected.pred_low,
        pred_close=selected.pred_close,
        direction=selected.direction,
        confidence=selected.confidence,
        direction_prob_calibrated=meta.get("direction_prob_calibrated"),
        confidence_score=float(meta.get("confidence_score", selected.confidence)),
        confidence_bucket=meta.get("confidence_bucket"),
        pred_interval=meta.get("pred_interval"),
        model_family=meta.get("model_family"),
        calibration_version=meta.get("calibration_version"),
        confidence_components=components if isinstance(components, dict) else None,
        model_version=selected.model_version,
        feature_cutoff_ist=selected.feature_cutoff_ist,
        generated_at=selected.generated_at,
    )
