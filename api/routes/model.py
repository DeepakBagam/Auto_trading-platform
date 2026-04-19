from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from api.deps import get_db
from db.models import BacktestRun, CalibrationRegistry, DriftMetric, PredictionsDaily, RawCandle
from utils.symbols import instrument_key_filter, symbol_value_filter

router = APIRouter(prefix="/model", tags=["model"])


def _rolling_error_metrics(db: Session, symbol: str, window: int = 120) -> dict:
    pred_rows = (
        db.execute(
        select(PredictionsDaily)
        .where(and_(symbol_value_filter(PredictionsDaily.symbol, symbol), PredictionsDaily.interval == "day"))
        .order_by(PredictionsDaily.target_session_date.desc(), PredictionsDaily.generated_at.desc())
        .limit(window * 6)
        )
        .scalars()
        .all()
    )
    latest_by_date = {}
    for row in pred_rows:
        prev = latest_by_date.get(row.target_session_date)
        if prev is None or row.generated_at > prev.generated_at:
            latest_by_date[row.target_session_date] = row
    rows = sorted(latest_by_date.values(), key=lambda r: r.target_session_date)[-window:]
    if not rows:
        return {"rolling_mae": None, "rolling_mape": None, "samples": 0}

    abs_err = []
    ape = []
    for row in rows:
        actual = db.scalar(
            select(RawCandle.close).where(
                and_(
                    RawCandle.interval == "day",
                    func.date(RawCandle.ts) == row.target_session_date,
                    instrument_key_filter(RawCandle.instrument_key, symbol),
                )
            )
        )
        if actual is None:
            continue
        err = abs(float(row.pred_close) - float(actual))
        abs_err.append(err)
        ape.append(err / (abs(float(actual)) + 1e-9))
    if not abs_err:
        return {"rolling_mae": None, "rolling_mape": None, "samples": 0}
    return {
        "rolling_mae": float(sum(abs_err) / len(abs_err)),
        "rolling_mape": float(sum(ape) / len(ape)),
        "samples": int(len(abs_err)),
    }


@router.get("/diagnostics")
def diagnostics(
    symbol: str = Query(..., description="Symbol like Nifty 50 / India VIX / SENSEX"),
    db: Session = Depends(get_db),
) -> dict:
    latest_backtest = db.scalar(
        select(BacktestRun)
        .where(and_(symbol_value_filter(BacktestRun.symbol, symbol), BacktestRun.model_family == "meta_v3"))
        .order_by(BacktestRun.created_at.desc())
        .limit(1)
    )
    latest_cal = db.scalar(
        select(CalibrationRegistry)
        .where(
            and_(
                symbol_value_filter(CalibrationRegistry.symbol, symbol),
                CalibrationRegistry.model_family == "meta_v3",
            )
        )
        .order_by(CalibrationRegistry.created_at.desc())
        .limit(1)
    )
    drift_rows = (
        db.execute(
            select(DriftMetric)
            .where(and_(symbol_value_filter(DriftMetric.symbol, symbol), DriftMetric.model_family == "meta_v3"))
            .order_by(DriftMetric.computed_at.desc())
            .limit(20)
        )
        .scalars()
        .all()
    )
    latest_drift = {}
    for r in drift_rows:
        latest_drift.setdefault(r.metric_name, float(r.metric_value))

    rolling = _rolling_error_metrics(db, symbol=symbol, window=120)
    metrics = (latest_backtest.metrics or {}) if latest_backtest else {}
    return {
        "symbol": symbol,
        "model_family": "meta_v3",
        "calibration_version": latest_cal.calibration_version if latest_cal else None,
        "ece": float(metrics.get("ece", latest_drift.get("ece", 1.0))),
        "brier_score": float(metrics.get("brier_calibrated", latest_drift.get("brier_calibrated", 1.0))),
        "interval_coverage": float(metrics.get("coverage_close", latest_drift.get("coverage_close", 0.0))),
        "rolling_mae": rolling["rolling_mae"],
        "rolling_mape": rolling["rolling_mape"],
        "samples": rolling["samples"],
        "promotion_passed": bool(metrics.get("promotion_passed", False)),
        "latest_metrics": metrics,
    }
