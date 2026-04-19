from __future__ import annotations

from datetime import datetime, time

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from db.models import DataFreshness, FeaturesDaily, PredictionsDaily, PredictionsIntraday, RawCandle
from feature_engine.feature_builder import latest_feature_cutoff_ist
from models.common import load_feature_frame
from models.intraday_v1 import predict_intraday_horizon
from models.meta_model.predict import predict_symbol as predict_meta_symbol
from models.meta_v3.predict import predict_symbol as predict_meta_v3_symbol
from models.xgboost.predict import predict_from_feature_frame, predict_symbol
from prediction_engine.combine_logic import combine_predictions
from prediction_engine.confidence_engine import compute_confidence, compute_confidence_v3
from utils.calendar_utils import next_trading_day
from utils.constants import DIRECTION_BUY, DIRECTION_HOLD, DIRECTION_SELL, IST_ZONE
from utils.intervals import normalize_interval
from utils.logger import get_logger

logger = get_logger(__name__)


def _bucket_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _downgrade_bucket(bucket: str) -> str:
    if bucket == "high":
        return "medium"
    return "low"


class PredictionOrchestrator:
    @staticmethod
    def _staleness_minutes(db: Session) -> float:
        rows = db.execute(select(DataFreshness)).scalars().all()
        if not rows:
            return 9999.0
        now = datetime.now(IST_ZONE)
        lags = []
        for row in rows:
            last = row.last_success_at
            if last is None:
                continue
            if last.tzinfo is None:
                last = last.replace(tzinfo=IST_ZONE)
            lags.append(max(0.0, (now - last).total_seconds() / 60.0))
        return max(lags) if lags else 9999.0

    def run_daily_inference(self, db: Session, symbols: list[str]) -> dict:
        summary = {}
        for symbol in symbols:
            plain_symbol = symbol.split("|", 1)[1] if "|" in symbol else symbol
            try:
                result = self._run_symbol(db, plain_symbol)
                summary[plain_symbol] = result
            except Exception as exc:
                logger.exception("Inference failed for %s: %s", plain_symbol, exc)
                summary[plain_symbol] = {"error": str(exc)}
        return summary

    def run_intraday_inference(
        self,
        db: Session,
        symbols: list[str],
        interval: str,
        horizon: int | None = None,
    ) -> dict:
        interval = normalize_interval(interval)
        if interval != "1minute":
            raise ValueError(f"Unsupported intraday interval={interval}")
        bars = horizon if horizon is not None else 375
        summary: dict[str, dict] = {}
        for symbol in symbols:
            plain_symbol = symbol.split("|", 1)[1] if "|" in symbol else symbol
            try:
                out = self._run_intraday_symbol(db, plain_symbol, interval=interval, horizon=bars)
                summary[plain_symbol] = out
            except Exception as exc:
                logger.exception("Intraday inference failed for %s %s: %s", plain_symbol, interval, exc)
                summary[plain_symbol] = {"error": str(exc)}
        return summary

    def _run_symbol(self, db: Session, symbol: str) -> dict:
        used_fallback = False
        model_source = "xgboost_v1"
        try:
            raw_out = predict_meta_v3_symbol(db, symbol)
            model_source = "meta_v3"
            conf = compute_confidence_v3(
                calibrated_direction_prob=float(raw_out.get("direction_prob_calibrated", 0.5)),
                interval_coverage=float(raw_out.get("interval_coverage", 0.0)),
                interval_width_pct=float(raw_out.get("interval_width_pct", 1.0)),
                staleness_minutes=self._staleness_minutes(db),
            )
            raw_out["confidence_score"] = float(conf["confidence_score"])
            raw_out["confidence_bucket"] = conf["confidence_bucket"]
            raw_out["confidence_components"] = conf["components"]
        except Exception:
            used_fallback = True
            try:
                raw_out = predict_meta_symbol(db, symbol)
                model_source = "meta_model_v2"
            except Exception:
                raw_out = predict_symbol(db, symbol)
        out = combine_predictions(raw_out)
        out["meta"]["model_source"] = model_source
        out["meta"]["fallback_used"] = used_fallback
        if used_fallback:
            downgraded = max(0.0, min(1.0, float(out["confidence"]) * 0.85))
            out["confidence"] = downgraded
            out["meta"]["confidence_score"] = downgraded
            existing_bucket = str(out["meta"].get("confidence_bucket") or _bucket_from_score(downgraded))
            out["meta"]["confidence_bucket"] = _downgrade_bucket(existing_bucket)
            out["meta"]["model_family"] = model_source
            out["meta"]["calibration_version"] = None
            if "direction_prob_calibrated" not in out["meta"]:
                out["meta"]["direction_prob_calibrated"] = float(out["meta"].get("direction_prob", 0.5))
        latest_feature_date = db.scalar(
            select(FeaturesDaily.session_date)
            .where(FeaturesDaily.symbol == symbol)
            .order_by(FeaturesDaily.session_date.desc())
            .limit(1)
        )
        base_date = latest_feature_date or datetime.now(IST_ZONE).date()
        target_date = next_trading_day(base_date)
        cutoff = latest_feature_cutoff_ist()
        existing = db.scalar(
            select(PredictionsDaily.id).where(
                and_(
                    PredictionsDaily.symbol == symbol,
                    PredictionsDaily.interval == "day",
                    PredictionsDaily.target_session_date == target_date,
                    PredictionsDaily.model_version == out["model_version"],
                )
            )
        )
        if not existing:
            db.add(
                PredictionsDaily(
                    symbol=symbol,
                    interval="day",
                    target_session_date=target_date,
                    pred_open=out["pred_open"],
                    pred_high=out["pred_high"],
                    pred_low=out["pred_low"],
                    pred_close=out["pred_close"],
                    direction=out["direction"],
                    confidence=out["confidence"],
                    model_version=out["model_version"],
                    feature_cutoff_ist=cutoff,
                    metadata_json=out["meta"],
                )
            )
            db.commit()
        return out

    def _run_intraday_symbol(self, db: Session, symbol: str, interval: str, horizon: int) -> dict:
        model_out = predict_intraday_horizon(db=db, symbol=symbol, interval=interval, horizon=horizon)
        model_version = str(model_out.get("model_version") or f"intraday_v1_{interval}")
        preds = model_out.get("predictions", [])
        source_candle_ts = model_out.get("source_candle_ts")
        if source_candle_ts is not None and getattr(source_candle_ts, "tzinfo", None) is None:
            source_candle_ts = source_candle_ts.replace(tzinfo=IST_ZONE)
        if not preds:
            return {"model_version": model_version, "inserted": 0, "horizon": horizon}
        cutoff = latest_feature_cutoff_ist()
        inserted = 0
        for p in preds:
            target_ts = p.target_ts if p.target_ts.tzinfo is not None else p.target_ts.replace(tzinfo=IST_ZONE)
            exists = db.scalar(
                select(PredictionsIntraday.id).where(
                    and_(
                        PredictionsIntraday.symbol == symbol,
                        PredictionsIntraday.interval == interval,
                        PredictionsIntraday.target_ts == target_ts,
                        PredictionsIntraday.model_version == model_version,
                    )
                )
            )
            if exists:
                continue
            db.add(
                PredictionsIntraday(
                    symbol=symbol,
                    interval=interval,
                    target_ts=target_ts,
                    pred_open=float(p.pred_open),
                    pred_high=float(p.pred_high),
                    pred_low=float(p.pred_low),
                    pred_close=float(p.pred_close),
                    direction=p.direction,
                    confidence=float(p.confidence),
                    model_version=model_version,
                    feature_cutoff_ist=cutoff,
                    metadata_json={
                        "model_family": "intraday_v1",
                        "direction_prob_up": float(p.direction_prob_up),
                        "direction_prob_down": float(p.direction_prob_down),
                        "direction_prob_flat": float(p.direction_prob_flat),
                        "direction_prob_calibrated": float(p.direction_prob_up),
                        "confidence_score": float(p.confidence),
                        "confidence_bucket": _bucket_from_score(float(p.confidence)),
                        "source_candle_ts": source_candle_ts.isoformat() if source_candle_ts is not None else None,
                        "source_close": model_out.get("source_close"),
                        "prediction_horizon": int(horizon),
                    },
                )
            )
            inserted += 1
        if inserted:
            db.commit()
        return {"model_version": model_version, "inserted": inserted, "horizon": horizon}

    def run_historical_daily_signals(self, db: Session, symbols: list[str], limit: int = 2000) -> dict:
        summary: dict[str, dict] = {}
        for symbol in symbols:
            plain_symbol = symbol.split("|", 1)[1] if "|" in symbol else symbol
            try:
                summary[plain_symbol] = self._backfill_symbol_daily_signals(db, plain_symbol, limit=limit)
            except Exception as exc:
                logger.exception("Historical signal generation failed for %s: %s", plain_symbol, exc)
                summary[plain_symbol] = {"inserted": 0, "error": str(exc)}
        return summary

    def _backfill_symbol_daily_signals(self, db: Session, symbol: str, limit: int = 2000) -> dict:
        feature_df = load_feature_frame(db, symbol)
        if feature_df.empty:
            return {"inserted": 0, "rows_scored": 0}

        feature_df = feature_df.sort_values("session_date").tail(limit)
        pred_df = predict_from_feature_frame(db, symbol, feature_df)
        if pred_df.empty:
            return {"inserted": 0, "rows_scored": 0}

        model_version = str(pred_df["model_version"].iloc[-1])
        existing_target_dates = set(
            db.scalars(
                select(PredictionsDaily.target_session_date).where(
                    and_(
                        PredictionsDaily.symbol == symbol,
                        PredictionsDaily.interval == "day",
                        PredictionsDaily.model_version == model_version,
                    )
                )
            ).all()
        )

        direction_map = {-1: DIRECTION_SELL, 0: DIRECTION_HOLD, 1: DIRECTION_BUY}
        inserted = 0
        for _, row in pred_df.iterrows():
            source_session_raw = row["session_date"]
            source_session = (
                source_session_raw.date() if hasattr(source_session_raw, "date") else source_session_raw
            )
            target_date = next_trading_day(source_session)
            if target_date in existing_target_dates:
                continue
            direction_code = int(row.get("direction_code", 0))
            direction_prob = float(row.get("direction_prob", 0.5))
            confidence = compute_confidence(direction_prob=direction_prob, metrics={})
            feature_cutoff = datetime.combine(source_session, time(15, 30), tzinfo=IST_ZONE)

            db.add(
                PredictionsDaily(
                    symbol=symbol,
                    interval="day",
                    target_session_date=target_date,
                    pred_open=float(row["pred_open"]),
                    pred_high=float(row["pred_high"]),
                    pred_low=float(row["pred_low"]),
                    pred_close=float(row["pred_close"]),
                    direction=direction_map.get(direction_code, DIRECTION_HOLD),
                    confidence=confidence,
                    model_version=model_version,
                    feature_cutoff_ist=feature_cutoff,
                    metadata_json={
                        "direction_code": direction_code,
                        "direction_prob": direction_prob,
                        "mode": "historical_signal",
                    },
                )
            )
            existing_target_dates.add(target_date)
            inserted += 1

        if inserted:
            db.commit()

        logger.info(
            "Backfilled daily signals for %s inserted=%s scored=%s model=%s",
            symbol,
            inserted,
            len(pred_df),
            model_version,
        )
        return {"inserted": inserted, "rows_scored": int(len(pred_df)), "model_version": model_version}

    def run_daily_audit(self, db: Session, symbol: str, session_date) -> dict:
        pred = db.scalar(
            select(PredictionsDaily)
            .where(
                and_(
                    PredictionsDaily.symbol == symbol,
                    PredictionsDaily.interval == "day",
                    PredictionsDaily.target_session_date == session_date,
                )
            )
            .order_by(PredictionsDaily.generated_at.desc())
        )
        actual = db.scalar(
            select(RawCandle)
            .where(
                and_(
                    RawCandle.interval == "day",
                    RawCandle.instrument_key.like(f"%{symbol}%"),
                    func.date(RawCandle.ts) == session_date,
                )
            )
            .order_by(RawCandle.ts.asc())
        )
        if pred is None or actual is None:
            return {"status": "missing"}
        err_close = abs(pred.pred_close - actual.close)
        return {
            "symbol": symbol,
            "session_date": str(session_date),
            "pred_close": pred.pred_close,
            "actual_close": actual.close,
            "abs_err_close": err_close,
            "direction": pred.direction,
            "confidence": pred.confidence,
        }
