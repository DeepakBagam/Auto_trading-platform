try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import argparse
from datetime import date, datetime, time, timedelta

import pandas as pd
from sqlalchemy import and_, select

from data_layer.collectors.upstox_collector import UpstoxCollector
from db.connection import SessionLocal
from db.models import PredictionsDaily, PredictionsIntraday, RawCandle
from prediction_engine.orchestrator import PredictionOrchestrator
from utils.config import get_settings
from utils.constants import IST_ZONE, SUPPORTED_INTERVALS
from utils.logger import setup_logging


def _priority(model_version: str) -> int:
    if str(model_version).startswith("meta_v3_"):
        return 3
    if str(model_version).startswith("meta_"):
        return 2
    return 1


def _fetch_candles_for_date(
    collector: UpstoxCollector,
    instrument_key: str,
    interval: str,
    target_date: date,
    now_ist: datetime,
) -> list:
    # User rule: after 4 PM for "today", use intraday endpoint for intraday intervals.
    if interval == "1minute" and target_date == now_ist.date() and now_ist.time() >= time(16, 0):
        records = collector.fetch_intraday_candles(instrument_key=instrument_key, interval=interval)
        return [r for r in records if r.ts.date() == target_date]
    return collector.fetch_historical_candles(
        instrument_key=instrument_key,
        interval=interval,
        from_date=target_date,
        to_date=target_date,
    )


def _crosscheck_daily(db, symbol: str, dates: list[date]) -> dict:
    pred_rows = (
        db.execute(
            select(PredictionsDaily)
            .where(
                and_(
                    PredictionsDaily.symbol == symbol,
                    PredictionsDaily.interval == "day",
                    PredictionsDaily.target_session_date.in_(dates),
                )
            )
            .order_by(PredictionsDaily.target_session_date.asc(), PredictionsDaily.generated_at.desc())
        )
        .scalars()
        .all()
    )
    best_by_date = {}
    for row in pred_rows:
        prev = best_by_date.get(row.target_session_date)
        if prev is None:
            best_by_date[row.target_session_date] = row
            continue
        if _priority(row.model_version) > _priority(prev.model_version):
            best_by_date[row.target_session_date] = row
        elif _priority(row.model_version) == _priority(prev.model_version) and row.generated_at > prev.generated_at:
            best_by_date[row.target_session_date] = row

    out = []
    for d in dates:
        pred = best_by_date.get(d)
        candle = db.scalar(
            select(RawCandle)
            .where(
                and_(
                    RawCandle.interval == "day",
                    RawCandle.instrument_key.like(f"%|{symbol}"),
                    RawCandle.ts >= datetime.combine(d, time(0, 0), tzinfo=IST_ZONE),
                    RawCandle.ts < datetime.combine(d + timedelta(days=1), time(0, 0), tzinfo=IST_ZONE),
                )
            )
            .order_by(RawCandle.ts.desc())
        )
        if pred is None or candle is None:
            out.append(
                {
                    "date": str(d),
                    "status": "missing_prediction_or_actual",
                }
            )
            continue
        err = abs(float(pred.pred_close) - float(candle.close))
        mape = err / (abs(float(candle.close)) + 1e-9)
        out.append(
            {
                "date": str(d),
                "pred_close": float(pred.pred_close),
                "actual_close": float(candle.close),
                "abs_err_close": float(err),
                "mape_close_pct": float(mape * 100.0),
                "model_version": str(pred.model_version),
            }
        )
    return {"symbol": symbol, "interval": "day", "rows": out}


def _crosscheck_intraday(db, symbol: str, interval: str, target_date: date) -> dict:
    pred_rows = (
        db.execute(
            select(PredictionsIntraday)
            .where(
                and_(
                    PredictionsIntraday.symbol == symbol,
                    PredictionsIntraday.interval == interval,
                )
            )
            .order_by(PredictionsIntraday.target_ts.asc(), PredictionsIntraday.generated_at.desc())
        )
        .scalars()
        .all()
    )
    if not pred_rows:
        return {"symbol": symbol, "interval": interval, "date": str(target_date), "overlap": 0, "status": "no_predictions"}

    by_ts = {}
    for row in pred_rows:
        ts = row.target_ts.date()
        if ts != target_date:
            continue
        prev = by_ts.get(row.target_ts)
        if prev is None or row.generated_at > prev.generated_at:
            by_ts[row.target_ts] = row
    if not by_ts:
        return {"symbol": symbol, "interval": interval, "date": str(target_date), "overlap": 0, "status": "no_predictions_for_date"}

    actual_rows = (
        db.execute(
            select(RawCandle)
            .where(
                and_(
                    RawCandle.interval == interval,
                    RawCandle.instrument_key.like(f"%|{symbol}"),
                )
            )
            .order_by(RawCandle.ts.asc())
        )
        .scalars()
        .all()
    )
    actual_by_ts = {r.ts: r for r in actual_rows if r.ts.date() == target_date}
    overlap = []
    for ts, pred in by_ts.items():
        act = actual_by_ts.get(ts)
        if act is None:
            continue
        overlap.append(
            {
                "ts": ts,
                "abs_err_close": abs(float(pred.pred_close) - float(act.close)),
                "pred_close": float(pred.pred_close),
                "actual_close": float(act.close),
            }
        )
    if not overlap:
        return {"symbol": symbol, "interval": interval, "date": str(target_date), "overlap": 0, "status": "no_overlap"}

    df = pd.DataFrame(overlap)
    mae = float(df["abs_err_close"].mean())
    mape = float((df["abs_err_close"] / (df["actual_close"].abs() + 1e-9)).mean() * 100.0)
    return {
        "symbol": symbol,
        "interval": interval,
        "date": str(target_date),
        "overlap": int(len(df)),
        "mae_close": mae,
        "mape_close_pct": mape,
        "status": "ok",
    }


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Fetch Upstox candles for date range and cross-check predictions.")
    parser.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--run-predict-now",
        action="store_true",
        help="Run current intraday prediction after fetch (for latest context only).",
    )
    args = parser.parse_args()

    from_date = date.fromisoformat(args.from_date)
    to_date = date.fromisoformat(args.to_date)
    if from_date > to_date:
        raise ValueError("from-date must be <= to-date")

    settings = get_settings()
    if not settings.upstox_access_token:
        raise RuntimeError("UPSTOX_ACCESS_TOKEN is empty. Set token first, then rerun.")
    if not settings.instrument_keys:
        raise RuntimeError("No instruments configured. Set UPSTOX_INSTRUMENT_KEYS.")

    collector = UpstoxCollector()
    db = SessionLocal()
    try:
        now_ist = datetime.now(IST_ZONE)
        fetch_summary: dict[str, dict] = {}
        probe = from_date
        while probe <= to_date:
            for instrument_key in settings.instrument_keys:
                for interval in SUPPORTED_INTERVALS:
                    key = f"{probe.isoformat()}:{instrument_key}:{interval}"
                    try:
                        records = _fetch_candles_for_date(
                            collector=collector,
                            instrument_key=instrument_key,
                            interval=interval,
                            target_date=probe,
                            now_ist=now_ist,
                        )
                        inserted = collector.persist(db, records)
                        fetch_summary[key] = {"records_seen": len(records), "inserted": inserted}
                    except Exception as exc:
                        fetch_summary[key] = {"error": str(exc)}
            probe += timedelta(days=1)

        predict_summary = None
        if args.run_predict_now:
            orchestrator = PredictionOrchestrator()
            predict_summary = {
                "1minute": orchestrator.run_intraday_inference(db, settings.instrument_keys, interval="1minute"),
            }

        symbols = [x.split("|", 1)[1] if "|" in x else x for x in settings.instrument_keys]
        dates = []
        probe = from_date
        while probe <= to_date:
            dates.append(probe)
            probe += timedelta(days=1)

        report = {"intraday": []}
        for symbol in symbols:
            for d in dates:
                report["intraday"].append(_crosscheck_intraday(db, symbol, "1minute", d))

        print({"fetch_summary": fetch_summary, "predict_summary": predict_summary, "crosscheck": report})
    finally:
        db.close()


if __name__ == "__main__":
    main()
