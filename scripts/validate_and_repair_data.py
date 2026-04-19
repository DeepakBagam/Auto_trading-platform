try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import argparse
import json
from datetime import datetime, time

from sqlalchemy import and_, func, or_, select, text

from data_layer.collectors.upstox_collector import UpstoxCollector
from db.connection import SessionLocal
from db.models import RawCandle
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.logger import setup_logging

EXPECTED_BARS = {"1minute": 375, "30minute": 13}


def _integrity_counts(db, instrument_key: str, interval: str) -> dict:
    row = db.execute(
        text(
            """
            SELECT
              COALESCE(SUM(CASE WHEN high < low OR high < open OR high < close OR low > open OR low > close THEN 1 ELSE 0 END), 0) AS bad_ohlc,
              COALESCE(SUM(CASE WHEN open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 THEN 1 ELSE 0 END), 0) AS non_positive,
              COALESCE(SUM(CASE WHEN volume < 0 THEN 1 ELSE 0 END), 0) AS neg_volume
            FROM raw_candles
            WHERE instrument_key = :instrument_key AND interval = :interval
            """
        ),
        {"instrument_key": instrument_key, "interval": interval},
    ).one()
    return {
        "bad_ohlc": int(row.bad_ohlc or 0),
        "non_positive": int(row.non_positive or 0),
        "neg_volume": int(row.neg_volume or 0),
    }


def _coverage(db, instrument_key: str, interval: str) -> dict:
    row = db.execute(
        text(
            """
            SELECT COUNT(*) AS count_rows, MIN(ts) AS min_ts, MAX(ts) AS max_ts
            FROM raw_candles
            WHERE instrument_key = :instrument_key AND interval = :interval
            """
        ),
        {"instrument_key": instrument_key, "interval": interval},
    ).one()
    return {
        "count": int(row.count_rows or 0),
        "min_ts": str(row.min_ts) if row.min_ts is not None else None,
        "max_ts": str(row.max_ts) if row.max_ts is not None else None,
    }


def _recent_completeness(db, instrument_key: str, lookback_sessions: int) -> list[dict]:
    one_min_rows = db.execute(
        text(
            """
            SELECT DATE(ts) AS d, COUNT(*) AS c
            FROM raw_candles
            WHERE instrument_key = :instrument_key AND interval = '1minute'
            GROUP BY DATE(ts)
            ORDER BY d DESC
            LIMIT :lookback
            """
        ),
        {"instrument_key": instrument_key, "lookback": lookback_sessions},
    ).all()
    out: list[dict] = []
    for row in one_min_rows:
        day = str(row.d)
        c1 = int(row.c or 0)
        c30_row = db.execute(
            text(
                """
                SELECT COUNT(*) AS c
                FROM raw_candles
                WHERE instrument_key = :instrument_key
                  AND interval = '30minute'
                  AND DATE(ts) = :session_date
                """
            ),
            {"instrument_key": instrument_key, "session_date": day},
        ).one()
        c30 = int(c30_row.c or 0)
        out.append(
            {
                "session_date": day,
                "count_1minute": c1,
                "count_30minute": c30,
                "expected_1minute": EXPECTED_BARS["1minute"],
                "expected_30minute": EXPECTED_BARS["30minute"],
                "ok_1minute": c1 == EXPECTED_BARS["1minute"],
                "ok_30minute": c30 == EXPECTED_BARS["30minute"],
            }
        )
    return out


def _day_quality(db, instrument_key: str) -> dict:
    non_midnight = db.execute(
        text(
            """
            SELECT COUNT(*) AS c
            FROM raw_candles
            WHERE instrument_key = :instrument_key
              AND interval = 'day'
              AND TIME(ts) <> '00:00:00'
            """
        ),
        {"instrument_key": instrument_key},
    ).scalar_one()
    duplicate_dates = db.execute(
        text(
            """
            SELECT COUNT(*) AS c
            FROM (
              SELECT DATE(ts) AS d, COUNT(*) AS cnt
              FROM raw_candles
              WHERE instrument_key = :instrument_key
                AND interval = 'day'
              GROUP BY DATE(ts)
              HAVING COUNT(*) > 1
            ) x
            """
        ),
        {"instrument_key": instrument_key},
    ).scalar_one()
    return {
        "non_midnight": int(non_midnight or 0),
        "duplicate_dates": int(duplicate_dates or 0),
    }


def _repair_day_series(db, instrument_key: str) -> dict:
    rows = (
        db.execute(
            select(RawCandle)
            .where(and_(RawCandle.instrument_key == instrument_key, RawCandle.interval == "day"))
            .order_by(RawCandle.ts.desc(), RawCandle.id.desc())
        )
        .scalars()
        .all()
    )
    seen_dates: set = set()
    winners: list[RawCandle] = []
    deleted = 0
    normalized_ts = 0
    normalized_ohlc = 0

    for row in rows:
        day = row.ts.date()
        if day in seen_dates:
            db.delete(row)
            deleted += 1
            continue
        seen_dates.add(day)
        winners.append(row)

    if deleted:
        db.flush()

    for row in winners:
        day_midnight = datetime.combine(row.ts.date(), time(0, 0), tzinfo=IST_ZONE)
        if row.ts != day_midnight:
            row.ts = day_midnight
            normalized_ts += 1
        new_high = max(float(row.high), float(row.open), float(row.close), float(row.low))
        new_low = min(float(row.low), float(row.open), float(row.close), float(row.high))
        if new_high != float(row.high) or new_low != float(row.low):
            row.high = new_high
            row.low = new_low
            normalized_ohlc += 1
    return {
        "deleted_duplicate_rows": deleted,
        "normalized_day_ts_rows": normalized_ts,
        "normalized_day_ohlc_rows": normalized_ohlc,
    }


def _repair_non_day_ohlc(db, instrument_key: str) -> int:
    bad_rows = (
        db.execute(
            select(RawCandle).where(
                and_(
                    RawCandle.instrument_key == instrument_key,
                    RawCandle.interval != "day",
                    or_(
                        RawCandle.high < RawCandle.low,
                        RawCandle.high < RawCandle.open,
                        RawCandle.high < RawCandle.close,
                        RawCandle.low > RawCandle.open,
                        RawCandle.low > RawCandle.close,
                    ),
                )
            )
        )
        .scalars()
        .all()
    )
    changed = 0
    for row in bad_rows:
        new_high = max(float(row.high), float(row.open), float(row.close), float(row.low))
        new_low = min(float(row.low), float(row.open), float(row.close), float(row.high))
        if new_high != float(row.high) or new_low != float(row.low):
            row.high = new_high
            row.low = new_low
            changed += 1
    return changed


def _build_report(db, instrument_keys: list[str], lookback_sessions: int) -> dict:
    report = {
        "generated_at_ist": datetime.now(IST_ZONE).isoformat(),
        "today_ist": datetime.now(IST_ZONE).date().isoformat(),
        "symbols": [],
    }
    for instrument_key in instrument_keys:
        symbol_report = {
            "instrument_key": instrument_key,
            "coverage": {},
            "integrity": {},
            "recent_completeness": _recent_completeness(db, instrument_key, lookback_sessions),
            "day_quality": _day_quality(db, instrument_key),
        }
        for interval in ("1minute", "30minute", "day"):
            symbol_report["coverage"][interval] = _coverage(db, instrument_key, interval)
            symbol_report["integrity"][interval] = _integrity_counts(db, instrument_key, interval)
        report["symbols"].append(symbol_report)
    return report


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Validate and optionally repair market data quality.")
    parser.add_argument("--fetch-days-back", type=int, default=14, help="Backfill window before validation.")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetch/backfill step.")
    parser.add_argument("--repair", action="store_true", help="Apply data repairs for day-series and OHLC anomalies.")
    parser.add_argument("--lookback-sessions", type=int, default=7, help="Session count for completeness checks.")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.instrument_keys:
        raise RuntimeError("No instruments configured. Set UPSTOX_INSTRUMENT_KEYS first.")

    db = SessionLocal()
    try:
        fetch_summary = None
        if not args.skip_fetch:
            collector = UpstoxCollector()
            fetch_summary = collector.ingest_historical_batch(db=db, days_back=args.fetch_days_back)

        repair_summary = None
        if args.repair:
            repair_summary = {"day_series": {}, "intraday_ohlc_rows": {}}
            for instrument_key in settings.instrument_keys:
                repair_summary["day_series"][instrument_key] = _repair_day_series(db, instrument_key)
                repair_summary["intraday_ohlc_rows"][instrument_key] = _repair_non_day_ohlc(
                    db, instrument_key
                )
            db.commit()

        report = _build_report(db, settings.instrument_keys, args.lookback_sessions)
        print(json.dumps({"fetch_summary": fetch_summary, "repair_summary": repair_summary, "report": report}, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
