try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import argparse

from backend.data_layer.collectors.upstox_collector import UpstoxCollector
from backend.db.connection import SessionLocal
from backend.utils.logger import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Ingest Upstox candles")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--days-back", type=int, default=180, help="Used in quick mode")
    parser.add_argument(
        "--one-minute-days",
        type=int,
        default=730,
        help="Deprecated; full mode now uses Upstox availability start dates per interval",
    )
    parser.add_argument(
        "--interval",
        action="append",
        dest="intervals",
        help="Optional interval to backfill in full mode. Repeat for multiple values. Defaults to all Upstox V3 intervals.",
    )
    parser.add_argument(
        "--instrument",
        action="append",
        dest="instruments",
        help="Optional instrument key to backfill in full mode. Repeat for multiple values. Defaults to configured instruments.",
    )
    args = parser.parse_args()
    db = SessionLocal()
    try:
        collector = UpstoxCollector()
        if args.mode == "full":
            summary = collector.ingest_historical_full(
                db,
                one_minute_days=args.one_minute_days,
                intervals=args.intervals,
                instrument_keys=args.instruments,
            )
        else:
            summary = collector.ingest_historical_batch(db, days_back=args.days_back)
        print(summary)
    finally:
        db.close()


if __name__ == "__main__":
    main()
