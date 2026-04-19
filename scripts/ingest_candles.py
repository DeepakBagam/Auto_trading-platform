try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import argparse

from data_layer.collectors.upstox_collector import UpstoxCollector
from db.connection import SessionLocal
from utils.logger import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Ingest Upstox candles")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--days-back", type=int, default=180, help="Used in quick mode")
    parser.add_argument(
        "--one-minute-days",
        type=int,
        default=730,
        help="Used in full mode for 1-minute interval backfill",
    )
    args = parser.parse_args()
    db = SessionLocal()
    try:
        collector = UpstoxCollector()
        if args.mode == "full":
            summary = collector.ingest_historical_full(db, one_minute_days=args.one_minute_days)
        else:
            summary = collector.ingest_historical_batch(db, days_back=args.days_back)
        print(summary)
    finally:
        db.close()


if __name__ == "__main__":
    main()
