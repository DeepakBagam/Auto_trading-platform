try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import time
from datetime import datetime, timedelta

from backend.data_layer.collectors.upstox_collector import UpstoxCollector
from backend.db.connection import SessionLocal
from backend.db.init_db import init_db
from backend.utils.config import get_settings
from backend.utils.constants import IST_ZONE
from backend.utils.logger import setup_logging


def next_run(hour_ist: int) -> datetime:
    now = datetime.now(IST_ZONE)
    run_at = now.replace(hour=hour_ist, minute=5, second=0, microsecond=0)
    if run_at <= now:
        run_at = run_at + timedelta(days=1)
    return run_at


def main() -> None:
    setup_logging("daily_maintenance")
    init_db()
    settings = get_settings()
    collector = UpstoxCollector()
    while True:
        run_at = next_run(int(settings.data_ingestion_daily_hour_ist))
        time.sleep(max(60, int((run_at - datetime.now(IST_ZONE)).total_seconds())))
        db = SessionLocal()
        try:
            collector.ingest_historical_batch(db, days_back=7)
        finally:
            db.close()


if __name__ == "__main__":
    main()
