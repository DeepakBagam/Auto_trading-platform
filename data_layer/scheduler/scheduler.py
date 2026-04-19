from __future__ import annotations

from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session

from data_layer.collectors.news_collector import NewsCollector
from data_layer.collectors.upstox_collector import UpstoxCollector
from db.connection import SessionLocal
from execution_engine.engine import IntradayOptionsExecutionEngine
from feature_engine.feature_builder import build_daily_features
from feature_engine.labels import build_daily_labels
from models.meta_v3.train import train_all_symbols as train_meta_v3_all
from prediction_engine.orchestrator import PredictionOrchestrator
from utils.email_reports import send_daily_summary_report, send_gap_report
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)


def _with_db(fn):
    def wrapped():
        db: Session = SessionLocal()
        try:
            fn(db)
        finally:
            db.close()

    return wrapped


def _ingest_news(db: Session) -> None:
    inserted = NewsCollector().run_once(db)
    logger.info("Scheduler news job inserted=%s", inserted)


def _ingest_candles(db: Session) -> None:
    settings = get_settings()
    if settings.market_data_mode.lower() == "websocket":
        logger.info("Skipping candle polling because MARKET_DATA_MODE=websocket")
        return
    summary = UpstoxCollector().ingest_historical_batch(db, days_back=3)
    logger.info("Scheduler candle job summary=%s", summary)


def _eod_pipeline(db: Session) -> None:
    settings = get_settings()
    symbols = settings.instrument_keys
    if not symbols:
        logger.warning("No instruments configured; skipping EOD pipeline")
        return
    build_daily_features(db, symbols)
    build_daily_labels(db, symbols)
    orchestrator = PredictionOrchestrator()
    orchestrator.run_daily_inference(db, symbols)


def _weekly_retrain(db: Session) -> None:
    settings = get_settings()
    symbols = [s.split("|", 1)[1] if "|" in s else s for s in settings.instrument_keys]
    if not symbols:
        return
    summary = train_meta_v3_all(db, symbols)
    logger.info("Weekly meta_v3 retrain summary=%s", summary)


def _execution_cycle(db: Session) -> None:
    settings = get_settings()
    if not settings.execution_enabled:
        return
    out = IntradayOptionsExecutionEngine(settings=settings).run_once(db)
    logger.info("Execution cycle summary=%s", out)


def _morning_gap_report(db: Session) -> None:
    sent = send_gap_report(db, report_date=datetime.now(IST_ZONE).date(), settings=get_settings())
    logger.info("Morning gap report sent=%s", sent)


def _daily_summary_report(db: Session) -> None:
    sent = send_daily_summary_report(
        db,
        trade_date=datetime.now(IST_ZONE).date(),
        settings=get_settings(),
    )
    logger.info("Daily summary report sent=%s", sent)


def start_scheduler() -> None:
    settings = get_settings()
    scheduler = BlockingScheduler(timezone=settings.timezone)
    scheduler.add_job(
        _with_db(_ingest_news),
        IntervalTrigger(minutes=settings.news_poll_minutes),
        id="news_poll_job",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _with_db(_ingest_candles),
        IntervalTrigger(minutes=settings.candle_poll_minutes),
        id="candle_poll_job",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _with_db(_eod_pipeline),
        CronTrigger(hour=15, minute=40),
        id="eod_pipeline_job",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _with_db(_weekly_retrain),
        CronTrigger(day_of_week="sun", hour=10, minute=0),
        id="weekly_retrain_job",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _with_db(_execution_cycle),
        IntervalTrigger(seconds=max(5, int(settings.execution_poll_seconds))),
        id="execution_cycle_job",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _with_db(_morning_gap_report),
        CronTrigger(day_of_week="mon-fri", hour=9, minute=15, second=0),
        id="morning_gap_report_job",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        _with_db(_daily_summary_report),
        CronTrigger(day_of_week="mon-fri", hour=15, minute=31),
        id="daily_summary_report_job",
        max_instances=1,
        coalesce=True,
    )
    logger.info("Scheduler started")
    scheduler.start()


if __name__ == "__main__":
    start_scheduler()
