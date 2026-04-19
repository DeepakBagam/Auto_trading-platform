"""Data ingestion tasks for Celery distributed processing."""
from celery import Task
from celery.utils.log import get_task_logger

from task_queue.celery_app import app

logger = get_task_logger(__name__)


class DatabaseTask(Task):
    """Base task with database session management."""

    _db = None

    @property
    def db(self):
        if self._db is None:
            from db.connection import SessionLocal

            self._db = SessionLocal()
        return self._db

    def after_return(self, *args, **kwargs):
        if self._db is not None:
            self._db.close()
            self._db = None


@app.task(bind=True, base=DatabaseTask, name="task_queue.tasks.data_tasks.ingest_news_task")
def ingest_news_task(self):
    """Ingest news from all configured sources."""
    from data_layer.collectors.news_collector import NewsCollector
    from utils.config import get_settings

    settings = get_settings()
    collector = NewsCollector(settings=settings, session_factory=lambda: self.db)

    try:
        articles = collector.collect_all()
        logger.info(f"Ingested {len(articles)} news articles")
        return {"status": "success", "articles_count": len(articles)}
    except Exception as e:
        logger.exception("Failed to ingest news")
        return {"status": "error", "error": str(e)}


@app.task(bind=True, base=DatabaseTask, name="task_queue.tasks.data_tasks.ingest_candles_task")
def ingest_candles_task(self):
    """Ingest historical candles from Upstox."""
    from data_layer.collectors.upstox_collector import UpstoxCollector
    from utils.config import get_settings

    settings = get_settings()
    collector = UpstoxCollector(settings=settings, session_factory=lambda: self.db)

    try:
        total_candles = 0
        for instrument_key in settings.instrument_keys:
            candles = collector.fetch_historical_candles(
                instrument_key=instrument_key, interval="1minute", days_back=1
            )
            total_candles += len(candles)

        logger.info(f"Ingested {total_candles} candles")
        return {"status": "success", "candles_count": total_candles}
    except Exception as e:
        logger.exception("Failed to ingest candles")
        return {"status": "error", "error": str(e)}


@app.task(bind=True, base=DatabaseTask, name="task_queue.tasks.data_tasks.ingest_order_book_task")
def ingest_order_book_task(self, instrument_key: str):
    """Ingest order book snapshot for a specific instrument."""
    from data_layer.collectors.order_book_collector import OrderBookCollector
    from utils.config import get_settings

    settings = get_settings()
    collector = OrderBookCollector(settings=settings, session_factory=lambda: self.db)

    try:
        snapshot = collector.collect_snapshot(instrument_key)
        logger.info(f"Ingested order book for {instrument_key}")
        return {"status": "success", "instrument_key": instrument_key, "snapshot": snapshot}
    except Exception as e:
        logger.exception(f"Failed to ingest order book for {instrument_key}")
        return {"status": "error", "error": str(e)}


@app.task(bind=True, base=DatabaseTask, name="task_queue.tasks.data_tasks.ingest_option_chain_task")
def ingest_option_chain_task(self, symbol: str, expiry_date: str):
    """Ingest option chain data for a symbol and expiry."""
    from data_layer.collectors.upstox_option_chain import UpstoxOptionChainCollector
    from utils.config import get_settings

    settings = get_settings()
    collector = UpstoxOptionChainCollector(settings=settings, session_factory=lambda: self.db)

    try:
        options = collector.fetch_option_chain(symbol=symbol, expiry_date=expiry_date)
        logger.info(f"Ingested {len(options)} options for {symbol} expiry {expiry_date}")
        return {"status": "success", "symbol": symbol, "options_count": len(options)}
    except Exception as e:
        logger.exception(f"Failed to ingest option chain for {symbol}")
        return {"status": "error", "error": str(e)}
