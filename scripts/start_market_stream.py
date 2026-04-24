try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from data_layer.collectors.upstox_collector import UpstoxCollector
from data_layer.streamers.upstox_market_stream import UpstoxMarketStream
from db.connection import SessionLocal
from utils.config import get_settings
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    setup_logging("market_stream")
    settings = get_settings()
    if settings.history_bootstrap_on_start:
        logger.info("History bootstrap is enabled on startup; checking retained window before live stream connects.")
        db = SessionLocal()
        try:
            UpstoxCollector().ensure_history_window(db)
        finally:
            db.close()
    else:
        logger.info("History bootstrap on startup is disabled; connecting live market stream immediately.")
    UpstoxMarketStream().run_forever()


if __name__ == "__main__":
    main()
