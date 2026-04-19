try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import time

from db.connection import SessionLocal
from execution_engine.engine import IntradayOptionsExecutionEngine
from utils.config import get_settings
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    setup_logging("execution_loop")
    settings = get_settings()
    engine = IntradayOptionsExecutionEngine(settings=settings)
    logger.info(
        "Starting execution loop enabled=%s mode=%s symbols=%s interval=%s poll=%ss",
        settings.execution_enabled,
        settings.execution_mode,
        settings.execution_symbol_list,
        "1minute",
        settings.execution_poll_seconds,
    )
    while True:
        db = SessionLocal()
        try:
            out = engine.run_once(db)
            logger.info("Execution loop tick: %s", out)
        except Exception as exc:
            logger.exception("Execution loop error: %s", exc)
        finally:
            db.close()
        time.sleep(max(1, int(settings.execution_poll_seconds)))


if __name__ == "__main__":
    main()
