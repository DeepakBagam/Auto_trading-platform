try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import os
import tempfile
import time
from pathlib import Path
from typing import TextIO

from backend.db.connection import SessionLocal
from backend.db.init_db import init_db
from backend.execution_engine.engine import IntradayOptionsExecutionEngine
from backend.utils.app_state import apply_runtime_execution_settings
from backend.utils.config import get_settings
from backend.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def _acquire_single_instance_lock() -> TextIO:
    lock_path = Path(
        os.getenv(
            "EXECUTION_LOOP_LOCK_FILE",
            str(Path(tempfile.gettempdir()) / "auto-trading-execution.lock"),
        )
    )
    handle = lock_path.open("a+", encoding="ascii")
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        if not handle.read(1):
            handle.write(" ")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            handle.close()
            raise RuntimeError(f"Another execution loop already holds {lock_path}") from exc
    else:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(f"Another execution loop already holds {lock_path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main() -> None:
    setup_logging("execution_loop")
    try:
        instance_lock = _acquire_single_instance_lock()
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    init_db()
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
    try:
        while True:
            db = SessionLocal()
            try:
                apply_runtime_execution_settings(db, settings)
                out = engine.run_once(db)
                logger.info("Execution loop tick: %s", out)
            except Exception as exc:
                logger.exception("Execution loop error: %s", exc)
            finally:
                db.close()
            time.sleep(max(1, int(settings.execution_poll_seconds)))
    finally:
        instance_lock.close()


if __name__ == "__main__":
    main()
