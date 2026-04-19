"""Execution tasks for Celery distributed processing."""
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


@app.task(
    bind=True,
    base=DatabaseTask,
    name="task_queue.tasks.execution_tasks.execute_signal_task",
)
def execute_signal_task(self, symbol: str, signal_action: str, confidence: float):
    """Execute a trading signal."""
    from execution_engine.engine import ExecutionEngine
    from utils.config import get_settings

    settings = get_settings()
    engine = ExecutionEngine(settings=settings, session_factory=lambda: self.db)

    try:
        result = engine.execute_signal(symbol=symbol, action=signal_action, confidence=confidence)
        logger.info(f"Executed signal for {symbol}: {result}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception(f"Failed to execute signal for {symbol}")
        return {"status": "error", "error": str(e)}


@app.task(
    bind=True,
    base=DatabaseTask,
    name="task_queue.tasks.execution_tasks.monitor_positions_task",
)
def monitor_positions_task(self):
    """Monitor open positions and update stop losses."""
    from execution_engine.engine import ExecutionEngine
    from utils.config import get_settings

    settings = get_settings()
    engine = ExecutionEngine(settings=settings, session_factory=lambda: self.db)

    try:
        updates = engine.monitor_and_update_positions()
        logger.info(f"Monitored positions: {updates}")
        return {"status": "success", "updates": updates}
    except Exception as e:
        logger.exception("Failed to monitor positions")
        return {"status": "error", "error": str(e)}


@app.task(
    bind=True,
    base=DatabaseTask,
    name="task_queue.tasks.execution_tasks.close_position_task",
)
def close_position_task(self, position_id: int, reason: str = "manual"):
    """Close a specific position."""
    from execution_engine.engine import ExecutionEngine
    from utils.config import get_settings

    settings = get_settings()
    engine = ExecutionEngine(settings=settings, session_factory=lambda: self.db)

    try:
        result = engine.close_position(position_id=position_id, reason=reason)
        logger.info(f"Closed position {position_id}: {result}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception(f"Failed to close position {position_id}")
        return {"status": "error", "error": str(e)}
