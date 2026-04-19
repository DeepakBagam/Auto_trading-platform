from sqlalchemy import inspect, text

from db.connection import engine
from db.models import Base
from db.view_manager import create_symbol_interval_views
from utils.config import get_settings
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def init_db() -> None:
    settings = get_settings()
    Base.metadata.create_all(bind=engine)
    _migrate_predictions_interval_column()
    _migrate_option_quotes_columns()
    _migrate_execution_position_columns()
    _migrate_execution_order_columns()
    _ensure_indexes()
    if settings.database_url.startswith("postgresql"):
        with engine.begin() as conn:
            # Enables Timescale extension when available; harmless if already enabled.
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            except Exception:
                logger.warning(
                    "Timescale extension unavailable; running with standard PostgreSQL tables."
                )
    else:
        logger.info("Using SQLite mode (Timescale extension skipped).")
    create_symbol_interval_views(engine)
    logger.info("Database initialized")


def _migrate_predictions_interval_column() -> None:
    inspector = inspect(engine)
    if not inspector.has_table("predictions_daily"):
        return
    cols = {c["name"] for c in inspector.get_columns("predictions_daily")}
    if "interval" in cols:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE predictions_daily ADD COLUMN interval VARCHAR(32) DEFAULT 'day'"))
        conn.execute(text("UPDATE predictions_daily SET interval = 'day' WHERE interval IS NULL"))


def _migrate_option_quotes_columns() -> None:
    inspector = inspect(engine)
    if not inspector.has_table("option_quotes"):
        return

    cols = {c["name"] for c in inspector.get_columns("option_quotes")}
    additions = {
        "underlying_key": "VARCHAR(128)",
        "close_price": "FLOAT",
        "bid_qty": "FLOAT",
        "ask_qty": "FLOAT",
        "prev_oi": "FLOAT",
        "pop": "FLOAT",
        "pcr": "FLOAT",
        "underlying_spot_price": "FLOAT",
    }
    with engine.begin() as conn:
        for col_name, col_type in additions.items():
            if col_name in cols:
                continue
            conn.execute(text(f"ALTER TABLE option_quotes ADD COLUMN {col_name} {col_type}"))


def _ensure_table_columns(table_name: str, additions: dict[str, str]) -> None:
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        return
    cols = {c["name"] for c in inspector.get_columns(table_name)}
    with engine.begin() as conn:
        for col_name, col_type in additions.items():
            if col_name in cols:
                continue
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"))


def _migrate_execution_position_columns() -> None:
    _ensure_table_columns(
        "execution_positions",
        {
            "entry_premium": "FLOAT",
            "initial_sl": "FLOAT",
            "current_sl": "FLOAT",
            "peak_premium": "FLOAT",
            "tsl_active": "BOOLEAN DEFAULT 0",
            "target_premium": "FLOAT",
            "current_premium": "FLOAT",
            "exit_premium": "FLOAT",
            "realized_pnl": "FLOAT",
            "unrealized_pnl": "FLOAT",
            "ml_confidence": "FLOAT",
            "ai_score": "FLOAT",
            "pine_signal": "VARCHAR(16)",
            "consensus_reason": "TEXT",
            "entry_order_id": "VARCHAR(128)",
        },
    )


def _migrate_execution_order_columns() -> None:
    _ensure_table_columns(
        "execution_orders",
        {
            "strike_price": "FLOAT",
            "option_type": "VARCHAR(2)",
            "expiry_date": "DATE",
            "entry_premium": "FLOAT",
            "initial_sl": "FLOAT",
            "current_sl": "FLOAT",
            "target_premium": "FLOAT",
            "peak_premium": "FLOAT",
            "tsl_active": "BOOLEAN DEFAULT 0",
            "exit_premium": "FLOAT",
            "exit_reason": "VARCHAR(64)",
            "realized_pnl": "FLOAT",
            "unrealized_pnl": "FLOAT",
            "ml_confidence": "FLOAT",
            "ai_score": "FLOAT",
            "pine_signal": "VARCHAR(16)",
            "consensus_reason": "TEXT",
        },
    )


def _ensure_indexes() -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_raw_candles_instrument_interval_ts "
                "ON raw_candles (instrument_key, interval, ts)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_predictions_symbol_interval_target "
                "ON predictions_daily (symbol, interval, target_session_date)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_predictions_intraday_symbol_interval_ts "
                "ON predictions_intraday (symbol, interval, target_ts)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_oof_run_symbol_date "
                "ON oof_predictions (run_id, symbol, session_date)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_calibration_symbol_family_active "
                "ON calibration_registry (symbol, model_family, is_active)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_backtest_symbol_family_created "
                "ON backtest_runs (symbol, model_family, created_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_drift_symbol_family_metric "
                "ON drift_metrics (symbol, model_family, metric_name, computed_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_option_quotes_underlying_expiry_strike_type_ts "
                "ON option_quotes (underlying_symbol, expiry_date, strike, option_type, ts)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_option_signals_symbol_interval_expiry_generated "
                "ON option_trade_signals (symbol, interval, expiry_date, generated_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_execution_positions_trade_status_symbol "
                "ON execution_positions (trade_date, status, symbol)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_execution_orders_trade_symbol_kind "
                "ON execution_orders (trade_date, symbol, order_kind, created_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_execution_orders_position_status "
                "ON execution_orders (position_id, status, created_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_execution_signal_audit_trade_symbol "
                "ON execution_signal_audit (trade_date, symbol, interval, created_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_execution_external_signals_source_processed "
                "ON execution_external_signals (source, processed, signal_ts)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_execution_external_signal_consume "
                "ON execution_external_signals (symbol, interval, processed, signal_ts, created_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_daily_summary_pnl "
                "ON daily_summary (date, total_pnl, total_trades)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_signal_log_symbol_ts "
                "ON signal_log (symbol, interval, timestamp DESC)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_signal_log_trade_date "
                "ON signal_log (trade_date, trade_placed, consensus)"
            )
        )


if __name__ == "__main__":
    setup_logging()
    init_db()
