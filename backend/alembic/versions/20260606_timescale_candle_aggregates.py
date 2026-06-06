"""Timescale candle hypertable and continuous aggregates

Revision ID: 20260606_timescale_candle_aggregates
Revises: 481b49b49769
Create Date: 2026-06-06 11:30:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = "20260606_timescale_candle_aggregates"
down_revision: Union[str, None] = "481b49b49769"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


AGGREGATES = {
    "candles_5m": "5 minutes",
    "candles_15m": "15 minutes",
    "candles_30m": "30 minutes",
    "candles_1h": "1 hour",
    "candles_4h": "4 hours",
    "candles_1d": "1 day",
    "candles_1w": "1 week",
}


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume BIGINT NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time ON candles(symbol, timestamp DESC)")
    op.execute(
        """
        SELECT create_hypertable(
            'candles',
            'timestamp',
            if_not_exists => TRUE,
            migrate_data => TRUE
        )
        """
    )
    for view_name, bucket in AGGREGATES.items():
        op.execute(
            f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS {view_name}
            WITH (timescaledb.continuous)
            AS
            SELECT
                symbol,
                time_bucket('{bucket}', timestamp) AS bucket,
                first(open, timestamp) AS open,
                max(high) AS high,
                min(low) AS low,
                last(close, timestamp) AS close,
                sum(volume) AS volume
            FROM candles
            GROUP BY symbol, bucket
            WITH NO DATA
            """
        )
        op.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{view_name}_symbol_bucket ON {view_name}(symbol, bucket DESC)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for view_name in reversed(tuple(AGGREGATES)):
        op.execute(f"DROP MATERIALIZED VIEW IF EXISTS {view_name}")
    op.execute("DROP TABLE IF EXISTS candles")
