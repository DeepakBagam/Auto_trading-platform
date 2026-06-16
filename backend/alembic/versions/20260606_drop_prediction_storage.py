"""Drop legacy prediction storage

Revision ID: 20260606_drop_prediction_storage
Revises: 20260606_timescale_candle_aggregates
Create Date: 2026-06-06 19:20:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = "20260606_drop_prediction_storage"
down_revision: Union[str, None] = "20260606_timescale_candle_aggregates"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


LEGACY_VIEWS = (
    "predictions_nifty50_1d",
    "predictions_nifty50_1m",
    "predictions_nifty50_30m",
    "predictions_banknifty_1d",
    "predictions_banknifty_1m",
    "predictions_banknifty_30m",
    "predictions_indiavix_1d",
    "predictions_indiavix_1m",
    "predictions_indiavix_30m",
    "predictions_sensex_1d",
    "predictions_sensex_1m",
    "predictions_sensex_30m",
)
LEGACY_TABLES = ("predictions_daily", "predictions_intraday", "oof_predictions")


def upgrade() -> None:
    bind = op.get_bind()
    for view_name in LEGACY_VIEWS:
        op.execute(f"DROP VIEW IF EXISTS {view_name}")
    for table_name in LEGACY_TABLES:
        suffix = " CASCADE" if bind.dialect.name == "postgresql" else ""
        op.execute(f"DROP TABLE IF EXISTS {table_name}{suffix}")


def downgrade() -> None:
    pass
