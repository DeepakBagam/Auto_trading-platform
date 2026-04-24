"""sync_nullable_and_constraints

Revision ID: 481b49b49769
Revises: 0001
Create Date: 2026-04-24 15:41:32.315550+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '481b49b49769'
down_revision: Union[str, None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# SQLite doesn't allow renaming a table while views reference it.
# Drop them before batch ops; they are recreated by create_symbol_interval_views()
# on the next app startup.
_PREDICTION_VIEWS = [
    "predictions_nifty50_1d", "predictions_nifty50_1m", "predictions_nifty50_30m",
    "predictions_banknifty_1d", "predictions_banknifty_1m", "predictions_banknifty_30m",
    "predictions_indiavix_1d", "predictions_indiavix_1m", "predictions_indiavix_30m",
    "predictions_sensex_1d", "predictions_sensex_1m", "predictions_sensex_30m",
]


def upgrade() -> None:
    # Drop views that reference predictions_daily/predictions_intraday so SQLite
    # allows the batch table-rebuild below.
    for view in _PREDICTION_VIEWS:
        op.execute(f"DROP VIEW IF EXISTS {view}")

    with op.batch_alter_table('execution_orders', schema=None) as batch_op:
        batch_op.alter_column('tsl_active',
               existing_type=sa.BOOLEAN(),
               server_default=None,
               nullable=False)

    with op.batch_alter_table('execution_positions', schema=None) as batch_op:
        batch_op.alter_column('tsl_active',
               existing_type=sa.BOOLEAN(),
               server_default=None,
               nullable=False)

    with op.batch_alter_table('option_quotes', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_option_quotes_underlying_key'), ['underlying_key'], unique=False)

    with op.batch_alter_table('predictions_daily', schema=None) as batch_op:
        batch_op.alter_column('interval',
               existing_type=sa.VARCHAR(length=32),
               server_default=None,
               nullable=False)
        batch_op.drop_constraint(batch_op.f('uq_pred_daily'), type_='unique')
        batch_op.create_unique_constraint('uq_pred_daily', ['symbol', 'interval', 'target_session_date', 'model_version'])
        batch_op.create_index(batch_op.f('ix_predictions_daily_interval'), ['interval'], unique=False)


def downgrade() -> None:
    for view in _PREDICTION_VIEWS:
        op.execute(f"DROP VIEW IF EXISTS {view}")

    with op.batch_alter_table('predictions_daily', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_predictions_daily_interval'))
        batch_op.drop_constraint('uq_pred_daily', type_='unique')
        batch_op.create_unique_constraint(batch_op.f('uq_pred_daily'), ['symbol', 'target_session_date', 'model_version'])
        batch_op.alter_column('interval',
               existing_type=sa.VARCHAR(length=32),
               server_default=sa.text("'day'"),
               nullable=True)

    with op.batch_alter_table('option_quotes', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_option_quotes_underlying_key'))

    with op.batch_alter_table('execution_positions', schema=None) as batch_op:
        batch_op.alter_column('tsl_active',
               existing_type=sa.BOOLEAN(),
               server_default=sa.text('0'),
               nullable=True)

    with op.batch_alter_table('execution_orders', schema=None) as batch_op:
        batch_op.alter_column('tsl_active',
               existing_type=sa.BOOLEAN(),
               server_default=sa.text('0'),
               nullable=True)
