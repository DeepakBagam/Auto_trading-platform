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

def upgrade() -> None:
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


def downgrade() -> None:
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
