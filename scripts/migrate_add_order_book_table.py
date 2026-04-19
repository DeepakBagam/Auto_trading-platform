"""Add order_book_snapshots table for microstructure features."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from db.connection import get_engine
from utils.logger import get_logger

logger = get_logger(__name__)


def run_migration():
    """Add order_book_snapshots table."""
    engine = get_engine()
    
    logger.info("Running migration: add_order_book_snapshots")
    
    # Check if table already exists
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'order_book_snapshots'
            );
        """))
        exists = result.scalar()
        
        if exists:
            logger.info("Table order_book_snapshots already exists, skipping migration")
            return
    
    # Create table
    create_table_sql = """
    CREATE TABLE order_book_snapshots (
        id SERIAL PRIMARY KEY,
        instrument_key VARCHAR(128) NOT NULL,
        ts TIMESTAMP WITH TIME ZONE NOT NULL,
        best_bid FLOAT NOT NULL,
        best_ask FLOAT NOT NULL,
        mid_price FLOAT NOT NULL,
        spread_bps FLOAT NOT NULL,
        bid_volume FLOAT NOT NULL,
        ask_volume FLOAT NOT NULL,
        depth_imbalance FLOAT NOT NULL,
        liquidity_score FLOAT NOT NULL,
        depth_data JSON DEFAULT '{}',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        CONSTRAINT uq_order_book_snapshot UNIQUE (instrument_key, ts)
    );
    
    CREATE INDEX idx_order_book_instrument ON order_book_snapshots(instrument_key);
    CREATE INDEX idx_order_book_ts ON order_book_snapshots(ts);
    """
    
    try:
        with engine.begin() as conn:
            conn.execute(text(create_table_sql))
        
        logger.info("Successfully created order_book_snapshots table")
        logger.info("Migration completed successfully")
        
    except Exception as exc:
        logger.error("Migration failed: %s", exc)
        raise


if __name__ == "__main__":
    run_migration()
