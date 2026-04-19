try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import argparse

from db.connection import SessionLocal
from prediction_engine.orchestrator import PredictionOrchestrator
from utils.config import get_settings
from utils.logger import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Backfill continuous daily prediction signals")
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="How many latest feature days to score per symbol",
    )
    args = parser.parse_args()

    settings = get_settings()
    symbols = settings.instrument_keys
    if not symbols:
        raise RuntimeError("No instruments configured. Set UPSTOX_INSTRUMENT_KEYS in .env.")

    db = SessionLocal()
    try:
        summary = PredictionOrchestrator().run_historical_daily_signals(
            db, symbols=symbols, limit=args.limit
        )
        print(summary)
    finally:
        db.close()


if __name__ == "__main__":
    main()
