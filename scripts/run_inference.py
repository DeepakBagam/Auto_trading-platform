try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from db.connection import SessionLocal
from prediction_engine.orchestrator import PredictionOrchestrator
from utils.config import get_settings
from utils.logger import setup_logging


def main() -> None:
    setup_logging()
    settings = get_settings()
    symbols = settings.instrument_keys
    if not symbols:
        raise RuntimeError("No instruments configured. Set UPSTOX_INSTRUMENT_KEYS in .env.")
    db = SessionLocal()
    try:
        orch = PredictionOrchestrator()
        summary = {
            "1minute": orch.run_intraday_inference(db, symbols, interval="1minute"),
        }
        print(summary)
    finally:
        db.close()


if __name__ == "__main__":
    main()
