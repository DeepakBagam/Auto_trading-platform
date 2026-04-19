try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from db.connection import SessionLocal
from feature_engine.feature_builder import build_daily_features
from feature_engine.labels import build_daily_labels
from utils.config import get_settings
from utils.logger import setup_logging


def main() -> None:
    setup_logging()
    settings = get_settings()
    if not settings.instrument_keys:
        raise RuntimeError("No instruments configured. Set UPSTOX_INSTRUMENT_KEYS in .env.")
    db = SessionLocal()
    try:
        feature_summary = build_daily_features(db, settings.instrument_keys)
        label_summary = build_daily_labels(db, settings.instrument_keys)
        print("feature_summary=", feature_summary)
        print("label_summary=", label_summary)
    finally:
        db.close()


if __name__ == "__main__":
    main()
