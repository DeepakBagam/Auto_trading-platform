import argparse

from data_layer.collectors.news_collector import NewsCollector
from data_layer.collectors.upstox_collector import UpstoxCollector
from data_layer.scheduler.scheduler import start_scheduler
from db.connection import SessionLocal
from feature_engine.feature_builder import build_daily_features
from feature_engine.labels import build_daily_labels
from models.gap_model.train import train_all_symbols as train_gap_all
from models.garch.train import train_all_symbols as train_garch_all
from models.lstm_gru.train import train_all_symbols as train_lstm_all
from models.meta_model.train import train_all_symbols as train_meta_all
from models.xgboost.train import train_all_symbols
from prediction_engine.orchestrator import PredictionOrchestrator
from utils.config import get_settings
from utils.logger import setup_logging


def main() -> None:
    setup_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Trading platform CLI")
    parser.add_argument(
        "command",
        choices=[
            "ingest-news",
            "ingest-candles",
            "build-features",
            "train",
            "infer",
            "scheduler",
        ],
    )
    args = parser.parse_args()
    db = SessionLocal()
    try:
        symbols = settings.instrument_keys
        plain_symbols = [s.split("|", 1)[1] if "|" in s else s for s in symbols]
        if args.command in {"build-features", "train", "infer"} and not symbols:
            raise RuntimeError("No instruments configured. Set UPSTOX_INSTRUMENT_KEYS in .env.")
        if args.command == "ingest-news":
            print(NewsCollector().run_once(db))
        elif args.command == "ingest-candles":
            print(UpstoxCollector().ingest_historical_batch(db, days_back=180))
        elif args.command == "build-features":
            print(build_daily_features(db, symbols))
            print(build_daily_labels(db, symbols))
        elif args.command == "train":
            print(
                {
                    "xgboost_v1": train_all_symbols(db, plain_symbols),
                    "gap_v2": train_gap_all(db, plain_symbols),
                    "garch_v2": train_garch_all(db, plain_symbols),
                    "lstm_gru_v2": train_lstm_all(db, plain_symbols),
                    "meta_model_v2": train_meta_all(db, plain_symbols),
                }
            )
        elif args.command == "infer":
            print(PredictionOrchestrator().run_daily_inference(db, symbols))
        elif args.command == "scheduler":
            db.close()
            start_scheduler()
    finally:
        if db.is_active:
            db.close()


if __name__ == "__main__":
    main()
