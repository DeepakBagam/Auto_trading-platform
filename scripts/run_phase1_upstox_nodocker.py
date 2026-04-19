try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from db.connection import SessionLocal
from db.init_db import init_db
from feature_engine.feature_builder import build_daily_features
from feature_engine.labels import build_daily_labels
from models.gap_model.train import train_all_symbols as train_gap_all
from models.garch.train import train_all_symbols as train_garch_all
from models.lstm_gru.train import train_all_symbols as train_lstm_all
from models.meta_model.train import train_all_symbols as train_meta_all
from models.meta_v3.train import train_all_symbols as train_meta_v3_all
from models.xgboost.train import train_all_symbols
from prediction_engine.orchestrator import PredictionOrchestrator
from data_layer.collectors.news_collector import NewsCollector
from data_layer.collectors.upstox_collector import UpstoxCollector
from utils.config import get_settings
from utils.logger import setup_logging


def main() -> None:
    setup_logging()
    settings = get_settings()
    if not settings.instrument_keys:
        raise RuntimeError("No instruments configured. Set UPSTOX_INSTRUMENT_KEYS in .env.")
    init_db()
    db = SessionLocal()
    try:
        news_inserted = NewsCollector().run_once(db)
        candles_summary = UpstoxCollector().ingest_historical_batch(db, days_back=365)
        feature_summary = build_daily_features(db, settings.instrument_keys)
        label_summary = build_daily_labels(db, settings.instrument_keys)
        symbols = [s.split("|", 1)[1] if "|" in s else s for s in settings.instrument_keys]
        train_summary = {
            "xgboost_v1": train_all_symbols(db, symbols),
            "gap_v2": train_gap_all(db, symbols),
            "garch_v2": train_garch_all(db, symbols),
            "lstm_gru_v2": train_lstm_all(db, symbols),
            "meta_model_v2": train_meta_all(db, symbols),
            "meta_v3": train_meta_v3_all(db, symbols),
        }
        orchestrator = PredictionOrchestrator()
        pred_summary = {
            "1minute": orchestrator.run_intraday_inference(db, settings.instrument_keys, interval="1minute"),
        }
        print(
            {
                "news_inserted": news_inserted,
                "candles_summary": candles_summary,
                "feature_summary": feature_summary,
                "label_summary": label_summary,
                "train_summary": train_summary,
                "pred_summary": pred_summary,
            }
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
