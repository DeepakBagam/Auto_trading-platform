try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import importlib.util

from db.connection import SessionLocal
from models.gap_model.train import train_all_symbols as train_gap_all
from models.garch.train import train_all_symbols as train_garch_all
from models.lstm_gru.train import train_all_symbols as train_lstm_all
from models.meta_model.train import train_all_symbols as train_meta_all
from models.meta_v3.train import train_all_symbols as train_meta_v3_all
from models.xgboost.train import train_all_symbols
from utils.config import get_settings
from utils.logger import setup_logging


def main() -> None:
    setup_logging()
    settings = get_settings()
    symbols = [s.split("|", 1)[1] if "|" in s else s for s in settings.instrument_keys]
    if not symbols:
        raise RuntimeError("No instruments configured. Set UPSTOX_INSTRUMENT_KEYS in .env.")
    db = SessionLocal()
    try:
        has_torch = importlib.util.find_spec("torch") is not None
        summary = {
            "xgboost_v1": train_all_symbols(db, symbols),
            "gap_v2": train_gap_all(db, symbols),
            "garch_v2": train_garch_all(db, symbols),
            "lstm_gru_v2": (
                train_lstm_all(db, symbols)
                if has_torch
                else {"status": "skipped", "reason": "torch not installed"}
            ),
            "meta_model_v2": train_meta_all(db, symbols),
            "meta_v3": train_meta_v3_all(db, symbols),
        }
        print(summary)
    finally:
        db.close()


if __name__ == "__main__":
    main()
