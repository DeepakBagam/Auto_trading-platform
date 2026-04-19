from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sklearn.metrics import mean_absolute_error
from sqlalchemy import and_, update
from sqlalchemy.orm import Session
from xgboost import XGBRegressor

from db.models import ModelRegistry
from models.common import load_feature_label_frame
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)


def train_symbol_model(db: Session, symbol: str) -> dict:
    df = load_feature_label_frame(db, symbol)
    if len(df) < 50:
        logger.warning("Not enough rows for gap_v2 symbol=%s rows=%s", symbol, len(df))
        return {}
    if "close" not in df.columns:
        raise ValueError(f"Feature column 'close' missing for symbol={symbol}")

    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "session_date",
            "next_open",
            "next_high",
            "next_low",
            "next_close",
            "next_direction",
        }
    ]
    y_gap = ((df["next_open"] - df["close"]) / (df["close"] + 1e-9)).astype(float)
    X = df[feature_cols].astype(float)
    split_idx = max(40, int(len(df) * 0.8))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_gap.iloc[:split_idx], y_gap.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"mae_gap": float(mean_absolute_error(y_test, pred))}

    settings = get_settings()
    version = f"gap_{datetime.now(IST_ZONE).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(settings.model_artifacts_dir) / symbol / version
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "model.json"))
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_cols), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    db.execute(
        update(ModelRegistry)
        .where(and_(ModelRegistry.model_name == "gap_v2", ModelRegistry.symbol == symbol))
        .values(is_active=False)
    )
    db.add(
        ModelRegistry(
            model_name="gap_v2",
            model_version=version,
            model_type="xgboost_gap_regressor",
            symbol=symbol,
            artifact_path=str(out_dir),
            metrics=metrics,
            trained_from=df["session_date"].min(),
            trained_to=df["session_date"].max(),
            is_active=True,
        )
    )
    db.commit()
    logger.info("Trained gap_v2 symbol=%s version=%s", symbol, version)
    return {"symbol": symbol, "version": version, "metrics": metrics}


def train_all_symbols(db: Session, symbols: list[str]) -> dict:
    out = {}
    for symbol in symbols:
        try:
            out[symbol] = train_symbol_model(db, symbol)
        except Exception as exc:
            logger.exception("gap_v2 training failed symbol=%s: %s", symbol, exc)
            out[symbol] = {"error": str(exc)}
    return out
