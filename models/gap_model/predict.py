from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.orm import Session
from xgboost import XGBRegressor

from db.models import ModelRegistry
from models.common import latest_feature_row


def _latest_model(db: Session, symbol: str) -> ModelRegistry | None:
    return db.scalar(
        select(ModelRegistry)
        .where(
            and_(
                ModelRegistry.model_name == "gap_v2",
                ModelRegistry.symbol == symbol,
                ModelRegistry.is_active.is_(True),
            )
        )
        .order_by(ModelRegistry.created_at.desc())
        .limit(1)
    )


def predict_from_feature_frame(db: Session, symbol: str, feature_df: pd.DataFrame) -> pd.DataFrame:
    model_row = _latest_model(db, symbol)
    if model_row is None or feature_df.empty:
        return pd.DataFrame()
    path = Path(model_row.artifact_path)
    feature_cols = json.loads((path / "feature_columns.json").read_text(encoding="utf-8"))
    model = XGBRegressor()
    model.load_model(str(path / "model.json"))
    frame = feature_df.copy().sort_values("session_date")
    X = frame[feature_cols].astype(float)
    gap_pred = model.predict(X)
    close = frame["close"].astype(float) if "close" in frame.columns else pd.Series([0.0] * len(frame))
    out = pd.DataFrame(
        {
            "session_date": frame["session_date"].values,
            "pred_gap": gap_pred.astype(float),
            "pred_open_from_gap": (close * (1 + gap_pred)).astype(float),
            "model_version": model_row.model_version,
        }
    )
    return out


def predict_symbol(db: Session, symbol: str) -> dict:
    model_row = _latest_model(db, symbol)
    feat_row = latest_feature_row(db, symbol)
    if model_row is None or feat_row is None:
        raise ValueError(f"Missing gap_v2 model/feature for symbol={symbol}")
    path = Path(model_row.artifact_path)
    feature_cols = json.loads((path / "feature_columns.json").read_text(encoding="utf-8"))
    model = XGBRegressor()
    model.load_model(str(path / "model.json"))
    X = pd.DataFrame([{k: float((feat_row.features or {}).get(k, 0.0)) for k in feature_cols}])
    pred_gap = float(model.predict(X)[0])
    close = float((feat_row.features or {}).get("close", 0.0))
    pred_open = close * (1 + pred_gap)
    return {
        "pred_gap": pred_gap,
        "pred_open": float(pred_open),
        "model_version": model_row.model_version,
        "metrics": model_row.metrics or {},
    }
