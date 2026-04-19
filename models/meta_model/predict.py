from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from joblib import load
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from db.models import ModelRegistry
from models.gap_model.predict import predict_symbol as gap_predict_symbol
from models.garch.predict import predict_symbol as garch_predict_symbol
from models.lstm_gru.predict import predict_symbol as lstm_predict_symbol
from models.xgboost.predict import predict_symbol as xgb_predict_symbol
from utils.constants import DIRECTION_BUY, DIRECTION_HOLD, DIRECTION_SELL

DIR_TO_LABEL = {-1: DIRECTION_SELL, 0: DIRECTION_HOLD, 1: DIRECTION_BUY}


def _latest_model(db: Session, symbol: str) -> ModelRegistry | None:
    return db.scalar(
        select(ModelRegistry)
        .where(
            and_(
                ModelRegistry.model_name == "meta_model_v2",
                ModelRegistry.symbol == symbol,
                ModelRegistry.is_active.is_(True),
            )
        )
        .order_by(ModelRegistry.created_at.desc())
        .limit(1)
    )


def _build_feature_vector(db: Session, symbol: str) -> dict:
    xgb = xgb_predict_symbol(db, symbol)
    try:
        lstm = lstm_predict_symbol(db, symbol)
    except Exception:
        lstm = {
            "pred_open": xgb["pred_open"],
            "pred_high": xgb["pred_high"],
            "pred_low": xgb["pred_low"],
            "pred_close": xgb["pred_close"],
            "direction_prob": xgb["direction_prob"],
        }
    gap = gap_predict_symbol(db, symbol)
    garch = garch_predict_symbol(db, symbol)
    return {
        "xgb_open": float(xgb["pred_open"]),
        "xgb_high": float(xgb["pred_high"]),
        "xgb_low": float(xgb["pred_low"]),
        "xgb_close": float(xgb["pred_close"]),
        "xgb_dir_prob": float(xgb["direction_prob"]),
        "lstm_open": float(lstm["pred_open"]),
        "lstm_high": float(lstm["pred_high"]),
        "lstm_low": float(lstm["pred_low"]),
        "lstm_close": float(lstm["pred_close"]),
        "lstm_dir_prob": float(lstm["direction_prob"]),
        "gap_pred": float(gap["pred_gap"]),
        "gap_open": float(gap["pred_open"]),
        "garch_sigma": float(garch["pred_sigma"]),
    }


def predict_symbol(db: Session, symbol: str) -> dict:
    model_row = _latest_model(db, symbol)
    if model_row is None:
        raise ValueError(f"Missing meta_model_v2 for symbol={symbol}")
    path = Path(model_row.artifact_path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    cols = meta.get("feature_columns", [])
    x = _build_feature_vector(db, symbol)
    X = pd.DataFrame([{c: float(x.get(c, 0.0)) for c in cols}])

    reg_preds = {}
    for target in ("open", "high", "low", "close"):
        reg = load(path / f"ridge_{target}.joblib")
        reg_preds[target] = float(reg.predict(X)[0])

    cls_constant = meta.get("cls_constant")
    if cls_constant is not None:
        direction_code = int(cls_constant)
        direction_prob = 0.55
    else:
        cls = load(path / "cls.joblib")
        direction_code = int(cls.predict(X)[0])
        probs = cls.predict_proba(X)[0]
        direction_prob = float(max(probs))

    return {
        "pred_open": reg_preds["open"],
        "pred_high": reg_preds["high"],
        "pred_low": reg_preds["low"],
        "pred_close": reg_preds["close"],
        "direction_code": direction_code,
        "direction": DIR_TO_LABEL.get(direction_code, DIRECTION_HOLD),
        "direction_prob": direction_prob,
        "model_version": model_row.model_version,
        "metrics": model_row.metrics or {},
    }
