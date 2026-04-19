from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.orm import Session
from xgboost import XGBClassifier, XGBRegressor

from db.models import FeaturesDaily, ModelRegistry
from utils.constants import DIRECTION_BUY, DIRECTION_HOLD, DIRECTION_SELL

LABEL_MAP = {-1: DIRECTION_SELL, 0: DIRECTION_HOLD, 1: DIRECTION_BUY}


def _latest_model(db: Session, symbol: str) -> ModelRegistry | None:
    return db.scalar(
        select(ModelRegistry)
        .where(
            and_(
                ModelRegistry.model_name == "xgboost_v1",
                ModelRegistry.symbol == symbol,
                ModelRegistry.is_active.is_(True),
            )
        )
        .order_by(ModelRegistry.created_at.desc())
        .limit(1)
    )


def _latest_feature_row(db: Session, symbol: str) -> FeaturesDaily | None:
    return db.scalar(
        select(FeaturesDaily)
        .where(FeaturesDaily.symbol == symbol)
        .order_by(FeaturesDaily.session_date.desc())
        .limit(1)
    )


def _load_bundle(artifact_path: str):
    path = Path(artifact_path)
    feature_cols = json.loads((path / "feature_columns.json").read_text(encoding="utf-8"))
    cls_decode = {0: -1, 1: 0, 2: 1}
    class_map_file = path / "direction_class_map.json"
    if class_map_file.exists():
        loaded = json.loads(class_map_file.read_text(encoding="utf-8"))
        cls_decode = {int(k): int(v) for k, v in loaded.items()}
    regs = {}
    for target in ("open", "high", "low", "close"):
        reg = XGBRegressor()
        reg.load_model(str(path / f"reg_{target}.json"))
        regs[target] = reg
    clf = XGBClassifier()
    clf.load_model(str(path / "cls_direction.json"))
    target_mode = "raw_price"
    meta_file = path / "prediction_meta.json"
    if meta_file.exists():
        target_mode = json.loads(meta_file.read_text(encoding="utf-8")).get("target_mode", "raw_price")
    return feature_cols, regs, clf, cls_decode, target_mode


def predict_from_feature_frame(db: Session, symbol: str, feature_df: pd.DataFrame) -> pd.DataFrame:
    model_row = _latest_model(db, symbol)
    if model_row is None or feature_df.empty:
        return pd.DataFrame()
    feature_cols, regs, clf, cls_decode, target_mode = _load_bundle(model_row.artifact_path)
    frame = feature_df.copy().sort_values("session_date")
    X = frame[feature_cols].astype(float)
    cls = clf.predict(X).astype(int)
    direction_code = np.vectorize(lambda v: int(cls_decode.get(int(v), int(v))))(cls)
    probs = clf.predict_proba(X)
    base_close = frame.get("close", pd.Series(np.zeros(len(frame), dtype=float))).astype(float).values
    pred_open = regs["open"].predict(X).astype(float)
    pred_high = regs["high"].predict(X).astype(float)
    pred_low = regs["low"].predict(X).astype(float)
    pred_close = regs["close"].predict(X).astype(float)
    if target_mode == "relative_to_close":
        pred_open = base_close * (1.0 + pred_open)
        pred_high = base_close * (1.0 + pred_high)
        pred_low = base_close * (1.0 + pred_low)
        pred_close = base_close * (1.0 + pred_close)
    fixed_high = np.maximum.reduce([pred_high, pred_open, pred_close])
    fixed_low = np.minimum.reduce([pred_low, pred_open, pred_close])
    return pd.DataFrame(
        {
            "session_date": frame["session_date"].values,
            "pred_open": pred_open.astype(float),
            "pred_high": fixed_high.astype(float),
            "pred_low": fixed_low.astype(float),
            "pred_close": pred_close.astype(float),
            "direction_code": direction_code,
            "direction_prob": np.max(probs, axis=1).astype(float),
            "model_version": model_row.model_version,
        }
    )


def predict_symbol(db: Session, symbol: str) -> dict:
    model_row = _latest_model(db, symbol)
    feat_row = _latest_feature_row(db, symbol)
    if model_row is None or feat_row is None:
        raise ValueError(f"Missing model/feature for symbol={symbol}")

    feature_cols, regs, clf, cls_decode, target_mode = _load_bundle(model_row.artifact_path)
    X = pd.DataFrame([{k: float(feat_row.features.get(k, 0.0)) for k in feature_cols}])

    reg_preds = {k: float(m.predict(X)[0]) for k, m in regs.items()}
    if target_mode == "relative_to_close":
        base_close = float(feat_row.features.get("close", 0.0))
        reg_preds = {k: base_close * (1.0 + float(v)) for k, v in reg_preds.items()}
    reg_preds["high"] = max(reg_preds["high"], reg_preds["open"], reg_preds["close"])
    reg_preds["low"] = min(reg_preds["low"], reg_preds["open"], reg_preds["close"])
    cls_raw = int(clf.predict(X)[0])
    direction_code = int(cls_decode.get(cls_raw, cls_raw))
    probs = clf.predict_proba(X)[0]
    max_prob = float(np.max(probs))

    return {
        "pred_open": reg_preds["open"],
        "pred_high": reg_preds["high"],
        "pred_low": reg_preds["low"],
        "pred_close": reg_preds["close"],
        "direction_code": direction_code,
        "direction": LABEL_MAP.get(direction_code, DIRECTION_HOLD),
        "direction_prob": max_prob,
        "model_version": model_row.model_version,
        "metrics": model_row.metrics or {},
    }
