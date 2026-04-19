from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error
from sqlalchemy import and_, select, update
from sqlalchemy.orm import Session

from db.models import ModelRegistry
from models.common import load_feature_frame, load_feature_label_frame
from models.gap_model.predict import predict_from_feature_frame as gap_predict_frame
from models.garch.predict import predict_symbol as garch_predict_symbol
from models.lstm_gru.predict import predict_from_feature_frame as lstm_predict_frame
from models.meta_model.utils import build_stacked_feature_frame
from models.xgboost.predict import predict_from_feature_frame as xgb_predict_frame
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)


def _latest_active(db: Session, model_name: str, symbol: str):
    return db.scalar(
        select(ModelRegistry)
        .where(
            and_(
                ModelRegistry.model_name == model_name,
                ModelRegistry.symbol == symbol,
                ModelRegistry.is_active.is_(True),
            )
        )
        .order_by(ModelRegistry.created_at.desc())
        .limit(1)
    )


def _ensure_base_models(db: Session, symbol: str) -> None:
    from models.gap_model.train import train_symbol_model as train_gap
    from models.garch.train import train_symbol_model as train_garch
    from models.lstm_gru.train import train_symbol_model as train_lstm
    from models.xgboost.train import train_symbol_models as train_xgb

    if _latest_active(db, "xgboost_v1", symbol) is None:
        train_xgb(db, symbol)
    if _latest_active(db, "gap_v2", symbol) is None:
        train_gap(db, symbol)
    if _latest_active(db, "garch_v2", symbol) is None:
        train_garch(db, symbol)
    if _latest_active(db, "lstm_gru_v2", symbol) is None:
        try:
            train_lstm(db, symbol)
        except Exception as exc:
            logger.warning("lstm_gru_v2 unavailable for symbol=%s: %s", symbol, exc)


def _prepare_stacked_training_frame(db: Session, symbol: str) -> pd.DataFrame:
    label_df = load_feature_label_frame(db, symbol)
    feature_df = load_feature_frame(db, symbol)
    if label_df.empty or feature_df.empty:
        return pd.DataFrame()
    xgb_df = xgb_predict_frame(db, symbol, feature_df).rename(
        columns={
            "pred_open": "xgb_open",
            "pred_high": "xgb_high",
            "pred_low": "xgb_low",
            "pred_close": "xgb_close",
            "direction_prob": "xgb_dir_prob",
        }
    )
    try:
        lstm_df = lstm_predict_frame(db, symbol, feature_df).rename(
            columns={
                "pred_open": "lstm_open",
                "pred_high": "lstm_high",
                "pred_low": "lstm_low",
                "pred_close": "lstm_close",
                "direction_prob": "lstm_dir_prob",
            }
        )
    except Exception:
        lstm_df = pd.DataFrame()
    gap_df = gap_predict_frame(db, symbol, feature_df).rename(
        columns={
            "pred_gap": "gap_pred",
            "pred_open_from_gap": "gap_open",
        }
    )
    garch = garch_predict_symbol(db, symbol)
    merged = build_stacked_feature_frame(
        label_df=label_df,
        xgb_df=xgb_df,
        lstm_df=lstm_df,
        gap_df=gap_df,
        garch_sigma=float(garch.get("pred_sigma", 0.0)),
    )
    return merged


def train_symbol_model(db: Session, symbol: str) -> dict:
    _ensure_base_models(db, symbol)
    frame = _prepare_stacked_training_frame(db, symbol)
    if len(frame) < 50:
        logger.warning("Not enough rows for meta_model_v2 symbol=%s rows=%s", symbol, len(frame))
        return {}

    feature_cols = [
        c
        for c in frame.columns
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
    X = frame[feature_cols].astype(float)
    y_reg = frame[["next_open", "next_high", "next_low", "next_close"]].astype(float)
    y_cls = frame["next_direction"].astype(int)

    split_idx = max(40, int(len(frame) * 0.8))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_reg, y_test_reg = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    y_train_cls, y_test_cls = y_cls.iloc[:split_idx], y_cls.iloc[split_idx:]

    reg_models = {}
    metrics = {}
    for i, target in enumerate(("open", "high", "low", "close")):
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train_reg.iloc[:, i])
        pred = model.predict(X_test)
        reg_models[target] = model
        metrics[f"mae_{target}"] = float(mean_absolute_error(y_test_reg.iloc[:, i], pred))

    cls_model = None
    cls_constant = None
    if y_train_cls.nunique() > 1:
        cls_model = LogisticRegression(max_iter=1000, random_state=42)
        cls_model.fit(X_train, y_train_cls)
        cls_pred = cls_model.predict(X_test)
        metrics["accuracy_direction"] = float(accuracy_score(y_test_cls, cls_pred))
    else:
        cls_constant = int(y_train_cls.iloc[0])
        metrics["accuracy_direction"] = float((y_test_cls == cls_constant).mean())

    settings = get_settings()
    version = f"meta_{datetime.now(IST_ZONE).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(settings.model_artifacts_dir) / symbol / version
    out_dir.mkdir(parents=True, exist_ok=True)
    for target, model in reg_models.items():
        dump(model, out_dir / f"ridge_{target}.joblib")
    if cls_model is not None:
        dump(cls_model, out_dir / "cls.joblib")
    meta_json = {"feature_columns": feature_cols, "cls_constant": cls_constant}
    (out_dir / "meta.json").write_text(json.dumps(meta_json), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    db.execute(
        update(ModelRegistry)
        .where(and_(ModelRegistry.model_name == "meta_model_v2", ModelRegistry.symbol == symbol))
        .values(is_active=False)
    )
    db.add(
        ModelRegistry(
            model_name="meta_model_v2",
            model_version=version,
            model_type="stacked_ridge_logreg",
            symbol=symbol,
            artifact_path=str(out_dir),
            metrics=metrics,
            trained_from=frame["session_date"].min(),
            trained_to=frame["session_date"].max(),
            is_active=True,
        )
    )
    db.commit()
    logger.info("Trained meta_model_v2 symbol=%s version=%s", symbol, version)
    return {"symbol": symbol, "version": version, "metrics": metrics}


def train_all_symbols(db: Session, symbols: list[str]) -> dict:
    out = {}
    for symbol in symbols:
        try:
            out[symbol] = train_symbol_model(db, symbol)
        except Exception as exc:
            logger.exception("meta_model_v2 training failed symbol=%s: %s", symbol, exc)
            out[symbol] = {"error": str(exc)}
    return out
