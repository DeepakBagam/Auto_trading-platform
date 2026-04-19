from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from db.models import CalibrationRegistry, FeaturesDaily, ModelRegistry, RawCandle
from models.calibration_v3 import apply_calibrator, load_calibrator
from utils.constants import DIRECTION_BUY, DIRECTION_HOLD, DIRECTION_SELL

DIR_LABEL = {-1: DIRECTION_SELL, 0: DIRECTION_HOLD, 1: DIRECTION_BUY}


def _require_lightgbm():
    import lightgbm as lgb

    return lgb


def _require_catboost():
    try:
        from catboost import CatBoostRegressor

        return "catboost", CatBoostRegressor
    except Exception:
        from xgboost import XGBRegressor

        return "xgboost_fallback", XGBRegressor


def _latest_model(db: Session, symbol: str) -> ModelRegistry | None:
    return db.scalar(
        select(ModelRegistry)
        .where(
            and_(
                ModelRegistry.model_name == "meta_v3",
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


def _latest_calibration_version(db: Session, symbol: str) -> str | None:
    row = db.scalar(
        select(CalibrationRegistry)
        .where(
            and_(
                CalibrationRegistry.symbol == symbol,
                CalibrationRegistry.model_family == "meta_v3",
                CalibrationRegistry.is_active.is_(True),
            )
        )
        .order_by(CalibrationRegistry.created_at.desc())
        .limit(1)
    )
    return row.calibration_version if row else None


def _rolling_sigma(close: np.ndarray, window: int = 20) -> float:
    if len(close) < 3:
        return 0.01
    ret = np.diff(close) / (close[:-1] + 1e-9)
    if len(ret) < window:
        return float(np.std(ret))
    return float(np.std(ret[-window:]))


def _load_recent_close(db: Session, symbol: str, limit: int = 40) -> np.ndarray:
    rows = (
        db.execute(
            select(RawCandle)
            .where(and_(RawCandle.interval == "day", RawCandle.instrument_key.like(f"%|{symbol}")))
            .order_by(RawCandle.ts.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    rows = list(reversed(rows))
    return np.asarray([float(r.close) for r in rows], dtype=float)


def _direction_from_prob(p_up: float) -> tuple[int, str]:
    if p_up >= 0.55:
        return 1, DIRECTION_BUY
    if p_up <= 0.45:
        return -1, DIRECTION_SELL
    return 0, DIRECTION_HOLD


def _meta_feature_frame(
    feature_row: dict[str, float],
    lgb_pred: dict[str, float],
    cat_pred: dict[str, float],
    dir_prob_raw: float,
    vol_sigma: float,
) -> pd.DataFrame:
    payload: dict[str, float] = {
        "dir_prob_raw": float(dir_prob_raw),
        "vol_sigma": float(vol_sigma),
    }
    for t in ("open", "high", "low", "close"):
        l_val = float(lgb_pred[t])
        c_val = float(cat_pred[t])
        payload[f"lgb_{t}"] = l_val
        payload[f"cat_{t}"] = c_val
        payload[f"base_mean_{t}"] = 0.5 * (l_val + c_val)
        payload[f"base_diff_{t}"] = l_val - c_val
    for key, value in feature_row.items():
        payload[f"feat_{key}"] = float(value)
    return pd.DataFrame([payload])


def _load_meta_regressor(base_dir: Path, target: str):
    reg_path = base_dir / "meta" / f"reg_{target}.joblib"
    if reg_path.exists():
        return load(reg_path)
    legacy_path = base_dir / "meta" / f"ridge_{target}.joblib"
    return load(legacy_path)


def predict_symbol(db: Session, symbol: str) -> dict:
    model_row = _latest_model(db, symbol)
    feat_row = _latest_feature_row(db, symbol)
    if model_row is None or feat_row is None:
        raise ValueError(f"Missing meta_v3 model/feature for symbol={symbol}")

    lgb = _require_lightgbm()
    cat_backend, CatBoostRegressor = _require_catboost()
    base_dir = Path(model_row.artifact_path)
    meta_cfg = json.loads((base_dir / "meta_v3.json").read_text(encoding="utf-8"))
    feature_cols = meta_cfg.get("feature_columns", [])
    meta_cols = meta_cfg.get("meta_feature_columns", [])
    dir_map = {str(k): int(v) for k, v in (meta_cfg.get("direction_mapping", {}) or {}).items()}
    up_encoded = int(dir_map.get("1", 2))

    X = pd.DataFrame([{k: float((feat_row.features or {}).get(k, 0.0)) for k in feature_cols}])
    base_close = float((feat_row.features or {}).get("close", 0.0))

    lgb_pred = {}
    cat_pred = {}
    configured_backend = str(meta_cfg.get("catboost_backend", cat_backend))
    for t in ("open", "high", "low", "close"):
        booster = lgb.Booster(model_file=str(base_dir / "lgbm" / f"reg_{t}.txt"))
        ret_pred = float(booster.predict(X)[0])
        lgb_pred[t] = base_close * (1.0 + ret_pred)

        cat = CatBoostRegressor()
        if configured_backend == "catboost":
            cat.load_model(str(base_dir / "catboost" / f"reg_{t}.cbm"))
        else:
            cat.load_model(str(base_dir / "catboost" / f"reg_{t}.json"))
        ret_pred_cat = float(np.asarray(cat.predict(X), dtype=float)[0])
        cat_pred[t] = base_close * (1.0 + ret_pred_cat)

    dir_booster = lgb.Booster(model_file=str(base_dir / "direction" / "clf.txt"))
    dir_pred = np.asarray(dir_booster.predict(X), dtype=float)
    if dir_pred.ndim == 1 and dir_pred.shape[0] > 1:
        probs = dir_pred
    else:
        probs = np.asarray(dir_pred).reshape(-1)
    if len(probs) > up_encoded:
        dir_prob_raw = float(probs[up_encoded])
    else:
        dir_prob_raw = float(np.max(probs))

    vol_sigma = _rolling_sigma(_load_recent_close(db, symbol))
    meta_x = _meta_feature_frame(
        feature_row={k: float((feat_row.features or {}).get(k, 0.0)) for k in feature_cols},
        lgb_pred=lgb_pred,
        cat_pred=cat_pred,
        dir_prob_raw=dir_prob_raw,
        vol_sigma=vol_sigma,
    )
    meta_x = meta_x[[c for c in meta_cols if c in meta_x.columns]]

    reg_mode = str(meta_cfg.get("meta_regression_mode", "direct_ridge"))
    reg_preds = {}
    for t in ("open", "high", "low", "close"):
        reg = _load_meta_regressor(base_dir, t)
        raw_pred = float(reg.predict(meta_x)[0])
        if reg_mode == "residual_hgb_mean_base_v1":
            reg_preds[t] = float(meta_x[f"base_mean_{t}"].iloc[0] + raw_pred)
        else:
            reg_preds[t] = raw_pred
    reg_preds["high"] = max(reg_preds["high"], reg_preds["open"], reg_preds["close"])
    reg_preds["low"] = min(reg_preds["low"], reg_preds["open"], reg_preds["close"])

    dir_meta = load(base_dir / "meta" / "dir_meta.joblib")
    dir_prob_meta_raw = float(dir_meta.predict_proba(meta_x)[0, 1])

    cal_method, cal_model = load_calibrator(base_dir / "calibration")
    dir_prob_cal = float(apply_calibrator(cal_method, cal_model, np.array([dir_prob_meta_raw]))[0])
    direction_code, direction = _direction_from_prob(dir_prob_cal)

    bands = meta_cfg.get("interval_residual_bands", {})
    pred_interval = {}
    for t in ("open", "high", "low", "close"):
        b = bands.get(t, {"q10": 0.0, "q90": 0.0})
        p50 = float(reg_preds[t])
        pred_interval[t] = {
            "p10": float(p50 + float(b.get("q10", 0.0))),
            "p50": p50,
            "p90": float(p50 + float(b.get("q90", 0.0))),
        }

    return {
        "pred_open": reg_preds["open"],
        "pred_high": reg_preds["high"],
        "pred_low": reg_preds["low"],
        "pred_close": reg_preds["close"],
        "direction_code": direction_code,
        "direction": direction,
        "direction_prob": dir_prob_meta_raw,
        "direction_prob_calibrated": dir_prob_cal,
        "model_version": model_row.model_version,
        "model_family": "meta_v3",
        "calibration_version": _latest_calibration_version(db, symbol),
        "pred_interval": pred_interval,
        "interval_coverage": float((meta_cfg.get("coverage", {}) or {}).get("close", 0.0)),
        "interval_width_pct": float((meta_cfg.get("width_pct", {}) or {}).get("close", 0.0)),
        "metrics": model_row.metrics or {},
    }
