from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import and_, delete, select, update
from sqlalchemy.orm import Session

from db.models import (
    BacktestRun,
    CalibrationRegistry,
    DriftMetric,
    ModelRegistry,
    OOFPrediction,
)
from models.calibration_v3 import (
    brier_score,
    expected_calibration_error,
    fit_direction_calibrator,
    save_calibrator,
)
from models.v3_pipeline import build_point_in_time_dataset, write_training_metadata
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)

TARGETS = ("open", "high", "low", "close")
RET_CLIP = 0.2
META_REGRESSION_MODE = "residual_hgb_mean_base_v1"


def _require_lightgbm():
    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("lightgbm is required for lgbm_v3. Install dependencies first.") from exc
    return lgb


def _require_catboost():
    try:
        from catboost import CatBoostRegressor
        return "catboost", CatBoostRegressor
    except Exception:
        from xgboost import XGBRegressor

        return "xgboost_fallback", XGBRegressor


def _rolling_sigma(close: pd.Series, window: int = 20) -> np.ndarray:
    ret = close.pct_change().fillna(0.0)
    return ret.rolling(window).std(ddof=0).fillna(ret.std(ddof=0)).to_numpy(dtype=float)


def _train_vol_model(close: pd.Series, out_dir: Path) -> tuple[dict, dict]:
    ret = close.pct_change().dropna().to_numpy(dtype=float)
    if len(ret) < 40:
        payload = {"method": "rolling_std", "window": 20, "last_sigma": float(np.std(ret) if len(ret) else 0.01)}
        (out_dir / "vol_model.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload, {"mae_sigma": 0.0}

    try:
        from arch import arch_model

        model = arch_model(ret * 100.0, mean="Zero", vol="EGARCH", p=1, q=1, dist="normal")
        fit = model.fit(disp="off")
        sigma = np.asarray(fit.conditional_volatility, dtype=float) / 100.0
        mae = float(mean_absolute_error(np.abs(ret[-len(sigma) :]), sigma))
        payload = {
            "method": "egarch",
            "params": {k: float(v) for k, v in fit.params.items()},
            "last_sigma": float(sigma[-1]),
        }
        (out_dir / "vol_model.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload, {"mae_sigma": mae}
    except Exception as exc:
        logger.warning("EGARCH unavailable, fallback to rolling std: %s", exc)
        sigma = _rolling_sigma(close, window=20)
        payload = {
            "method": "rolling_std",
            "window": 20,
            "last_sigma": float(sigma[-1] if len(sigma) else 0.01),
        }
        (out_dir / "vol_model.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload, {"mae_sigma": float(np.mean(np.abs(np.diff(close.pct_change().fillna(0.0).to_numpy()))))}


def _make_model_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "lgbm": base / "lgbm",
        "catboost": base / "catboost",
        "direction": base / "direction",
        "meta": base / "meta",
        "calibration": base / "calibration",
        "vol": base / "vol",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _meta_feature_frame(
    *,
    feature_df: pd.DataFrame,
    lgb_pred: dict[str, np.ndarray],
    cat_pred: dict[str, np.ndarray],
    dir_prob_raw: np.ndarray,
    sigma_series: np.ndarray,
) -> pd.DataFrame:
    feature_df = feature_df.reset_index(drop=True).astype(float)
    base = {
        "dir_prob_raw": np.asarray(dir_prob_raw, dtype=float),
        "vol_sigma": np.asarray(sigma_series, dtype=float),
    }
    for t in TARGETS:
        l_arr = np.asarray(lgb_pred[t], dtype=float)
        c_arr = np.asarray(cat_pred[t], dtype=float)
        base[f"lgb_{t}"] = l_arr
        base[f"cat_{t}"] = c_arr
        base[f"base_mean_{t}"] = 0.5 * (l_arr + c_arr)
        base[f"base_diff_{t}"] = l_arr - c_arr
    base_df = pd.DataFrame(base)
    aux_df = feature_df.add_prefix("feat_")
    return pd.concat([base_df, aux_df], axis=1)


def _build_meta_regressor(symbol: str, target: str) -> HistGradientBoostingRegressor:
    max_depth = 3
    max_iter = 260
    if symbol == "Nifty 50" and target == "close":
        max_depth = 4
        max_iter = 320
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=max_iter,
        max_depth=max_depth,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )


def _store_oof_predictions(
    db: Session,
    run_id: str,
    symbol: str,
    session_dates: np.ndarray,
    pred_close: np.ndarray,
    direction_prob: np.ndarray,
    actual_close: np.ndarray,
    actual_dir: np.ndarray,
    fold_ids: np.ndarray,
) -> None:
    db.execute(delete(OOFPrediction).where(and_(OOFPrediction.run_id == run_id, OOFPrediction.symbol == symbol)))
    for i in range(len(session_dates)):
        db.add(
            OOFPrediction(
                run_id=run_id,
                symbol=symbol,
                session_date=session_dates[i],
                model_name="meta_v3",
                fold=int(fold_ids[i]),
                pred_close=float(pred_close[i]),
                direction_prob=float(direction_prob[i]),
                actual_close=float(actual_close[i]),
                actual_direction=int(actual_dir[i]),
            )
        )


def _update_registry(
    db: Session,
    symbol: str,
    version: str,
    out_dir: Path,
    metrics: dict,
    date_from,
    date_to,
    passed: bool,
) -> None:
    names = ["lgbm_v3", "catboost_v3", "direction_clf_v3", "vol_model_v3", "meta_v3"]
    if passed:
        for name in names:
            db.execute(
                update(ModelRegistry)
                .where(and_(ModelRegistry.model_name == name, ModelRegistry.symbol == symbol))
                .values(is_active=False)
            )
    for name in names:
        db.add(
            ModelRegistry(
                model_name=name,
                model_version=version,
                model_type="ensemble_component" if name != "meta_v3" else "stacked_meta_v3",
                symbol=symbol,
                artifact_path=str(out_dir),
                metrics=metrics,
                trained_from=date_from,
                trained_to=date_to,
                is_active=bool(passed),
            )
        )


def train_symbol_model(db: Session, symbol: str) -> dict:
    settings = get_settings()
    df, quality = build_point_in_time_dataset(db, symbol)
    if not quality.passed:
        return {"error": "quality_gates_failed", "quality_report": quality.details | {"rows": quality.rows}}

    lgb = _require_lightgbm()
    cat_backend, CatBoostRegressor = _require_catboost()

    feature_cols = [
        c
        for c in df.columns
        if c not in {"session_date", "next_open", "next_high", "next_low", "next_close", "next_direction"}
    ]
    X = df[feature_cols].astype(float)
    close = df["close"].astype(float).to_numpy()
    y_price = {t: df[f"next_{t}"].astype(float).to_numpy() for t in TARGETS}
    y_ret = {
        t: np.clip((y_price[t] / (close + 1e-9)) - 1.0, -RET_CLIP, RET_CLIP).astype(float)
        for t in TARGETS
    }
    y_dir = df["next_direction"].astype(int).to_numpy()
    y_dir_bin = (y_dir > 0).astype(int)

    n = len(df)
    splits = max(3, min(6, n // 120))
    tscv = TimeSeriesSplit(n_splits=splits)
    fold_ids = np.full(n, -1, dtype=int)
    oof_lgb = {t: np.full(n, np.nan, dtype=float) for t in TARGETS}
    oof_cat = {t: np.full(n, np.nan, dtype=float) for t in TARGETS}
    oof_dir_prob = np.full(n, np.nan, dtype=float)

    classes = sorted(set(y_dir.tolist()))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_cls = {i: c for c, i in cls_to_idx.items()}
    y_cls = np.array([cls_to_idx[v] for v in y_dir], dtype=int)

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        close_va = close[va_idx]

        for t in TARGETS:
            lgb_reg = lgb.LGBMRegressor(
                n_estimators=350,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
            lgb_reg.fit(X_tr, y_ret[t][tr_idx])
            ret_pred = lgb_reg.predict(X_va)
            oof_lgb[t][va_idx] = close_va * (1.0 + ret_pred)

            if cat_backend == "catboost":
                cat_reg = CatBoostRegressor(
                    iterations=350,
                    depth=6,
                    learning_rate=0.04,
                    loss_function="RMSE",
                    random_seed=42,
                    verbose=False,
                )
            else:
                cat_reg = CatBoostRegressor(
                    n_estimators=350,
                    max_depth=5,
                    learning_rate=0.04,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                )
            cat_reg.fit(X_tr, y_ret[t][tr_idx])
            ret_pred_cat = cat_reg.predict(X_va)
            oof_cat[t][va_idx] = close_va * (1.0 + np.asarray(ret_pred_cat, dtype=float))

        dir_clf = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        dir_clf.fit(X_tr, y_cls[tr_idx])
        probs = dir_clf.predict_proba(X_va)
        cls_list = dir_clf.classes_.tolist()
        up_encoded = cls_to_idx.get(1, cls_list[-1])
        up_idx = cls_list.index(up_encoded) if up_encoded in cls_list else int(np.argmax(probs.mean(axis=0)))
        oof_dir_prob[va_idx] = probs[:, up_idx]
        fold_ids[va_idx] = fold

    valid = (~np.isnan(oof_dir_prob)).astype(bool)
    for t in TARGETS:
        valid &= ~np.isnan(oof_lgb[t]) & ~np.isnan(oof_cat[t])
    if int(valid.sum()) < 60:
        return {"error": "not_enough_oof_rows", "oof_rows": int(valid.sum())}

    sigma_series = _rolling_sigma(df["close"].astype(float), window=20)
    meta_x = _meta_feature_frame(
        feature_df=X.iloc[valid],
        lgb_pred={t: oof_lgb[t][valid] for t in TARGETS},
        cat_pred={t: oof_cat[t][valid] for t in TARGETS},
        dir_prob_raw=oof_dir_prob[valid],
        sigma_series=sigma_series[valid],
    )
    y_meta_price = {t: y_price[t][valid] for t in TARGETS}
    y_meta_dir_bin = y_dir_bin[valid]

    meta_regs: dict[str, HistGradientBoostingRegressor] = {}
    meta_preds = {}
    for t in TARGETS:
        anchor = meta_x[f"base_mean_{t}"].to_numpy(dtype=float)
        reg = _build_meta_regressor(symbol=symbol, target=t)
        reg.fit(meta_x, y_meta_price[t] - anchor)
        meta_regs[t] = reg
        meta_preds[t] = anchor + reg.predict(meta_x)

    dir_meta = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2500, random_state=42),
    )
    dir_meta.fit(meta_x, y_meta_dir_bin)
    raw_prob = dir_meta.predict_proba(meta_x)[:, 1]
    cal_method, cal_model, cal_prob = fit_direction_calibrator(raw_prob, y_meta_dir_bin)

    residual_bands: dict[str, dict] = {}
    coverage = {}
    width_pct = {}
    for t in TARGETS:
        residual = y_meta_price[t] - meta_preds[t]
        q10 = float(np.quantile(residual, 0.10))
        q90 = float(np.quantile(residual, 0.90))
        low = meta_preds[t] + q10
        high = meta_preds[t] + q90
        cov = float(np.mean((y_meta_price[t] >= low) & (y_meta_price[t] <= high)))
        w = float(np.mean(np.maximum(high - low, 0.0) / (np.abs(y_meta_price[t]) + 1e-9)))
        residual_bands[t] = {"q10": q10, "q90": q90}
        coverage[t] = cov
        width_pct[t] = w

    mae_close = float(mean_absolute_error(y_meta_price["close"], meta_preds["close"]))
    mape_close = float(
        np.mean(np.abs(y_meta_price["close"] - meta_preds["close"]) / (np.abs(y_meta_price["close"]) + 1e-9))
    )
    brier_raw = brier_score(y_meta_dir_bin, raw_prob)
    brier_cal = brier_score(y_meta_dir_bin, cal_prob)
    ece = expected_calibration_error(y_meta_dir_bin, cal_prob, bins=10)

    baseline_row = db.scalar(
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
    baseline_mape = float((baseline_row.metrics or {}).get("mae_close_pct", 1.0)) if baseline_row else 1.0
    pass_dir = ece <= settings.promotion_ece_max and brier_cal <= brier_raw
    pass_price = mape_close <= baseline_mape
    target_cov = settings.promotion_coverage_target
    tol = settings.promotion_coverage_tolerance
    pass_interval = abs(coverage["close"] - target_cov) <= tol
    passed = bool(pass_dir and pass_price and pass_interval)

    version = f"meta_v3_{datetime.now(IST_ZONE).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(settings.model_artifacts_dir) / symbol / version
    paths = _make_model_dirs(out_dir)

    # Train full base models for serving.
    lgb_full = {}
    cat_full = {}
    for t in TARGETS:
        reg = lgb.LGBMRegressor(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        reg.fit(X, y_ret[t])
        reg.booster_.save_model(str(paths["lgbm"] / f"reg_{t}.txt"))
        lgb_full[t] = reg

        if cat_backend == "catboost":
            cat = CatBoostRegressor(
                iterations=350,
                depth=6,
                learning_rate=0.04,
                loss_function="RMSE",
                random_seed=42,
                verbose=False,
            )
        else:
            cat = CatBoostRegressor(
                n_estimators=350,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
        cat.fit(X, y_ret[t])
        if cat_backend == "catboost":
            cat.save_model(str(paths["catboost"] / f"reg_{t}.cbm"))
        else:
            cat.save_model(str(paths["catboost"] / f"reg_{t}.json"))
        cat_full[t] = cat

    dir_full = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    dir_full.fit(X, y_cls)
    dir_full.booster_.save_model(str(paths["direction"] / "clf.txt"))

    # Train meta models on full (base full predictions).
    lgb_full_pred = {
        t: close * (1.0 + np.asarray(lgb_full[t].predict(X), dtype=float))
        for t in TARGETS
    }
    cat_full_pred = {
        t: close * (1.0 + np.asarray(cat_full[t].predict(X), dtype=float))
        for t in TARGETS
    }
    dir_full_probs = dir_full.predict_proba(X)
    base_meta_full = _meta_feature_frame(
        feature_df=X,
        lgb_pred=lgb_full_pred,
        cat_pred=cat_full_pred,
        dir_prob_raw=(
            dir_full_probs[:, list(dir_full.classes_).index(cls_to_idx.get(1, 0))]
            if cls_to_idx.get(1, 0) in list(dir_full.classes_)
            else np.max(dir_full_probs, axis=1)
        ),
        sigma_series=_rolling_sigma(df["close"].astype(float), window=20),
    )
    meta_full_models = {}
    for t in TARGETS:
        anchor = base_meta_full[f"base_mean_{t}"].to_numpy(dtype=float)
        reg = _build_meta_regressor(symbol=symbol, target=t)
        reg.fit(base_meta_full, y_price[t] - anchor)
        meta_full_models[t] = reg
        dump(reg, paths["meta"] / f"reg_{t}.joblib")
    dir_meta_full = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2500, random_state=42),
    )
    dir_meta_full.fit(base_meta_full, y_dir_bin)
    dump(dir_meta_full, paths["meta"] / "dir_meta.joblib")

    # Vol model v3 artifact
    _, vol_metrics = _train_vol_model(df["close"].astype(float), paths["vol"])

    # Calibration artifact and metadata
    save_calibrator(paths["calibration"], cal_method, cal_model)
    meta_json = {
        "feature_columns": feature_cols,
        "meta_feature_columns": meta_x.columns.tolist(),
        "direction_mapping": {str(k): int(v) for k, v in cls_to_idx.items()},
        "direction_inverse_mapping": {str(k): int(v) for k, v in idx_to_cls.items()},
        "calibration_method": cal_method,
        "interval_residual_bands": residual_bands,
        "coverage": coverage,
        "width_pct": width_pct,
        "feature_schema_version": settings.feature_schema_version,
        "label_schema_version": settings.label_schema_version,
        "target_mode": "return_vs_close",
        "catboost_backend": cat_backend,
        "meta_regression_mode": META_REGRESSION_MODE,
    }
    (out_dir / "meta_v3.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")
    write_training_metadata(
        out_dir / "training_metadata.json",
        symbol=symbol,
        quality=quality,
        window_from=df["session_date"].min(),
        window_to=df["session_date"].max(),
    )

    metrics = {
        "mae_close": mae_close,
        "mae_close_pct": mape_close,
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "ece": ece,
        "coverage_close": coverage["close"],
        "width_pct_close": width_pct["close"],
        "baseline_mape_close": baseline_mape,
        "pass_dir": pass_dir,
        "pass_price": pass_price,
        "pass_interval": pass_interval,
        "promotion_passed": passed,
        **vol_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    run_id = f"{symbol.replace(' ', '_')}_{version}"
    _store_oof_predictions(
        db=db,
        run_id=run_id,
        symbol=symbol,
        session_dates=df["session_date"].to_numpy()[valid],
        pred_close=np.asarray(meta_preds["close"], dtype=float),
        direction_prob=np.asarray(cal_prob, dtype=float),
        actual_close=np.asarray(y_meta_price["close"], dtype=float),
        actual_dir=np.asarray(y_dir[valid], dtype=int),
        fold_ids=fold_ids[valid],
    )

    # Calibration registry
    if passed:
        db.execute(
            update(CalibrationRegistry)
            .where(
                and_(
                    CalibrationRegistry.symbol == symbol,
                    CalibrationRegistry.model_family == "meta_v3",
                )
            )
            .values(is_active=False)
        )
    cal_version = f"cal_{datetime.now(IST_ZONE).strftime('%Y%m%d_%H%M%S')}"
    db.add(
        CalibrationRegistry(
            symbol=symbol,
            model_family="meta_v3",
            calibration_version=cal_version,
            method=cal_method,
            artifact_path=str(paths["calibration"]),
            metrics={"ece": ece, "brier": brier_cal},
            is_active=passed,
        )
    )

    db.add(
        BacktestRun(
            run_id=run_id,
            symbol=symbol,
            model_family="meta_v3",
            window_from=df["session_date"].min(),
            window_to=df["session_date"].max(),
            metrics=metrics,
            passed_gates=passed,
        )
    )
    for metric_name, metric_value in (
        ("ece", ece),
        ("brier_calibrated", brier_cal),
        ("mae_close_pct", mape_close),
        ("coverage_close", coverage["close"]),
    ):
        db.add(
            DriftMetric(
                symbol=symbol,
                model_family="meta_v3",
                metric_name=metric_name,
                metric_value=float(metric_value),
                details={"run_id": run_id, "version": version},
            )
        )

    _update_registry(
        db=db,
        symbol=symbol,
        version=version,
        out_dir=out_dir,
        metrics=metrics,
        date_from=df["session_date"].min(),
        date_to=df["session_date"].max(),
        passed=passed,
    )
    db.commit()
    logger.info("Trained meta_v3 symbol=%s version=%s passed=%s", symbol, version, passed)
    return {"symbol": symbol, "version": version, "metrics": metrics, "passed": passed}


def train_all_symbols(db: Session, symbols: list[str]) -> dict:
    out = {}
    for symbol in symbols:
        try:
            out[symbol] = train_symbol_model(db, symbol)
        except Exception as exc:
            logger.exception("meta_v3 training failed symbol=%s: %s", symbol, exc)
            out[symbol] = {"error": str(exc)}
    return out
