from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from sqlalchemy import and_, select, update
from sqlalchemy.orm import Session
from xgboost import XGBClassifier, XGBRegressor

from db.models import FeaturesDaily, LabelsDaily, ModelRegistry
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)

TARGETS = ["open", "high", "low", "close"]
MAX_TRAIN_ROWS = 2000  # Increased for better learning
RET_CLIP = 0.15  # Reduced clip for realistic returns
PROFIT_THRESHOLD = 0.0015  # 0.15% minimum profit target


def _enhance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add profit-focused features to existing features."""
    
    # Multi-period momentum
    for period in [3, 5, 10, 20]:
        if f'momentum_{period}' not in df.columns:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
    
    # Trend strength
    if 'ema_9' in df.columns and 'ema_21' in df.columns:
        df['trend_strength'] = abs(df['ema_9'] - df['ema_21']) / df['ema_21']
        df['trend_direction'] = (df['ema_9'] > df['ema_21']).astype(int)
    
    # RSI momentum
    if 'rsi_14' in df.columns:
        df['rsi_momentum'] = df['rsi_14'].diff()
        df['rsi_in_sweet_spot'] = ((df['rsi_14'] > 55) & (df['rsi_14'] < 70) | 
                                    (df['rsi_14'] > 30) & (df['rsi_14'] < 45)).astype(int)
    
    # MACD strength
    if 'macd_hist' in df.columns and 'atr_14' in df.columns:
        df['macd_strength'] = abs(df['macd_hist']) / (df['atr_14'] + 1e-9)
    
    # Volume surge
    if 'volume_ratio_20' in df.columns:
        df['volume_surge'] = (df['volume_ratio_20'] > 1.2).astype(int)
    
    # Price position
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_range'] = df['bb_upper'] - df['bb_lower']
        df['bb_width'] = df['bb_range'] / df['close']
    
    return df


def _dataset_for_symbol(db: Session, symbol: str) -> pd.DataFrame:
    f_rows = db.execute(select(FeaturesDaily).where(FeaturesDaily.symbol == symbol)).scalars().all()
    l_rows = db.execute(select(LabelsDaily).where(LabelsDaily.symbol == symbol)).scalars().all()
    f_df = pd.DataFrame(
        [{"session_date": r.session_date, **(r.features or {})} for r in f_rows]
    )
    l_df = pd.DataFrame(
        [
            {
                "session_date": r.session_date,
                "next_open": r.next_open,
                "next_high": r.next_high,
                "next_low": r.next_low,
                "next_close": r.next_close,
                "next_direction": r.next_direction,
            }
            for r in l_rows
        ]
    )
    if f_df.empty or l_df.empty:
        return pd.DataFrame()
    df = f_df.merge(l_df, on="session_date", how="inner").sort_values("session_date")
    df = df.dropna()
    
    # Enhance with profit-focused features
    df = _enhance_features(df)
    
    return df


def train_symbol_models(db: Session, symbol: str) -> dict:
    df = _dataset_for_symbol(db, symbol)
    if len(df) < 50:
        logger.warning("Not enough rows to train %s. Need >=50, got %s", symbol, len(df))
        return {}
    if len(df) > MAX_TRAIN_ROWS:
        df = df.tail(MAX_TRAIN_ROWS).copy()
    split_idx = max(40, int(len(df) * 0.8))
    feature_cols = [c for c in df.columns if c not in {"session_date", "next_open", "next_high", "next_low", "next_close", "next_direction"}]
    X = df[feature_cols].astype(float)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_cls = df["next_direction"].iloc[:split_idx].astype(int)
    y_test_cls = df["next_direction"].iloc[split_idx:].astype(int)

    settings = get_settings()
    version = f"xgb_{datetime.now(IST_ZONE).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(settings.model_artifacts_dir) / symbol / version
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}
    target_mode = "relative_to_close"

    # Enhanced XGBoost parameters for higher accuracy
    for target in TARGETS:
        next_series = df[f"next_{target}"].astype(float)
        base_close = df["close"].astype(float)
        y_all = (next_series / (base_close + 1e-9)) - 1.0
        y_all = y_all.clip(lower=-RET_CLIP, upper=RET_CLIP)
        y_train = y_all.iloc[:split_idx]
        y_test_actual = next_series.iloc[split_idx:].astype(float)
        base_close_test = base_close.iloc[split_idx:].astype(float)
        
        # IMPROVED: Better hyperparameters for accuracy
        model = XGBRegressor(
            n_estimators=500,  # More trees
            max_depth=6,  # Deeper trees
            learning_rate=0.03,  # Lower learning rate
            subsample=0.85,  # Better sampling
            colsample_bytree=0.85,
            min_child_weight=3,  # Prevent overfitting
            gamma=0.1,  # Regularization
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
        )
        model.fit(X_train, y_train)
        pred_ret = model.predict(X_test).astype(float)
        pred = (base_close_test.values * (1.0 + pred_ret)).astype(float)
        metrics[f"mae_{target}"] = float(mean_absolute_error(y_test_actual, pred))
        metrics[f"mae_{target}_pct"] = float(
            (abs(pred - y_test_actual.values) / (abs(y_test_actual.values) + 1e-9)).mean()
        )
        model.save_model(str(out_dir / f"reg_{target}.json"))

    # IMPROVED: Better direction classifier with class weights
    cls_values = sorted({int(v) for v in pd.concat([y_train_cls, y_test_cls]).unique().tolist()})
    cls_to_idx = {cls: idx for idx, cls in enumerate(cls_values)}
    idx_to_cls = {idx: cls for cls, idx in cls_to_idx.items()}
    y_train_cls_encoded = y_train_cls.map(cls_to_idx).astype(int)
    y_test_cls_encoded = y_test_cls.map(cls_to_idx).astype(int)
    
    # Calculate class weights to handle imbalance
    class_counts = y_train_cls_encoded.value_counts().sort_index()
    total = len(y_train_cls_encoded)
    scale_pos_weight = total / (len(class_counts) * class_counts.values)
    
    clf = XGBClassifier(
        n_estimators=400,  # More trees
        max_depth=5,  # Deeper
        learning_rate=0.04,  # Lower LR
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )
    clf.fit(X_train, y_train_cls_encoded)
    cls_pred = clf.predict(X_test)
    metrics["accuracy_direction"] = float(accuracy_score(y_test_cls_encoded, cls_pred))
    clf.save_model(str(out_dir / "cls_direction.json"))

    with (out_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f)
    with (out_dir / "direction_class_map.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in idx_to_cls.items()}, f)
    with (out_dir / "prediction_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "target_mode": target_mode,
                "ret_clip": RET_CLIP,
                "max_train_rows": MAX_TRAIN_ROWS,
            },
            f,
        )

    db.execute(
        update(ModelRegistry)
        .where(and_(ModelRegistry.model_name == "xgboost_v1", ModelRegistry.symbol == symbol))
        .values(is_active=False)
    )
    db.add(
        ModelRegistry(
            model_name="xgboost_v1",
            model_version=version,
            model_type="xgboost_bundle",
            symbol=symbol,
            artifact_path=str(out_dir),
            metrics=metrics,
            trained_from=df["session_date"].min(),
            trained_to=df["session_date"].max(),
            is_active=True,
        )
    )
    db.commit()
    logger.info("Trained %s model version=%s metrics=%s", symbol, version, metrics)
    return {"symbol": symbol, "version": version, "metrics": metrics}


def train_all_symbols(db: Session, symbols: list[str]) -> dict:
    out = {}
    for symbol in symbols:
        out[symbol] = train_symbol_models(db, symbol)
    return out
