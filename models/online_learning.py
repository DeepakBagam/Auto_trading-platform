"""Online learning for rapid model adaptation."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from db.models import FeaturesDaily, LabelsDaily, ModelRegistry
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)


class OnlineMetaModel:
    """Online learning wrapper for meta model using SGD."""
    
    def __init__(self, symbol: str, artifact_dir: Path):
        self.symbol = symbol
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SGD models
        self.regressors = {
            "open": SGDRegressor(
                loss="huber",
                penalty="l2",
                alpha=0.001,
                learning_rate="adaptive",
                eta0=0.01,
                random_state=42,
            ),
            "high": SGDRegressor(
                loss="huber",
                penalty="l2",
                alpha=0.001,
                learning_rate="adaptive",
                eta0=0.01,
                random_state=42,
            ),
            "low": SGDRegressor(
                loss="huber",
                penalty="l2",
                alpha=0.001,
                learning_rate="adaptive",
                eta0=0.01,
                random_state=42,
            ),
            "close": SGDRegressor(
                loss="huber",
                penalty="l2",
                alpha=0.001,
                learning_rate="adaptive",
                eta0=0.01,
                random_state=42,
            ),
        }
        
        self.classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.001,
            learning_rate="adaptive",
            eta0=0.01,
            random_state=42,
        )
        
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        self.update_count = 0
        self.last_update = None
    
    def load_from_disk(self) -> bool:
        """Load existing online model from disk."""
        try:
            for target in ["open", "high", "low", "close"]:
                model_path = self.artifact_dir / f"online_reg_{target}.joblib"
                if model_path.exists():
                    self.regressors[target] = load(model_path)
            
            clf_path = self.artifact_dir / "online_clf.joblib"
            if clf_path.exists():
                self.classifier = load(clf_path)
            
            scaler_path = self.artifact_dir / "online_scaler.joblib"
            if scaler_path.exists():
                self.scaler = load(scaler_path)
            
            meta_path = self.artifact_dir / "online_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.feature_columns = meta.get("feature_columns", [])
                self.is_fitted = meta.get("is_fitted", False)
                self.update_count = meta.get("update_count", 0)
                self.last_update = (
                    datetime.fromisoformat(meta["last_update"])
                    if meta.get("last_update")
                    else None
                )
            
            return self.is_fitted
        except Exception as exc:
            logger.warning("Failed to load online model for %s: %s", self.symbol, exc)
            return False
    
    def save_to_disk(self) -> None:
        """Save online model to disk."""
        try:
            for target, model in self.regressors.items():
                dump(model, self.artifact_dir / f"online_reg_{target}.joblib")
            
            dump(self.classifier, self.artifact_dir / "online_clf.joblib")
            dump(self.scaler, self.artifact_dir / "online_scaler.joblib")
            
            meta = {
                "feature_columns": self.feature_columns,
                "is_fitted": self.is_fitted,
                "update_count": self.update_count,
                "last_update": self.last_update.isoformat() if self.last_update else None,
            }
            (self.artifact_dir / "online_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.error("Failed to save online model for %s: %s", self.symbol, exc)
    
    def partial_fit(self, X: np.ndarray, y_reg: dict, y_cls: np.ndarray) -> None:
        """
        Incrementally update model with new data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y_reg: Dict of target arrays {"open": [...], "high": [...], ...}
            y_cls: Direction labels (n_samples,)
        """
        if not self.is_fitted:
            # First fit - initialize scaler
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        
        # Update regressors
        for target in ["open", "high", "low", "close"]:
            self.regressors[target].partial_fit(X_scaled, y_reg[target])
        
        # Update classifier
        classes = np.array([-1, 0, 1])  # Possible direction classes
        self.classifier.partial_fit(X_scaled, y_cls, classes=classes)
        
        self.update_count += 1
        self.last_update = datetime.now(IST_ZONE)
    
    def predict(self, X: np.ndarray) -> dict:
        """Make predictions with online model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for target in ["open", "high", "low", "close"]:
            predictions[f"pred_{target}"] = self.regressors[target].predict(X_scaled)
        
        direction_probs = self.classifier.predict_proba(X_scaled)
        predictions["direction_prob"] = direction_probs[:, 2]  # Probability of UP (class 1)
        predictions["direction"] = self.classifier.predict(X_scaled)
        
        return predictions


def load_recent_data(
    db: Session,
    symbol: str,
    lookback_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load recent features and labels for online learning."""
    cutoff = datetime.now(IST_ZONE).date() - timedelta(days=lookback_days)
    
    features = db.execute(
        select(FeaturesDaily).where(
            and_(
                FeaturesDaily.symbol == symbol,
                FeaturesDaily.session_date >= cutoff,
            )
        ).order_by(FeaturesDaily.session_date.asc())
    ).scalars().all()
    
    labels = db.execute(
        select(LabelsDaily).where(
            and_(
                LabelsDaily.symbol == symbol,
                LabelsDaily.session_date >= cutoff,
            )
        ).order_by(LabelsDaily.session_date.asc())
    ).scalars().all()
    
    if not features or not labels:
        return pd.DataFrame(), pd.DataFrame()
    
    feat_df = pd.DataFrame([
        {"session_date": f.session_date, **(f.features or {})}
        for f in features
    ])
    
    label_df = pd.DataFrame([
        {
            "session_date": l.session_date,
            "next_open": l.next_open,
            "next_high": l.next_high,
            "next_low": l.next_low,
            "next_close": l.next_close,
            "next_direction": l.next_direction,
        }
        for l in labels
    ])
    
    return feat_df, label_df


def update_online_model(db: Session, symbol: str) -> dict:
    """
    Update online model with recent data.
    
    This should be called daily after market close.
    """
    settings = get_settings()
    artifact_dir = Path(settings.model_artifacts_dir) / symbol / "online"
    
    # Load or create online model
    model = OnlineMetaModel(symbol, artifact_dir)
    model.load_from_disk()
    
    # Load recent data (last 7 days)
    feat_df, label_df = load_recent_data(db, symbol, lookback_days=7)
    
    if feat_df.empty or label_df.empty:
        return {"status": "no_data", "symbol": symbol}
    
    # Merge features and labels
    merged = feat_df.merge(label_df, on="session_date", how="inner")
    merged = merged.dropna()
    
    if len(merged) < 3:
        return {"status": "insufficient_data", "symbol": symbol, "rows": len(merged)}
    
    # Prepare data
    feature_cols = [
        c for c in merged.columns
        if c not in {"session_date", "next_open", "next_high", "next_low", "next_close", "next_direction"}
    ]
    
    if not model.feature_columns:
        model.feature_columns = feature_cols
    
    X = merged[feature_cols].astype(float).to_numpy()
    y_reg = {
        "open": merged["next_open"].astype(float).to_numpy(),
        "high": merged["next_high"].astype(float).to_numpy(),
        "low": merged["next_low"].astype(float).to_numpy(),
        "close": merged["next_close"].astype(float).to_numpy(),
    }
    y_cls = merged["next_direction"].astype(int).to_numpy()
    
    # Update model
    model.partial_fit(X, y_reg, y_cls)
    model.save_to_disk()
    
    logger.info(
        "Updated online model for %s: update_count=%s, rows=%s",
        symbol,
        model.update_count,
        len(merged),
    )
    
    return {
        "status": "updated",
        "symbol": symbol,
        "update_count": model.update_count,
        "rows_processed": len(merged),
        "last_update": model.last_update.isoformat() if model.last_update else None,
    }


def should_update_online_model(db: Session, symbol: str) -> bool:
    """
    Determine if online model should be updated.
    
    Update if:
    - Last update was more than 1 day ago
    - New data is available
    """
    settings = get_settings()
    artifact_dir = Path(settings.model_artifacts_dir) / symbol / "online"
    meta_path = artifact_dir / "online_meta.json"
    
    if not meta_path.exists():
        return True  # No model exists, should create
    
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        last_update_str = meta.get("last_update")
        
        if not last_update_str:
            return True
        
        last_update = datetime.fromisoformat(last_update_str)
        hours_since_update = (datetime.now(IST_ZONE) - last_update).total_seconds() / 3600
        
        # Update if more than 24 hours since last update
        return hours_since_update > 24
    except Exception:
        return True
