from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sqlalchemy import and_, update
from sqlalchemy.orm import Session

from db.models import ModelRegistry
from models.common import load_feature_label_frame
from models.lstm_gru.model import build_lstm_gru_model, require_torch
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)

DIR_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_DIR = {0: -1, 1: 0, 2: 1}


def _make_sequences(X: np.ndarray, y_reg: np.ndarray, y_dir: np.ndarray, seq_len: int):
    xs, ys_reg, ys_dir = [], [], []
    for i in range(seq_len - 1, len(X)):
        xs.append(X[i - seq_len + 1 : i + 1])
        ys_reg.append(y_reg[i])
        ys_dir.append(DIR_TO_CLASS.get(int(y_dir[i]), 1))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys_reg, dtype=np.float32), np.asarray(ys_dir)


def train_symbol_model(db: Session, symbol: str, seq_len: int = 20, epochs: int = 40) -> dict:
    torch, nn = require_torch()
    df = load_feature_label_frame(db, symbol)
    if len(df) < max(80, seq_len + 10):
        logger.warning("Not enough rows for lstm_gru_v2 symbol=%s rows=%s", symbol, len(df))
        return {}

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
    X_raw = df[feature_cols].astype(float).to_numpy()
    y_reg = df[["next_open", "next_high", "next_low", "next_close"]].astype(float).to_numpy()
    y_dir = df["next_direction"].astype(int).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_seq, y_seq_reg, y_seq_dir = _make_sequences(X_scaled, y_reg, y_dir, seq_len=seq_len)
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train_reg, y_test_reg = y_seq_reg[:split_idx], y_seq_reg[split_idx:]
    y_train_dir, y_test_dir = y_seq_dir[:split_idx], y_seq_dir[split_idx:]

    model = build_lstm_gru_model(input_size=len(feature_cols), hidden_size=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_reg = nn.MSELoss()
    loss_cls = nn.CrossEntropyLoss()

    x_t = torch.tensor(X_train, dtype=torch.float32)
    y_reg_t = torch.tensor(y_train_reg, dtype=torch.float32)
    y_dir_t = torch.tensor(y_train_dir, dtype=torch.long)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out_reg, out_cls = model(x_t)
        loss = loss_reg(out_reg, y_reg_t) + 0.3 * loss_cls(out_cls, y_dir_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_reg, pred_cls_logits = model(torch.tensor(X_test, dtype=torch.float32))
    pred_reg_np = pred_reg.detach().cpu().numpy()
    pred_cls_np = np.argmax(pred_cls_logits.detach().cpu().numpy(), axis=1)

    metrics = {
        "mae_open": float(mean_absolute_error(y_test_reg[:, 0], pred_reg_np[:, 0])),
        "mae_high": float(mean_absolute_error(y_test_reg[:, 1], pred_reg_np[:, 1])),
        "mae_low": float(mean_absolute_error(y_test_reg[:, 2], pred_reg_np[:, 2])),
        "mae_close": float(mean_absolute_error(y_test_reg[:, 3], pred_reg_np[:, 3])),
        "accuracy_direction": float(accuracy_score(y_test_dir, pred_cls_np)),
    }

    settings = get_settings()
    version = f"lstm_gru_{datetime.now(IST_ZONE).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(settings.model_artifacts_dir) / symbol / version
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")
    dump(scaler, out_dir / "scaler.joblib")
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_cols), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps({"seq_len": seq_len, "hidden_size": 64, "num_layers": 2}), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    db.execute(
        update(ModelRegistry)
        .where(and_(ModelRegistry.model_name == "lstm_gru_v2", ModelRegistry.symbol == symbol))
        .values(is_active=False)
    )
    db.add(
        ModelRegistry(
            model_name="lstm_gru_v2",
            model_version=version,
            model_type="torch_lstm_gru",
            symbol=symbol,
            artifact_path=str(out_dir),
            metrics=metrics,
            trained_from=df["session_date"].min(),
            trained_to=df["session_date"].max(),
            is_active=True,
        )
    )
    db.commit()
    logger.info("Trained lstm_gru_v2 symbol=%s version=%s", symbol, version)
    return {"symbol": symbol, "version": version, "metrics": metrics}


def train_all_symbols(db: Session, symbols: list[str], seq_len: int = 20, epochs: int = 40) -> dict:
    out = {}
    for symbol in symbols:
        try:
            out[symbol] = train_symbol_model(db, symbol, seq_len=seq_len, epochs=epochs)
        except Exception as exc:
            logger.exception("lstm_gru_v2 training failed symbol=%s: %s", symbol, exc)
            out[symbol] = {"error": str(exc)}
    return out
