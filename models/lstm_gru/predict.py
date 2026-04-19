from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from db.models import ModelRegistry
from models.common import latest_feature_rows
from models.lstm_gru.model import build_lstm_gru_model, require_torch
from utils.constants import DIRECTION_BUY, DIRECTION_HOLD, DIRECTION_SELL

CLASS_TO_DIR = {0: -1, 1: 0, 2: 1}
DIR_TO_LABEL = {-1: DIRECTION_SELL, 0: DIRECTION_HOLD, 1: DIRECTION_BUY}


def _latest_model(db: Session, symbol: str) -> ModelRegistry | None:
    return db.scalar(
        select(ModelRegistry)
        .where(
            and_(
                ModelRegistry.model_name == "lstm_gru_v2",
                ModelRegistry.symbol == symbol,
                ModelRegistry.is_active.is_(True),
            )
        )
        .order_by(ModelRegistry.created_at.desc())
        .limit(1)
    )


def _load_bundle(artifact_path: str):
    torch, _ = require_torch()
    path = Path(artifact_path)
    feature_cols = json.loads((path / "feature_columns.json").read_text(encoding="utf-8"))
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    model = build_lstm_gru_model(
        input_size=len(feature_cols),
        hidden_size=int(meta.get("hidden_size", 64)),
        num_layers=int(meta.get("num_layers", 2)),
    )
    state_dict = torch.load(path / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    scaler = load(path / "scaler.joblib")
    return model, scaler, feature_cols, int(meta.get("seq_len", 20))


def predict_from_feature_frame(db: Session, symbol: str, feature_df: pd.DataFrame) -> pd.DataFrame:
    model_row = _latest_model(db, symbol)
    if model_row is None or feature_df.empty:
        return pd.DataFrame()
    torch, nn = require_torch()
    model, scaler, feature_cols, seq_len = _load_bundle(model_row.artifact_path)
    frame = feature_df.copy().sort_values("session_date")
    X = frame[feature_cols].astype(float).to_numpy()
    X_scaled = scaler.transform(X)
    preds = []
    for i in range(seq_len - 1, len(X_scaled)):
        seq = X_scaled[i - seq_len + 1 : i + 1]
        x_t = torch.tensor(seq[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            out_reg, out_cls = model(x_t)
            out_prob = nn.functional.softmax(out_cls, dim=1).cpu().numpy()[0]
        dir_class = int(np.argmax(out_prob))
        preds.append(
            {
                "session_date": frame.iloc[i]["session_date"],
                "pred_open": float(out_reg.cpu().numpy()[0][0]),
                "pred_high": float(out_reg.cpu().numpy()[0][1]),
                "pred_low": float(out_reg.cpu().numpy()[0][2]),
                "pred_close": float(out_reg.cpu().numpy()[0][3]),
                "direction_code": CLASS_TO_DIR.get(dir_class, 0),
                "direction_prob": float(out_prob[dir_class]),
                "model_version": model_row.model_version,
            }
        )
    return pd.DataFrame(preds)


def predict_symbol(db: Session, symbol: str) -> dict:
    model_row = _latest_model(db, symbol)
    if model_row is None:
        raise ValueError(f"Missing lstm_gru_v2 model for symbol={symbol}")
    model, scaler, feature_cols, seq_len = _load_bundle(model_row.artifact_path)
    rows = latest_feature_rows(db, symbol, limit=seq_len)
    if len(rows) < seq_len:
        raise ValueError(f"Need at least {seq_len} feature rows for symbol={symbol}")
    values = [{k: float((r.features or {}).get(k, 0.0)) for k in feature_cols} for r in rows]
    X = pd.DataFrame(values)
    X_scaled = scaler.transform(X)

    torch, nn = require_torch()
    x_t = torch.tensor(X_scaled[None, :, :], dtype=torch.float32)
    with torch.no_grad():
        reg, cls = model(x_t)
        probs = nn.functional.softmax(cls, dim=1).cpu().numpy()[0]
    reg_np = reg.cpu().numpy()[0]
    cls_ix = int(np.argmax(probs))
    direction_code = CLASS_TO_DIR.get(cls_ix, 0)

    return {
        "pred_open": float(reg_np[0]),
        "pred_high": float(reg_np[1]),
        "pred_low": float(reg_np[2]),
        "pred_close": float(reg_np[3]),
        "direction_code": direction_code,
        "direction": DIR_TO_LABEL[direction_code],
        "direction_prob": float(probs[cls_ix]),
        "model_version": model_row.model_version,
        "metrics": model_row.metrics or {},
    }
