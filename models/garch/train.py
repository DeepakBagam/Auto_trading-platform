from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sqlalchemy import and_, select, update
from sqlalchemy.orm import Session

from db.models import ModelRegistry, RawCandle
from utils.config import get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)


def _load_returns(db: Session, symbol: str) -> pd.Series:
    rows = (
        db.execute(
            select(RawCandle)
            .where(and_(RawCandle.interval == "day", RawCandle.instrument_key.like(f"%{symbol}%")))
            .order_by(RawCandle.ts.asc())
        )
        .scalars()
        .all()
    )
    close = pd.Series([r.close for r in rows], dtype=float)
    ret = close.pct_change().dropna()
    return ret


def _ewma_sigma(returns: np.ndarray, lam: float, sigma0: float | None = None) -> np.ndarray:
    out = np.zeros_like(returns, dtype=float)
    if len(returns) == 0:
        return out
    var = (sigma0**2) if sigma0 is not None else float(np.var(returns[: min(20, len(returns))]))
    for i, r in enumerate(returns):
        var = lam * var + (1 - lam) * (r**2)
        out[i] = float(np.sqrt(max(var, 1e-12)))
    return out


def train_symbol_model(db: Session, symbol: str) -> dict:
    ret = _load_returns(db, symbol)
    if len(ret) < 80:
        logger.warning("Not enough daily candles for garch_v2 symbol=%s rows=%s", symbol, len(ret))
        return {}

    arr = ret.to_numpy()
    split = int(len(arr) * 0.8)
    train_arr, test_arr = arr[:split], arr[split:]

    best_lam = 0.94
    best_mae = float("inf")
    realized = np.abs(test_arr)
    for lam in (0.90, 0.92, 0.94, 0.96, 0.98):
        sigmas = _ewma_sigma(arr[: split + len(test_arr)], lam=lam)
        pred = sigmas[split:]
        mae = float(mean_absolute_error(realized, pred))
        if mae < best_mae:
            best_mae = mae
            best_lam = lam

    all_sigma = _ewma_sigma(arr, lam=best_lam)
    model_payload = {
        "method": "ewma_garch_proxy",
        "lambda": best_lam,
        "last_sigma": float(all_sigma[-1]),
        "train_std": float(np.std(train_arr)),
    }
    metrics = {"mae_abs_return_vs_sigma": best_mae}

    settings = get_settings()
    version = f"garch_{datetime.now(IST_ZONE).strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(settings.model_artifacts_dir) / symbol / version
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.json").write_text(json.dumps(model_payload), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    db.execute(
        update(ModelRegistry)
        .where(and_(ModelRegistry.model_name == "garch_v2", ModelRegistry.symbol == symbol))
        .values(is_active=False)
    )
    db.add(
        ModelRegistry(
            model_name="garch_v2",
            model_version=version,
            model_type="ewma_volatility",
            symbol=symbol,
            artifact_path=str(out_dir),
            metrics=metrics,
            trained_from=datetime.now(IST_ZONE).date(),
            trained_to=datetime.now(IST_ZONE).date(),
            is_active=True,
        )
    )
    db.commit()
    logger.info("Trained garch_v2 symbol=%s version=%s", symbol, version)
    return {"symbol": symbol, "version": version, "metrics": metrics}


def train_all_symbols(db: Session, symbols: list[str]) -> dict:
    out = {}
    for symbol in symbols:
        try:
            out[symbol] = train_symbol_model(db, symbol)
        except Exception as exc:
            logger.exception("garch_v2 training failed symbol=%s: %s", symbol, exc)
            out[symbol] = {"error": str(exc)}
    return out
