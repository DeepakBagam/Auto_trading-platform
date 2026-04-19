from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from db.models import ModelRegistry, RawCandle


def _latest_model(db: Session, symbol: str) -> ModelRegistry | None:
    return db.scalar(
        select(ModelRegistry)
        .where(
            and_(
                ModelRegistry.model_name == "garch_v2",
                ModelRegistry.symbol == symbol,
                ModelRegistry.is_active.is_(True),
            )
        )
        .order_by(ModelRegistry.created_at.desc())
        .limit(1)
    )


def _latest_daily_returns(db: Session, symbol: str) -> tuple[float, float]:
    rows = (
        db.execute(
            select(RawCandle)
            .where(and_(RawCandle.interval == "day", RawCandle.instrument_key.like(f"%{symbol}%")))
            .order_by(RawCandle.ts.desc())
            .limit(2)
        )
        .scalars()
        .all()
    )
    if len(rows) < 2:
        return 0.0, 0.0
    c0, c1 = float(rows[0].close), float(rows[1].close)
    last_return = (c0 / (c1 + 1e-9)) - 1.0
    return c0, last_return


def predict_symbol(db: Session, symbol: str) -> dict:
    model_row = _latest_model(db, symbol)
    if model_row is None:
        raise ValueError(f"Missing garch_v2 model for symbol={symbol}")
    model = json.loads(Path(model_row.artifact_path, "model.json").read_text(encoding="utf-8"))
    lam = float(model.get("lambda", 0.94))
    prev_sigma = float(model.get("last_sigma", model.get("train_std", 0.01)))
    last_close, last_ret = _latest_daily_returns(db, symbol)
    new_var = lam * (prev_sigma**2) + (1 - lam) * (last_ret**2)
    pred_sigma = float(max(new_var, 1e-12) ** 0.5)
    range_width = float(last_close * pred_sigma * 1.75) if last_close > 0 else 0.0
    return {
        "pred_sigma": pred_sigma,
        "pred_range_width": range_width,
        "model_version": model_row.model_version,
        "metrics": model_row.metrics or {},
    }
