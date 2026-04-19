from __future__ import annotations

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from db.models import LabelsDaily, RawCandle
from utils.logger import get_logger

logger = get_logger(__name__)


def _symbol_from_instrument(instrument_key: str) -> str:
    if "|" in instrument_key:
        return instrument_key.split("|", 1)[1]
    return instrument_key


def build_daily_labels_for_symbol(db: Session, instrument_key: str) -> int:
    symbol = _symbol_from_instrument(instrument_key)
    rows = db.execute(
        select(RawCandle)
        .where(and_(RawCandle.instrument_key == instrument_key, RawCandle.interval == "day"))
        .order_by(RawCandle.ts.asc())
    ).scalars().all()
    if len(rows) < 2:
        return 0
    df = pd.DataFrame(
        [
            {
                "ts": r.ts,
                "session_date": r.ts.date(),
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
            }
            for r in rows
        ]
    ).sort_values(["session_date", "ts"])
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["session_date"], keep="last")
    if before_dedup != len(df):
        logger.warning(
            "Dropped %s duplicate daily rows for %s before label generation",
            before_dedup - len(df),
            symbol,
        )

    df["next_open"] = df["open"].shift(-1)
    df["next_high"] = df["high"].shift(-1)
    df["next_low"] = df["low"].shift(-1)
    df["next_close"] = df["close"].shift(-1)
    diff = df["next_close"] - df["close"]
    df["next_direction"] = diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df = df.dropna()
    inserted = 0

    existing_dates = set(
        db.scalars(select(LabelsDaily.session_date).where(LabelsDaily.symbol == symbol)).all()
    )

    for _, row in df.iterrows():
        session_date = row["session_date"]
        if session_date in existing_dates:
            continue
        db.add(
            LabelsDaily(
                symbol=symbol,
                session_date=session_date,
                next_open=float(row["next_open"]),
                next_high=float(row["next_high"]),
                next_low=float(row["next_low"]),
                next_close=float(row["next_close"]),
                next_direction=int(row["next_direction"]),
            )
        )
        existing_dates.add(session_date)
        inserted += 1
    db.commit()
    logger.info("Built %s label rows for %s", inserted, symbol)
    return inserted


def build_daily_labels(db: Session, instrument_keys: list[str]) -> dict:
    summary = {}
    for key in instrument_keys:
        try:
            summary[key] = build_daily_labels_for_symbol(db, key)
        except Exception as exc:
            db.rollback()
            logger.exception("Label build failed for %s: %s", key, exc)
            summary[key] = 0
    return summary
