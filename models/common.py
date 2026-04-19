from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import FeaturesDaily, LabelsDaily


def load_feature_label_frame(db: Session, symbol: str) -> pd.DataFrame:
    f_rows = db.execute(select(FeaturesDaily).where(FeaturesDaily.symbol == symbol)).scalars().all()
    l_rows = db.execute(select(LabelsDaily).where(LabelsDaily.symbol == symbol)).scalars().all()
    f_df = pd.DataFrame([{"session_date": r.session_date, **(r.features or {})} for r in f_rows])
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
    merged = f_df.merge(l_df, on="session_date", how="inner").sort_values("session_date")
    return merged.dropna()


def load_feature_frame(db: Session, symbol: str) -> pd.DataFrame:
    f_rows = db.execute(select(FeaturesDaily).where(FeaturesDaily.symbol == symbol)).scalars().all()
    f_df = pd.DataFrame([{"session_date": r.session_date, **(r.features or {})} for r in f_rows])
    if f_df.empty:
        return pd.DataFrame()
    return f_df.sort_values("session_date").dropna(how="all")


def latest_feature_row(db: Session, symbol: str):
    return db.scalar(
        select(FeaturesDaily)
        .where(FeaturesDaily.symbol == symbol)
        .order_by(FeaturesDaily.session_date.desc())
        .limit(1)
    )


def latest_feature_rows(db: Session, symbol: str, limit: int) -> list:
    rows = (
        db.execute(
            select(FeaturesDaily)
            .where(FeaturesDaily.symbol == symbol)
            .order_by(FeaturesDaily.session_date.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return list(reversed(rows))
