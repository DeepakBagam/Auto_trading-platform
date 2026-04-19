from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from db.models import DataFreshness, FeaturesDaily, LabelsDaily, RawCandle
from utils.calendar_utils import is_trading_day
from utils.config import get_settings
from utils.constants import IST_ZONE


@dataclass
class DataQualityReport:
    symbol: str
    rows: int
    duplicates_features: int
    duplicates_labels: int
    missing_candle_ratio: float
    stale_sources: list[str]
    passed: bool
    details: dict


def _load_feature_df(db: Session, symbol: str) -> pd.DataFrame:
    rows = db.execute(select(FeaturesDaily).where(FeaturesDaily.symbol == symbol)).scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{"session_date": r.session_date, **(r.features or {})} for r in rows])


def _load_label_df(db: Session, symbol: str) -> pd.DataFrame:
    rows = db.execute(select(LabelsDaily).where(LabelsDaily.symbol == symbol)).scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "session_date": r.session_date,
                "next_open": r.next_open,
                "next_high": r.next_high,
                "next_low": r.next_low,
                "next_close": r.next_close,
                "next_direction": r.next_direction,
            }
            for r in rows
        ]
    )


def _missing_candle_ratio(db: Session, symbol: str, lookback_days: int = 365) -> float:
    instrument_like = f"%|{symbol}"
    min_max = db.execute(
        select(func.min(RawCandle.ts), func.max(RawCandle.ts)).where(
            and_(RawCandle.interval == "day", RawCandle.instrument_key.like(instrument_like))
        )
    ).first()
    if not min_max or min_max[0] is None or min_max[1] is None:
        return 1.0
    end = min_max[1].date()
    start = max(min_max[0].date(), end - timedelta(days=max(lookback_days, 30)))
    present_dates = set(
        d
        for d in db.scalars(
            select(func.date(RawCandle.ts)).where(
                and_(RawCandle.interval == "day", RawCandle.instrument_key.like(instrument_like))
            )
        ).all()
        if d is not None
    )
    expected = 0
    have = 0
    probe = start
    while probe <= end:
        if is_trading_day(probe):
            expected += 1
            if str(probe) in present_dates or probe in present_dates:
                have += 1
        probe += timedelta(days=1)
    if expected == 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (have / expected)))


def _stale_sources(db: Session, stale_hours: int) -> list[str]:
    cutoff = datetime.now(IST_ZONE) - timedelta(hours=stale_hours)
    rows = db.execute(select(DataFreshness)).scalars().all()
    stale: list[str] = []
    for row in rows:
        if row.last_success_at is None:
            stale.append(row.source_name)
            continue
        last = row.last_success_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=IST_ZONE)
        if last < cutoff:
            stale.append(row.source_name)
    return sorted(set(stale))


def build_point_in_time_dataset(db: Session, symbol: str) -> tuple[pd.DataFrame, DataQualityReport]:
    settings = get_settings()
    f_df = _load_feature_df(db, symbol)
    l_df = _load_label_df(db, symbol)
    if f_df.empty or l_df.empty:
        report = DataQualityReport(
            symbol=symbol,
            rows=0,
            duplicates_features=0,
            duplicates_labels=0,
            missing_candle_ratio=1.0,
            stale_sources=[],
            passed=False,
            details={"reason": "missing_features_or_labels"},
        )
        return pd.DataFrame(), report

    f_dup = int(f_df.duplicated(subset=["session_date"]).sum())
    l_dup = int(l_df.duplicated(subset=["session_date"]).sum())
    f_df = f_df.sort_values("session_date").drop_duplicates(subset=["session_date"], keep="last")
    l_df = l_df.sort_values("session_date").drop_duplicates(subset=["session_date"], keep="last")
    merged = f_df.merge(l_df, on="session_date", how="inner").sort_values("session_date")
    merged = merged.dropna()
    if settings.train_window_days > 0 and len(merged) > settings.train_window_days:
        merged = merged.tail(settings.train_window_days).copy()

    missing_ratio = _missing_candle_ratio(db, symbol, lookback_days=365)
    stale = _stale_sources(db, stale_hours=settings.point_in_time_stale_hours)
    passed = (
        len(merged) >= 80
        and f_dup == 0
        and l_dup == 0
        and missing_ratio <= settings.missing_candle_ratio_max
        and len(stale) == 0
    )

    report = DataQualityReport(
        symbol=symbol,
        rows=int(len(merged)),
        duplicates_features=f_dup,
        duplicates_labels=l_dup,
        missing_candle_ratio=float(missing_ratio),
        stale_sources=stale,
        passed=passed,
        details={
            "feature_schema_version": settings.feature_schema_version,
            "label_schema_version": settings.label_schema_version,
            "train_window_days": settings.train_window_days,
            "point_in_time_stale_hours": settings.point_in_time_stale_hours,
        },
    )
    return merged, report


def write_training_metadata(path, symbol: str, quality: DataQualityReport, window_from, window_to) -> None:
    import json

    payload = {
        "symbol": symbol,
        "quality_report": {
            "rows": quality.rows,
            "duplicates_features": quality.duplicates_features,
            "duplicates_labels": quality.duplicates_labels,
            "missing_candle_ratio": quality.missing_candle_ratio,
            "stale_sources": quality.stale_sources,
            "passed": quality.passed,
            "details": quality.details,
        },
        "window_from": str(window_from),
        "window_to": str(window_to),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
