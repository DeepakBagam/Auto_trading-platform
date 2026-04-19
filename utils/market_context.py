from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import and_, desc, select
from sqlalchemy.orm import Session

from db.models import RawCandle, RawNews
from utils.constants import IST_ZONE
from utils.symbols import instrument_key_filter


def latest_price_for_instrument(db: Session, symbol: str, interval: str = "1minute") -> float | None:
    price = db.scalar(
        select(RawCandle.close)
        .where(
            and_(
                instrument_key_filter(RawCandle.instrument_key, symbol),
                RawCandle.interval == interval,
            )
        )
        .order_by(desc(RawCandle.ts))
        .limit(1)
    )
    return float(price) if price is not None else None


def latest_vix_level(db: Session, interval: str = "1minute") -> float | None:
    level = latest_price_for_instrument(db, "India VIX", interval=interval)
    if level is not None:
        return level
    return latest_price_for_instrument(db, "India VIX", interval="day")


def recent_news_sentiment_for_symbol(
    db: Session,
    *,
    symbol: str,
    now: datetime | None = None,
    lookback_hours: int = 2,
) -> float:
    now = now or datetime.now(IST_ZONE)
    cutoff = now - timedelta(hours=max(1, int(lookback_hours)))
    rows = (
        db.execute(
            select(RawNews)
            .where(RawNews.published_at >= cutoff)
            .order_by(RawNews.published_at.desc())
        )
        .scalars()
        .all()
    )
    if not rows:
        return 0.0

    weighted_total = 0.0
    weight_sum = 0.0
    target = str(symbol or "").upper()
    for row in rows:
        symbols = [str(item).upper() for item in (row.symbols or [])]
        if symbols and target not in symbols and "MARKET" not in symbols:
            continue
        age_hours = max(0.0, (now - (row.published_at if row.published_at.tzinfo else row.published_at.replace(tzinfo=IST_ZONE))).total_seconds() / 3600.0)
        recency_weight = max(0.15, 1.0 - (age_hours / max(1.0, float(lookback_hours))))
        relevance = 1.0 + float(row.relevance_score or 0.0)
        weight = recency_weight * relevance
        weighted_total += float(row.sentiment_score or 0.0) * weight
        weight_sum += weight
    if weight_sum <= 0:
        return 0.0
    score = weighted_total / weight_sum
    return max(-1.0, min(1.0, float(score)))
