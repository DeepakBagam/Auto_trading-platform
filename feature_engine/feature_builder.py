from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from db.models import FeaturesDaily, RawCandle, RawNews
from feature_engine.macro_features import build_macro_features
from feature_engine.microstructure_features import build_microstructure_features
from feature_engine.price_features import build_price_features
from feature_engine.sentiment_features import aggregate_daily_sentiment, explode_news_symbols
from feature_engine.volume_features import build_volume_features
from execution_engine.risk_manager import vix_position_multiplier
from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)


def _symbol_from_instrument(instrument_key: str) -> str:
    if "|" in instrument_key:
        return instrument_key.split("|", 1)[1]
    return instrument_key


def _load_daily_candles(db: Session, instrument_key: str) -> pd.DataFrame:
    rows = db.execute(
        select(RawCandle).where(
            and_(RawCandle.instrument_key == instrument_key, RawCandle.interval == "day")
        )
    ).scalars().all()
    data = [
        {
            "ts": r.ts,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


def _load_news_df(db: Session) -> pd.DataFrame:
    rows = db.execute(select(RawNews)).scalars().all()
    data = [
        {
            "source": r.source,
            "url": r.url,
            "published_at": r.published_at,
            "symbols": r.symbols,
            "sentiment_score": r.sentiment_score,
            "relevance_score": r.relevance_score,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


def _load_daily_market_candles(db: Session) -> pd.DataFrame:
    rows = db.execute(select(RawCandle).where(RawCandle.interval == "day")).scalars().all()
    data = [
        {
            "symbol": _symbol_from_instrument(r.instrument_key),
            "ts": r.ts,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


def build_daily_features_for_symbol(
    db: Session,
    instrument_key: str,
    news_df: pd.DataFrame | None = None,
    market_df: pd.DataFrame | None = None,
) -> int:
    symbol = _symbol_from_instrument(instrument_key)
    candle_df = _load_daily_candles(db, instrument_key)
    if candle_df.empty:
        return 0
    price_df = build_price_features(candle_df)
    volume_df = build_volume_features(price_df)
    
    # Add microstructure features
    try:
        micro_df = build_microstructure_features(volume_df)
    except Exception as exc:
        logger.warning("Microstructure features failed for %s: %s", symbol, exc)
        micro_df = volume_df
    
    feat_df = build_macro_features(micro_df, symbol=symbol, market_df=market_df)
    feat_df["india_vix_level"] = feat_df.get("india_vix_close_level", 0.0).fillna(0.0)
    feat_df["vix_position_multiplier"] = feat_df["india_vix_level"].apply(vix_position_multiplier)
    feat_df["session_date"] = pd.to_datetime(feat_df["ts"]).dt.date
    # Some providers can emit more than one daily row for the same session_date.
    # Keep the latest timestamp per day so DB uniqueness (symbol, session_date) holds.
    feat_df = feat_df.sort_values("ts").drop_duplicates(subset=["session_date"], keep="last")

    news_df = news_df.copy() if news_df is not None else _load_news_df(db)
    if not news_df.empty:
        expanded = explode_news_symbols(news_df.to_dict(orient="records"))
        daily_sent = aggregate_daily_sentiment(expanded)
        symbol_sent = daily_sent[daily_sent["symbol"].isin([symbol, "MARKET"])]
        if not symbol_sent.empty:
            symbol_sent = symbol_sent.groupby("session_date", as_index=False).agg(
                {
                    "sent_intraday": "mean",
                    "sent_overnight": "mean",
                    "news_count": "sum",
                }
            )
            feat_df = feat_df.merge(symbol_sent, on="session_date", how="left")
    for col in ["sent_intraday", "sent_overnight", "news_count"]:
        if col not in feat_df:
            feat_df[col] = 0.0
        feat_df[col] = feat_df[col].fillna(0.0)

    ignore_cols = {"ts", "session_date"}
    inserted = 0
    updated = 0
    for _, row in feat_df.iterrows():
        session_date = row["session_date"]
        features = {
            k: float(v)
            for k, v in row.items()
            if k not in ignore_cols and pd.notna(v) and isinstance(v, (int, float))
        }
        existing = db.scalar(
            select(FeaturesDaily).where(
                FeaturesDaily.symbol == symbol, FeaturesDaily.session_date == session_date
            )
        )
        if existing:
            existing.features = features
            updated += 1
        else:
            db.add(FeaturesDaily(symbol=symbol, session_date=session_date, features=features))
            inserted += 1
    db.commit()
    logger.info("Built feature rows for %s inserted=%s updated=%s", symbol, inserted, updated)
    return inserted


def build_daily_features(db: Session, instrument_keys: list[str]) -> dict:
    news_df = _load_news_df(db)
    market_df = _load_daily_market_candles(db)
    summary = {}
    for key in instrument_keys:
        try:
            summary[key] = build_daily_features_for_symbol(
                db,
                key,
                news_df=news_df,
                market_df=market_df,
            )
        except Exception as exc:
            db.rollback()
            logger.exception("Feature build failed for %s: %s", key, exc)
            summary[key] = 0
    return summary


def latest_feature_cutoff_ist() -> datetime:
    return datetime.now(IST_ZONE)
