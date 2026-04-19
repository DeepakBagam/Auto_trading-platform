from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass(slots=True)
class CandleRecord:
    instrument_key: str
    interval: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    oi: float | None = None
    source: str = "upstox"


@dataclass(slots=True)
class NewsRecord:
    source: str
    title: str
    url: str
    published_at: datetime
    content: str
    symbols: List[str]
    sentiment_score: float
    relevance_score: float
    raw_payload: Dict[str, Any]


@dataclass(slots=True)
class SentimentAggregate:
    symbol: str
    session_date: str
    score_intraday: float
    score_overnight: float
    article_count: int


@dataclass(slots=True)
class FeatureRow:
    symbol: str
    session_date: str
    features: Dict[str, float]


@dataclass(slots=True)
class PredictionRecord:
    symbol: str
    target_session_date: str
    pred_open: float
    pred_high: float
    pred_low: float
    pred_close: float
    direction: str
    confidence: float
    model_version: str
