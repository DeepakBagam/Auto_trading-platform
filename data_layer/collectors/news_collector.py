from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, List

import feedparser
import requests
from dateutil import parser
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import DataFreshness, RawNews
from feature_engine.sentiment_features import FinBertSentimentAnalyzer, extract_symbols
from utils.config import get_settings
from utils.constants import IST_ZONE, NEWS_RSS_FEEDS
from utils.logger import get_logger
from utils.types import NewsRecord

logger = get_logger(__name__)


class NewsCollector:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.sentiment = FinBertSentimentAnalyzer()

    def fetch_rss_news(self) -> List[Dict[str, Any]]:
        articles: list[dict[str, Any]] = []
        for feed_url in NEWS_RSS_FEEDS:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                articles.append(
                    {
                        "source_channel": "rss",
                        "source": feed.feed.get("title", "rss"),
                        "title": entry.get("title", ""),
                        "url": entry.get("link", ""),
                        "published_at": entry.get("published", entry.get("updated", "")),
                        "content": self._extract_content(entry),
                        "raw_payload": dict(entry),
                    }
                )
        return articles

    def fetch_newsapi_news(self, query: str = "nse OR nifty OR sensex OR india stock market") -> List[dict]:
        if not self.settings.newsapi_key:
            return []
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.settings.newsapi_key,
            "pageSize": 100,
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        out = []
        for a in data.get("articles", []):
            out.append(
                {
                    "source_channel": "newsapi",
                    "source": a.get("source", {}).get("name", "newsapi"),
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                    "content": a.get("content", "") or a.get("description", ""),
                    "raw_payload": a,
                }
            )
        return out

    def fetch_finnhub_news(self, category: str = "general") -> List[dict]:
        if not self.settings.enable_finnhub or not self.settings.finnhub_api_key:
            return []
        url = "https://finnhub.io/api/v1/news"
        params = {"category": category, "token": self.settings.finnhub_api_key}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        out = []
        for a in response.json():
            out.append(
                {
                    "source_channel": "finnhub",
                    "source": a.get("source", "finnhub"),
                    "title": a.get("headline", ""),
                    "url": a.get("url", ""),
                    "published_at": datetime.fromtimestamp(
                        int(a.get("datetime", 0)), tz=IST_ZONE
                    ).isoformat(),
                    "content": a.get("summary", ""),
                    "raw_payload": a,
                }
            )
        return out

    def collect(self) -> tuple[List[NewsRecord], dict[str, int]]:
        rss_articles = self.fetch_rss_news()
        newsapi_articles = self.fetch_newsapi_news()
        finnhub_articles = self.fetch_finnhub_news()
        articles = rss_articles + newsapi_articles + finnhub_articles
        source_seen = {"rss": 0, "newsapi": 0, "finnhub": 0}
        records: list[NewsRecord] = []
        for a in articles:
            try:
                channel = str(a.get("source_channel", "")).lower()
                if channel in source_seen:
                    source_seen[channel] += 1
                published_at = self._parse_datetime(a.get("published_at"))
                title = (a.get("title") or "").strip()
                url = (a.get("url") or "").strip()
                if not title or not url:
                    continue
                content = (a.get("content") or "").strip()
                text = f"{title}. {content}".strip()
                sentiment_score = self.sentiment.score_text(text)
                relevance = self._calc_relevance(text)
                records.append(
                    NewsRecord(
                        source=a.get("source", "unknown"),
                        title=title[:2048],
                        url=url[:4096],
                        published_at=published_at,
                        content=content,
                        symbols=extract_symbols(text),
                        sentiment_score=float(sentiment_score),
                        relevance_score=float(relevance),
                        raw_payload={
                            **(a.get("raw_payload", {}) or {}),
                            "source_channel": a.get("source_channel", "unknown"),
                        },
                    )
                )
            except Exception:
                logger.exception("Failed to normalize article: %s", a)
        logger.info(
            "Collected raw news counts rss=%s newsapi=%s finnhub=%s total=%s",
            source_seen["rss"],
            source_seen["newsapi"],
            source_seen["finnhub"],
            len(articles),
        )
        return records, source_seen

    def persist(self, db: Session, records: List[NewsRecord], source_seen: dict[str, int]) -> int:
        inserted = 0
        source_inserted = {"rss": 0, "newsapi": 0, "finnhub": 0}
        for r in records:
            exists = db.scalar(
                select(RawNews.id).where(RawNews.source == r.source, RawNews.url == r.url)
            )
            if exists:
                continue
            db.add(
                RawNews(
                    source=r.source,
                    title=r.title,
                    url=r.url,
                    published_at=r.published_at,
                    content=r.content,
                    symbols=r.symbols,
                    sentiment_score=r.sentiment_score,
                    relevance_score=r.relevance_score,
                    raw_payload=r.raw_payload,
                )
            )
            inserted += 1
            channel = str((r.raw_payload or {}).get("source_channel", "")).lower()
            if channel in source_inserted:
                source_inserted[channel] += 1
        self._mark_freshness(
            db,
            "news_collector",
            "ok",
            {
                "inserted": inserted,
                "records_seen": len(records),
                "seen_by_source": source_seen,
                "inserted_by_source": source_inserted,
            },
        )
        db.commit()
        return inserted

    def run_once(self, db: Session) -> int:
        try:
            records, source_seen = self.collect()
            return self.persist(db, records, source_seen)
        except Exception as exc:
            logger.exception("News collection failed: %s", exc)
            self._mark_freshness(db, "news_collector", "error", {"error": str(exc)})
            db.commit()
            return 0

    @staticmethod
    def _extract_content(entry: Any) -> str:
        if "summary" in entry:
            return entry.get("summary", "")
        if "description" in entry:
            return entry.get("description", "")
        return ""

    @staticmethod
    def _parse_datetime(value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            return value.astimezone(IST_ZONE)
        if not value:
            return datetime.now(IST_ZONE)
        parsed = parser.parse(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=IST_ZONE)
        return parsed.astimezone(IST_ZONE)

    @staticmethod
    def _calc_relevance(text: str) -> float:
        keys = ("nifty", "sensex", "nse", "bse", "rbi", "sebi", "bank nifty", "fii", "dii")
        lower = text.lower()
        hits = sum(1 for k in keys if k in lower)
        return min(1.0, hits / 5.0)

    def _mark_freshness(self, db: Session, source_name: str, status: str, details: dict) -> None:
        row = db.scalar(select(DataFreshness).where(DataFreshness.source_name == source_name))
        if row is None:
            row = DataFreshness(source_name=source_name, last_success_at=datetime.now(IST_ZONE))
            db.add(row)
        row.last_success_at = datetime.now(IST_ZONE)
        row.status = status
        row.details = details


def stable_news_hash(source: str, title: str, published_at: datetime) -> str:
    seed = f"{source}|{title}|{published_at.isoformat()}".encode("utf-8", errors="ignore")
    return hashlib.sha256(seed).hexdigest()
