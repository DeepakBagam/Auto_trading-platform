try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from sqlalchemy import select

from data_layer.collectors.news_collector import stable_news_hash
from db.connection import SessionLocal
from db.models import RawNews
from feature_engine.sentiment_features import FinBertSentimentAnalyzer
from utils.logger import setup_logging


def main() -> None:
    setup_logging()
    analyzer = FinBertSentimentAnalyzer()
    if analyzer.pipeline is None:
        raise RuntimeError("FinBERT is not available. Install NLP dependencies and ensure model weights are cached.")

    db = SessionLocal()
    try:
        rows = db.execute(select(RawNews).order_by(RawNews.published_at.asc(), RawNews.id.asc())).scalars().all()
        updated = 0
        for row in rows:
            title = (row.title or "").strip()
            content = (row.content or "").strip()
            text = f"{title}. {content}".strip()
            new_score = float(analyzer.score_text(text))
            if abs(float(row.sentiment_score or 0.0) - new_score) < 1e-9:
                continue
            row.sentiment_score = new_score
            payload = dict(row.raw_payload or {})
            payload["sentiment_model"] = "ProsusAI/finbert"
            payload["sentiment_hash"] = stable_news_hash(row.source, row.title, row.published_at)
            row.raw_payload = payload
            updated += 1
        db.commit()
        print(f"rescored_news={updated}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
