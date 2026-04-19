from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Iterable, List

import numpy as np
import pandas as pd

from utils.constants import IST_ZONE

_BULLISH = {"surge", "beat", "rally", "growth", "strong", "upgrade", "profit"}
_BEARISH = {"fall", "drop", "miss", "downgrade", "weak", "loss", "crash"}


def extract_symbols(text: str) -> List[str]:
    if not text:
        return []
    upper_tokens = re.findall(r"\b[A-Z]{2,12}\b", text.upper())
    blocked = {"THE", "AND", "FOR", "WITH", "NSE", "BSE", "RBI", "SEBI"}
    symbols = sorted({t for t in upper_tokens if t not in blocked})
    return symbols[:10]


class FinBertSentimentAnalyzer:
    def __init__(self) -> None:
        self.pipeline = None
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            self.pipeline = self._load_pipeline(
                pipeline=pipeline,
                AutoModelForSequenceClassification=AutoModelForSequenceClassification,
                AutoTokenizer=AutoTokenizer,
            )
        except Exception:
            # NLP extras are optional. We fall back to heuristic sentiment.
            self.pipeline = None

    @staticmethod
    def _load_pipeline(pipeline, AutoModelForSequenceClassification, AutoTokenizer):
        model_name = "ProsusAI/finbert"
        last_exc: Exception | None = None
        for local_only in (True, False):
            try:
                prev_hf_offline = os.environ.get("HF_HUB_OFFLINE")
                prev_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
                prev_disable_conversion = os.environ.get("DISABLE_SAFETENSORS_CONVERSION")
                os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
                if local_only:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                else:
                    if prev_hf_offline is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    if prev_tf_offline is None:
                        os.environ.pop("TRANSFORMERS_OFFLINE", None)
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    local_files_only=local_only,
                    use_safetensors=False,
                )
                return pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    truncation=True,
                    max_length=512,
                    framework="pt",
                )
            except Exception as exc:
                last_exc = exc
            finally:
                if prev_hf_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev_hf_offline
                if prev_tf_offline is None:
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                else:
                    os.environ["TRANSFORMERS_OFFLINE"] = prev_tf_offline
                if prev_disable_conversion is None:
                    os.environ.pop("DISABLE_SAFETENSORS_CONVERSION", None)
                else:
                    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = prev_disable_conversion
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unable to initialize FinBERT pipeline.")

    def score_text(self, text: str) -> float:
        text = (text or "").strip()
        if not text:
            return 0.0
        if self.pipeline:
            result = self.pipeline(text[:1000])[0]
            label = str(result.get("label", "")).lower()
            score = float(result.get("score", 0.0))
            if "positive" in label:
                return score
            if "negative" in label:
                return -score
            return 0.0
        return _heuristic_score(text)


def _heuristic_score(text: str) -> float:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if not tokens:
        return 0.0
    bull = sum(1 for t in tokens if t in _BULLISH)
    bear = sum(1 for t in tokens if t in _BEARISH)
    raw = (bull - bear) / max(1, bull + bear)
    return float(np.clip(raw, -1.0, 1.0))


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(columns=["symbol", "session_date", "sent_intraday", "sent_overnight", "news_count"])
    df = news_df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True).dt.tz_convert(IST_ZONE)
    df["session_date"] = df["published_at"].dt.date
    now = datetime.now(IST_ZONE)
    age_hours = (now - df["published_at"]).dt.total_seconds() / 3600.0
    df["weight"] = np.exp(-age_hours / 12.0)
    df["weighted_score"] = df["sentiment_score"] * df["weight"] * (1 + df["relevance_score"])
    intraday = (
        df.groupby(["symbol", "session_date"], as_index=False)
        .agg(
            sent_intraday=("weighted_score", "sum"),
            news_count=("url", "count"),
        )
        .fillna(0.0)
    )
    overnight = intraday.copy()
    overnight["session_date"] = overnight["session_date"].apply(lambda x: x)
    overnight = overnight.rename(columns={"sent_intraday": "sent_overnight"})
    merged = intraday.merge(
        overnight[["symbol", "session_date", "sent_overnight"]],
        on=["symbol", "session_date"],
        how="left",
    )
    merged["sent_overnight"] = merged["sent_overnight"].fillna(0.0)
    return merged


def explode_news_symbols(news_rows: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for item in news_rows:
        symbols = item.get("symbols") or ["MARKET"]
        for sym in symbols:
            rows.append(
                {
                    "symbol": sym,
                    "published_at": item.get("published_at"),
                    "url": item.get("url"),
                    "sentiment_score": item.get("sentiment_score", 0.0),
                    "relevance_score": item.get("relevance_score", 0.0),
                }
            )
    return pd.DataFrame(rows)
