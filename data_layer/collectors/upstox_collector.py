from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable, List

import requests
from dateutil import parser
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import DataFreshness, RawCandle
from data_layer.processors.candle_resampler import resample_candles
from utils.config import get_settings
from utils.constants import IST_ZONE, SUPPORTED_INTERVALS
from utils.logger import get_logger
from utils.types import CandleRecord

logger = get_logger(__name__)


class UpstoxCollector:
    LIVE_INTRADAY_INTERVALS = {"1minute"}
    HISTORICAL_BATCH_INTERVALS = ("1minute",)
    FULL_BACKFILL_PLAN = {
        "1minute": {"chunk_days": 5},
        "30minute": {"chunk_days": 45},
        "day": {"chunk_days": 365},
    }

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.upstox_base_url.rstrip("/")
        self.session = requests.Session()
        access_token = getattr(self.settings, "market_data_access_token", "") or getattr(
            self.settings, "upstox_access_token", ""
        )
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def fetch_historical_candles(
        self, instrument_key: str, interval: str, from_date: date, to_date: date
    ) -> List[CandleRecord]:
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval={interval}")
        version_pref = str(self.settings.upstox_history_api_version or "auto").strip().lower()
        methods = []
        if version_pref in {"v3", "auto"}:
            methods.append(self._fetch_historical_candles_v3)
        if version_pref in {"v2", "auto"}:
            methods.append(self._fetch_historical_candles_v2)
        last_error: Exception | None = None
        for method in methods:
            try:
                return method(instrument_key, interval, from_date, to_date)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Historical candle fetch failed via %s for %s %s %s..%s: %s",
                    method.__name__,
                    instrument_key,
                    interval,
                    from_date.isoformat(),
                    to_date.isoformat(),
                    exc,
                )
        if last_error is not None:
            raise last_error
        return []

    def fetch_intraday_candles(self, instrument_key: str, interval: str) -> List[CandleRecord]:
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval={interval}")
        version_pref = str(self.settings.upstox_history_api_version or "auto").strip().lower()
        methods = []
        if version_pref in {"v3", "auto"}:
            methods.append(self._fetch_intraday_candles_v3)
        if version_pref in {"v2", "auto"}:
            methods.append(self._fetch_intraday_candles_v2)
        last_error: Exception | None = None
        for method in methods:
            try:
                return method(instrument_key, interval)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Intraday candle fetch failed via %s for %s %s: %s",
                    method.__name__,
                    instrument_key,
                    interval,
                    exc,
                )
        if last_error is not None:
            raise last_error
        return []

    def _fetch_json(self, endpoint: str) -> dict:
        response = self.session.get(endpoint, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def _history_interval_params(self, interval: str) -> tuple[str, str]:
        mapping = {
            "1minute": ("minutes", "1"),
            "30minute": ("minutes", "30"),
            "day": ("days", "1"),
        }
        try:
            return mapping[interval]
        except KeyError as exc:
            raise ValueError(f"Unsupported interval={interval}") from exc

    def _fetch_historical_candles_v3(
        self,
        instrument_key: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> List[CandleRecord]:
        unit, interval_value = self._history_interval_params(interval)
        endpoint = (
            f"{self.base_url}/v3/historical-candle/{instrument_key}/{unit}/{interval_value}"
            f"/{to_date.isoformat()}/{from_date.isoformat()}"
        )
        payload = self._fetch_json(endpoint)
        candles = payload.get("data", {}).get("candles", [])
        return self._normalize_response(candles, instrument_key, interval)

    def _fetch_historical_candles_v2(
        self,
        instrument_key: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> List[CandleRecord]:
        endpoint = (
            f"{self.base_url}/v2/historical-candle/{instrument_key}/{interval}"
            f"/{to_date.isoformat()}/{from_date.isoformat()}"
        )
        payload = self._fetch_json(endpoint)
        candles = payload.get("data", {}).get("candles", [])
        return self._normalize_response(candles, instrument_key, interval)

    def _fetch_intraday_candles_v3(self, instrument_key: str, interval: str) -> List[CandleRecord]:
        unit, interval_value = self._history_interval_params(interval)
        endpoint = f"{self.base_url}/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval_value}"
        payload = self._fetch_json(endpoint)
        candles = payload.get("data", {}).get("candles", [])
        return self._normalize_response(candles, instrument_key, interval)

    def _fetch_intraday_candles_v2(self, instrument_key: str, interval: str) -> List[CandleRecord]:
        endpoint = f"{self.base_url}/v2/historical-candle/intraday/{instrument_key}/{interval}"
        payload = self._fetch_json(endpoint)
        candles = payload.get("data", {}).get("candles", [])
        return self._normalize_response(candles, instrument_key, interval)

    def _ingest_interval_range_chunked(
        self,
        db: Session,
        instrument_key: str,
        interval: str,
        from_date: date,
        to_date: date,
        chunk_days: int,
    ) -> dict:
        if from_date > to_date:
            return {"inserted": 0, "records_seen": 0, "chunks": 0, "errors": 0}

        cur = from_date
        total_inserted = 0
        total_seen = 0
        chunks = 0
        errors = 0
        adaptive_chunk_days = max(1, chunk_days)

        while cur <= to_date:
            chunk_to = min(cur + timedelta(days=adaptive_chunk_days - 1), to_date)
            try:
                records = self.fetch_historical_candles(
                    instrument_key=instrument_key,
                    interval=interval,
                    from_date=cur,
                    to_date=chunk_to,
                )
                seen = len(records)
                inserted = self.persist(db, records)
                total_seen += seen
                total_inserted += inserted
                chunks += 1
                logger.info(
                    "Ingested %s/%s candles for %s %s chunk=%s..%s",
                    inserted,
                    seen,
                    instrument_key,
                    interval,
                    cur.isoformat(),
                    chunk_to.isoformat(),
                )
                cur = chunk_to + timedelta(days=1)
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else None
                if code == 400 and adaptive_chunk_days > 1:
                    adaptive_chunk_days = max(1, adaptive_chunk_days // 2)
                    logger.warning(
                        "Reducing chunk size for %s %s to %s day(s) after 400 error on %s..%s",
                        instrument_key,
                        interval,
                        adaptive_chunk_days,
                        cur.isoformat(),
                        chunk_to.isoformat(),
                    )
                    continue
                errors += 1
                logger.exception(
                    "Skipping failed chunk for %s %s %s..%s: %s",
                    instrument_key,
                    interval,
                    cur.isoformat(),
                    chunk_to.isoformat(),
                    exc,
                )
                cur = chunk_to + timedelta(days=1)
            except Exception as exc:
                errors += 1
                logger.exception(
                    "Skipping failed chunk for %s %s %s..%s: %s",
                    instrument_key,
                    interval,
                    cur.isoformat(),
                    chunk_to.isoformat(),
                    exc,
                )
                cur = chunk_to + timedelta(days=1)

        return {
            "inserted": total_inserted,
            "records_seen": total_seen,
            "chunks": chunks,
            "errors": errors,
        }

    def _normalize_response(
        self, candles: Iterable[list], instrument_key: str, interval: str
    ) -> List[CandleRecord]:
        output: list[CandleRecord] = []
        for row in candles:
            # Expected order: [timestamp, open, high, low, close, volume, oi]
            if len(row) < 6:
                continue
            ts = parser.isoparse(str(row[0])).astimezone(IST_ZONE)
            oi = float(row[6]) if len(row) > 6 and row[6] is not None else None
            output.append(
                self._normalize_record(
                    CandleRecord(
                        instrument_key=instrument_key,
                        interval=interval,
                        ts=ts,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        oi=oi,
                    )
                )
            )
        return output

    def _normalize_record(self, record: CandleRecord) -> CandleRecord:
        ts = record.ts.astimezone(IST_ZONE) if record.ts.tzinfo is not None else record.ts.replace(tzinfo=IST_ZONE)
        if record.interval == "day":
            ts = datetime.combine(ts.date(), datetime.min.time(), tzinfo=IST_ZONE)

        high = max(float(record.high), float(record.open), float(record.close), float(record.low))
        low = min(float(record.low), float(record.open), float(record.close), float(record.high))
        return CandleRecord(
            instrument_key=record.instrument_key,
            interval=record.interval,
            ts=ts,
            open=float(record.open),
            high=high,
            low=low,
            close=float(record.close),
            volume=float(record.volume),
            oi=record.oi,
            source=record.source,
        )

    def persist(self, db: Session, records: List[CandleRecord], update_existing: bool = False) -> int:
        if not records:
            return 0
        inserted = 0
        updated = 0
        for raw_record in records:
            r = self._normalize_record(raw_record)
            if r.interval == "day":
                session_start = datetime.combine(r.ts.date(), datetime.min.time(), tzinfo=IST_ZONE)
                session_end = session_start + timedelta(days=1)
                existing_day_row = db.scalar(
                    select(RawCandle).where(
                        RawCandle.instrument_key == r.instrument_key,
                        RawCandle.interval == "day",
                        RawCandle.ts >= session_start,
                        RawCandle.ts < session_end,
                    )
                )
                if existing_day_row is not None:
                    changed = (
                        existing_day_row.ts != session_start
                        or float(existing_day_row.open) != float(r.open)
                        or float(existing_day_row.high) != float(r.high)
                        or float(existing_day_row.low) != float(r.low)
                        or float(existing_day_row.close) != float(r.close)
                        or float(existing_day_row.volume) != float(r.volume)
                        or existing_day_row.oi != r.oi
                        or existing_day_row.source != r.source
                    )
                    if changed:
                        existing_day_row.ts = session_start
                        existing_day_row.open = float(r.open)
                        existing_day_row.high = float(r.high)
                        existing_day_row.low = float(r.low)
                        existing_day_row.close = float(r.close)
                        existing_day_row.volume = float(r.volume)
                        existing_day_row.oi = r.oi
                        existing_day_row.source = r.source
                        updated += 1
                    continue
            exists = db.scalar(
                select(RawCandle).where(
                    RawCandle.instrument_key == r.instrument_key,
                    RawCandle.interval == r.interval,
                    RawCandle.ts == r.ts,
                )
            )
            if exists is not None:
                if not update_existing:
                    continue
                changed = (
                    float(exists.open) != float(r.open)
                    or float(exists.high) != float(r.high)
                    or float(exists.low) != float(r.low)
                    or float(exists.close) != float(r.close)
                    or float(exists.volume) != float(r.volume)
                    or exists.oi != r.oi
                    or exists.source != r.source
                )
                if changed:
                    exists.open = float(r.open)
                    exists.high = float(r.high)
                    exists.low = float(r.low)
                    exists.close = float(r.close)
                    exists.volume = float(r.volume)
                    exists.oi = r.oi
                    exists.source = r.source
                    updated += 1
                continue
            db.add(
                RawCandle(
                    instrument_key=r.instrument_key,
                    interval=r.interval,
                    ts=r.ts,
                    open=r.open,
                    high=r.high,
                    low=r.low,
                    close=r.close,
                    volume=r.volume,
                    oi=r.oi,
                    source=r.source,
                )
            )
            inserted += 1
        if inserted or updated:
            self._mark_freshness(
                db,
                "upstox_candles",
                "ok",
                {"inserted": inserted, "updated": updated, "records_seen": len(records)},
            )
            db.commit()
        return inserted

    def derive_daily_candle(self, db: Session, instrument_key: str, target_date: date) -> int:
        existing = db.scalar(
            select(RawCandle.id).where(
                RawCandle.instrument_key == instrument_key,
                RawCandle.interval == "day",
                RawCandle.ts == datetime.combine(target_date, datetime.min.time(), tzinfo=IST_ZONE),
            )
        )
        if existing:
            return 0

        intraday_rows = (
            db.execute(
                select(RawCandle)
                .where(
                    RawCandle.instrument_key == instrument_key,
                    RawCandle.interval == "1minute",
                )
                .order_by(RawCandle.ts.asc())
            )
            .scalars()
            .all()
        )
        rows_for_date = [r for r in intraday_rows if r.ts.date() == target_date]
        if not rows_for_date:
            return 0

        import pandas as pd

        intraday_df = pd.DataFrame(
            [
                {
                    "ts": r.ts,
                    "open": float(r.open),
                    "high": float(r.high),
                    "low": float(r.low),
                    "close": float(r.close),
                    "volume": float(r.volume),
                }
                for r in rows_for_date
            ]
        )
        day_df = resample_candles(intraday_df, "day")
        if day_df.empty:
            return 0
        row = day_df.iloc[-1]
        db.add(
            RawCandle(
                instrument_key=instrument_key,
                interval="day",
                ts=datetime.combine(target_date, datetime.min.time(), tzinfo=IST_ZONE),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                oi=None,
                source="derived_intraday",
            )
        )
        db.commit()
        return 1

    def ingest_historical_batch(self, db: Session, days_back: int = 180) -> dict:
        now = datetime.now(IST_ZONE).date()
        from_date = now - timedelta(days=days_back)
        summary: dict[str, int] = {}
        for instrument_key in self.settings.instrument_keys:
            for interval in self.HISTORICAL_BATCH_INTERVALS:
                try:
                    historical_records = self.fetch_historical_candles(
                        instrument_key=instrument_key,
                        interval=interval,
                        from_date=from_date,
                        to_date=now,
                    )
                    historical_inserted = self.persist(db, historical_records)
                    live_records: list[CandleRecord] = []
                    live_inserted = 0
                    if interval in self.LIVE_INTRADAY_INTERVALS:
                        # Upstox's historical endpoint can lag during the active session.
                        # Merge the dedicated intraday feed so live bars are persisted.
                        live_records = self.fetch_intraday_candles(
                            instrument_key=instrument_key,
                            interval=interval,
                        )
                        live_inserted = self.persist(db, live_records, update_existing=True)
                    count = historical_inserted + live_inserted
                    summary[f"{instrument_key}:{interval}"] = count
                    logger.info(
                        (
                            "Ingested %s candles for %s %s "
                            "(historical_inserted=%s historical_seen=%s live_inserted=%s live_seen=%s)"
                        ),
                        count,
                        instrument_key,
                        interval,
                        historical_inserted,
                        len(historical_records),
                        live_inserted,
                        len(live_records),
                    )
                except Exception as exc:
                    logger.exception(
                        "Upstox ingestion failed for %s %s: %s", instrument_key, interval, exc
                    )
                    self._mark_freshness(
                        db,
                        "upstox_candles",
                        "error",
                        {"instrument_key": instrument_key, "interval": interval, "error": str(exc)},
                    )
                    db.commit()
        return summary

    def ingest_historical_full(self, db: Session, one_minute_days: int = 730) -> dict:
        now = datetime.now(IST_ZONE).date()
        one_min_start = now - timedelta(days=one_minute_days)
        plan = {
            "1minute": {
                "from": one_min_start,
                "to": now,
                "chunk_days": self.FULL_BACKFILL_PLAN["1minute"]["chunk_days"],
            },
            "30minute": {
                "from": one_min_start,
                "to": now,
                "chunk_days": self.FULL_BACKFILL_PLAN["30minute"]["chunk_days"],
            },
            "day": {
                "from": one_min_start,
                "to": now,
                "chunk_days": self.FULL_BACKFILL_PLAN["day"]["chunk_days"],
            },
        }
        summary: dict[str, dict] = {}
        for instrument_key in self.settings.instrument_keys:
            for interval in SUPPORTED_INTERVALS:
                cfg = plan[interval]
                try:
                    out = self._ingest_interval_range_chunked(
                        db=db,
                        instrument_key=instrument_key,
                        interval=interval,
                        from_date=cfg["from"],
                        to_date=cfg["to"],
                        chunk_days=cfg["chunk_days"],
                    )
                    summary[f"{instrument_key}:{interval}"] = out
                    self._mark_freshness(
                        db,
                        "upstox_candles",
                        "ok",
                        {
                            "instrument_key": instrument_key,
                            "interval": interval,
                            "from": cfg["from"].isoformat(),
                            "to": cfg["to"].isoformat(),
                            **out,
                        },
                    )
                    db.commit()
                except Exception as exc:
                    logger.exception(
                        "Full backfill failed for %s %s: %s", instrument_key, interval, exc
                    )
                    self._mark_freshness(
                        db,
                        "upstox_candles",
                        "error",
                        {"instrument_key": instrument_key, "interval": interval, "error": str(exc)},
                    )
                    db.commit()
        return summary

    def _mark_freshness(self, db: Session, source_name: str, status: str, details: dict) -> None:
        row = db.scalar(select(DataFreshness).where(DataFreshness.source_name == source_name))
        if row is None:
            row = DataFreshness(source_name=source_name, last_success_at=datetime.now(IST_ZONE))
            db.add(row)
        row.last_success_at = datetime.now(IST_ZONE)
        row.status = status
        row.details = details
