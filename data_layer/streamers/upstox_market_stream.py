from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from threading import Event, Lock
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from upstox_client import ApiClient, Configuration, MarketDataStreamerV3

from db.connection import SessionLocal
from db.models import DataFreshness, OrderBookSnapshot, RawCandle
from utils.config import Settings, get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger
from utils.types import CandleRecord

logger = get_logger(__name__)
SESSION_START = dt_time(9, 15)


@dataclass(slots=True)
class _MinuteBarState:
    instrument_key: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    tick_count: int = 0


class UpstoxMarketStream:
    def __init__(
        self,
        settings: Settings | None = None,
        session_factory: Callable[[], Session] | None = None,
        streamer: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session_factory = session_factory or SessionLocal
        self._lock = Lock()
        self._stop_event = Event()
        self._bars: dict[str, _MinuteBarState] = {}
        self._pending_candles: list[CandleRecord] = []
        self._pending_order_books: list[dict[str, Any]] = []
        self.streamer = streamer or self._build_streamer()
        self._bind_streamer_events()

    def _build_streamer(self) -> MarketDataStreamerV3:
        access_token = getattr(self.settings, "market_data_access_token", "") or getattr(
            self.settings, "upstox_access_token", ""
        )
        if not access_token:
            raise RuntimeError("UPSTOX_ACCESS_TOKEN is required for websocket streaming.")
        if not self.settings.instrument_keys:
            raise RuntimeError("UPSTOX_INSTRUMENT_KEYS is required for websocket streaming.")

        configuration = Configuration()
        configuration.access_token = access_token
        api_client = ApiClient(configuration)
        mode = self._resolve_stream_mode()
        return MarketDataStreamerV3(api_client, self.settings.instrument_keys, mode)

    def _resolve_stream_mode(self) -> str:
        requested = str(self.settings.upstox_websocket_mode or "ltpc").strip().lower()
        modes = MarketDataStreamerV3.Mode
        aliases = {
            "ltpc": modes["LTPC"],
            "full": modes["FULL"],
            "option_greeks": modes["OPTION"],
            "full_d30": modes["D30"],
        }
        if requested not in aliases:
            raise RuntimeError(f"Unsupported UPSTOX_WEBSOCKET_MODE={requested}")
        return aliases[requested]

    def _bind_streamer_events(self) -> None:
        self.streamer.on(self.streamer.Event["OPEN"], self._on_open)
        self.streamer.on(self.streamer.Event["MESSAGE"], self.handle_market_data)
        self.streamer.on(self.streamer.Event["ERROR"], self._on_error)
        self.streamer.on(self.streamer.Event["CLOSE"], self._on_close)
        self.streamer.on(self.streamer.Event["RECONNECTING"], self._on_reconnecting)
        self.streamer.on(
            self.streamer.Event["AUTO_RECONNECT_STOPPED"],
            self._on_auto_reconnect_stopped,
        )

    def start(self) -> None:
        self.streamer.auto_reconnect(
            True,
            interval=max(1, int(self.settings.upstox_websocket_reconnect_interval_seconds)),
            retry_count=max(1, int(self.settings.upstox_websocket_retry_count)),
        )
        self.streamer.connect()

    def stop(self) -> None:
        self._stop_event.set()
        self.flush_closed_candles(datetime.now(IST_ZONE), force=True)
        try:
            self.streamer.disconnect()
        except Exception:
            logger.exception("Failed to disconnect websocket streamer cleanly")

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop_event.is_set():
                self.flush_closed_candles(datetime.now(IST_ZONE))
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping websocket market stream on keyboard interrupt")
        finally:
            self.stop()

    def handle_market_data(self, message: dict[str, Any]) -> None:
        received_at = datetime.now(IST_ZONE)
        received_ns = time.time_ns()
        feeds = message.get("feeds") or {}
        current_ts = self._parse_epoch_ms(message.get("currentTs"))
        latest_exchange_ts = current_ts
        for instrument_key, payload in feeds.items():
            saw_structured_bar = False
            for record in self._extract_ohlc_records(instrument_key, payload):
                self._persist_candle_record(record)
                saw_structured_bar = True
                latest_exchange_ts = self._latest_timestamp(latest_exchange_ts, record.ts)
            snapshot = self._extract_order_book_snapshot(instrument_key, payload, current_ts)
            if snapshot is not None:
                self._persist_order_book_snapshot(snapshot)
                latest_exchange_ts = self._latest_timestamp(latest_exchange_ts, snapshot["ts"])

            ltpc = self._extract_ltpc(payload)
            if saw_structured_bar and ltpc is None:
                continue
            if not ltpc:
                continue
            try:
                price = float(ltpc["ltp"])
            except (KeyError, TypeError, ValueError):
                continue
            tick_ts = self._parse_epoch_ms(ltpc.get("ltt")) or current_ts
            if tick_ts is None:
                continue
            latest_exchange_ts = self._latest_timestamp(latest_exchange_ts, tick_ts)
            self._update_bar(instrument_key, tick_ts, price)
        if current_ts is not None:
            self.flush_closed_candles(current_ts)
        self._flush_pending_records(
            latest_exchange_ts=latest_exchange_ts,
            received_at=received_at,
            received_ns=received_ns,
        )

    def flush_closed_candles(self, reference_ts: datetime, force: bool = False) -> None:
        cutoff_minute = reference_ts.astimezone(IST_ZONE).replace(second=0, microsecond=0)
        to_persist: list[CandleRecord] = []
        with self._lock:
            for instrument_key, bar in list(self._bars.items()):
                if not force and bar.ts >= cutoff_minute:
                    continue
                to_persist.append(self._to_candle_record(bar))
                del self._bars[instrument_key]
        for record in to_persist:
            self._persist_candle_record(record)
        if to_persist:
            self._flush_pending_records()

    def _update_bar(self, instrument_key: str, tick_ts: datetime, price: float) -> None:
        minute_ts = tick_ts.astimezone(IST_ZONE).replace(second=0, microsecond=0)
        to_persist: CandleRecord | None = None
        live_record: CandleRecord | None = None
        with self._lock:
            current = self._bars.get(instrument_key)
            if current is None:
                current = _MinuteBarState(
                    instrument_key=instrument_key,
                    ts=minute_ts,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0.0,
                    tick_count=1,
                )
                self._bars[instrument_key] = current
                live_record = self._to_candle_record(current)
            if minute_ts < current.ts:
                return
            elif minute_ts > current.ts:
                to_persist = self._to_candle_record(current)
                current = _MinuteBarState(
                    instrument_key=instrument_key,
                    ts=minute_ts,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0.0,
                    tick_count=1,
                )
                self._bars[instrument_key] = current
                live_record = self._to_candle_record(current)
            else:
                current.high = max(current.high, price)
                current.low = min(current.low, price)
                current.close = price
                current.tick_count += 1
                live_record = self._to_candle_record(current)
        if to_persist is not None:
            self._persist_candle_record(to_persist)
        if live_record is not None:
            self._persist_candle_record(live_record)

    def _persist_candle_record(self, record: CandleRecord) -> None:
        self._pending_candles.append(record)

    def _persist_order_book_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._pending_order_books.append(snapshot)

    def _flush_pending_records(
        self,
        *,
        latest_exchange_ts: datetime | None = None,
        received_at: datetime | None = None,
        received_ns: int | None = None,
    ) -> None:
        if not self._pending_candles and not self._pending_order_books:
            return

        with self._lock:
            candle_buffer = self._pending_candles
            order_book_buffer = self._pending_order_books
            self._pending_candles = []
            self._pending_order_books = []

        deduped_candles: dict[tuple[str, str, datetime], CandleRecord] = {}
        for record in candle_buffer:
            key = (record.instrument_key, record.interval, record.ts)
            existing = deduped_candles.get(key)
            # Prefer records with actual volume (OHLC-sourced) over LTPC-assembled ones.
            # When both have volume, keep the higher value (more complete bar).
            if existing is None or record.volume > existing.volume:
                deduped_candles[key] = record
        deduped_order_books: dict[tuple[str, datetime], dict[str, Any]] = {}
        for snapshot in order_book_buffer:
            deduped_order_books[(str(snapshot["instrument_key"]), snapshot["ts"])] = snapshot

        db = self.session_factory()
        try:
            wrote = 0
            # Collect unique (instrument_key, ts) pairs needing derived candle sync.
            # Batch this once per flush rather than per-record to avoid N+1 queries.
            derive_keys: dict[str, datetime] = {}
            for record in deduped_candles.values():
                row_written = self._upsert_raw_candle(
                    db,
                    record,
                    update_existing=record.source in {"upstox_ws", "upstox_ws_ohlc", "derived_intraday"},
                )
                wrote += int(bool(row_written))
                if record.interval == "1minute":
                    existing_ts = derive_keys.get(record.instrument_key)
                    if existing_ts is None or record.ts > existing_ts:
                        derive_keys[record.instrument_key] = record.ts
            for instrument_key, latest_ts in derive_keys.items():
                self._sync_derived_candles(db, instrument_key, latest_ts)
            for snapshot in deduped_order_books.values():
                self._upsert_order_book_snapshot(db, snapshot)
            last_record = next(reversed(list(deduped_candles.values())), None)
            write_completed_at = datetime.now(IST_ZONE)
            write_completed_ns = time.time_ns()
            exchange_to_receive_latency_ns = None
            receive_to_persist_latency_ns = None
            exchange_to_persist_latency_ns = None
            if latest_exchange_ts is not None and received_at is not None:
                exchange_to_receive_latency_ns = max(
                    0,
                    int((received_at - latest_exchange_ts).total_seconds() * 1_000_000_000),
                )
            if received_ns is not None:
                receive_to_persist_latency_ns = max(0, write_completed_ns - received_ns)
            if latest_exchange_ts is not None:
                exchange_to_persist_latency_ns = max(
                    0,
                    int((write_completed_at - latest_exchange_ts).total_seconds() * 1_000_000_000),
                )
            self._mark_freshness(
                db,
                source_name="upstox_market_stream",
                status="ok",
                details={
                    "candles_buffered": len(candle_buffer),
                    "candles_flushed": len(deduped_candles),
                    "order_books_flushed": len(deduped_order_books),
                    "instrument_key": getattr(last_record, "instrument_key", None),
                    "interval": getattr(last_record, "interval", None),
                    "ts": last_record.ts.isoformat() if last_record is not None else None,
                    "close": getattr(last_record, "close", None),
                    "source": getattr(last_record, "source", None),
                    "latest_exchange_ts": latest_exchange_ts.isoformat() if latest_exchange_ts is not None else None,
                    "message_received_at": received_at.isoformat() if received_at is not None else None,
                    "write_completed_at": write_completed_at.isoformat(),
                    "exchange_timestamp_precision": "milliseconds",
                    "estimated_exchange_to_receive_latency_ns": exchange_to_receive_latency_ns,
                    "estimated_receive_to_persist_latency_ns": receive_to_persist_latency_ns,
                    "estimated_exchange_to_persist_latency_ns": exchange_to_persist_latency_ns,
                },
            )
            db.commit()
            if wrote or deduped_order_books:
                logger.info(
                    "Persisted websocket batch candles=%s order_books=%s",
                    len(deduped_candles),
                    len(deduped_order_books),
                )
        except IntegrityError:
            db.rollback()
        except Exception:
            db.rollback()
            logger.exception("Failed to persist websocket batch")
        finally:
            db.close()

    def _upsert_raw_candle(self, db: Session, record: CandleRecord, update_existing: bool) -> bool:
        existing = db.scalar(
            select(RawCandle).where(
                RawCandle.instrument_key == record.instrument_key,
                RawCandle.interval == record.interval,
                RawCandle.ts == record.ts,
            )
        )
        if existing is None:
            db.add(
                RawCandle(
                    instrument_key=record.instrument_key,
                    interval=record.interval,
                    ts=record.ts,
                    open=record.open,
                    high=record.high,
                    low=record.low,
                    close=record.close,
                    volume=record.volume,
                    oi=record.oi,
                    source=record.source,
                )
            )
            return True
        if not update_existing:
            return False
        existing.open = record.open
        existing.high = record.high
        existing.low = record.low
        existing.close = record.close
        existing.volume = record.volume
        existing.oi = record.oi
        existing.source = record.source
        return True

    def _sync_derived_candles(self, db: Session, instrument_key: str, minute_ts: datetime) -> None:
        for interval in ("30minute", "day"):
            record = self._derive_from_minutes(db, instrument_key, minute_ts, interval)
            if record is None:
                continue
            self._upsert_raw_candle(db, record, update_existing=True)

    def _upsert_order_book_snapshot(self, db: Session, snapshot: dict[str, Any]) -> None:
        existing = db.scalar(
            select(OrderBookSnapshot).where(
                OrderBookSnapshot.instrument_key == snapshot["instrument_key"],
                OrderBookSnapshot.ts == snapshot["ts"],
            )
        )
        if existing is None:
            db.add(
                OrderBookSnapshot(
                    instrument_key=str(snapshot["instrument_key"]),
                    ts=snapshot["ts"],
                    best_bid=float(snapshot["best_bid"]),
                    best_ask=float(snapshot["best_ask"]),
                    mid_price=float(snapshot["mid_price"]),
                    spread_bps=float(snapshot["spread_bps"]),
                    bid_volume=float(snapshot["bid_volume"]),
                    ask_volume=float(snapshot["ask_volume"]),
                    depth_imbalance=float(snapshot["depth_imbalance"]),
                    liquidity_score=float(snapshot["liquidity_score"]),
                    depth_data=snapshot["depth_data"],
                )
            )
            return
        existing.best_bid = float(snapshot["best_bid"])
        existing.best_ask = float(snapshot["best_ask"])
        existing.mid_price = float(snapshot["mid_price"])
        existing.spread_bps = float(snapshot["spread_bps"])
        existing.bid_volume = float(snapshot["bid_volume"])
        existing.ask_volume = float(snapshot["ask_volume"])
        existing.depth_imbalance = float(snapshot["depth_imbalance"])
        existing.liquidity_score = float(snapshot["liquidity_score"])
        existing.depth_data = snapshot["depth_data"]

    def _derive_from_minutes(
        self,
        db: Session,
        instrument_key: str,
        minute_ts: datetime,
        interval: str,
    ) -> CandleRecord | None:
        minute_ts = minute_ts.astimezone(IST_ZONE).replace(second=0, microsecond=0)
        if interval == "30minute":
            bucket_start = self._thirty_minute_bucket_start(minute_ts)
            range_start = bucket_start
            range_end = bucket_start + timedelta(minutes=30)
            candle_ts = bucket_start
        elif interval == "day":
            range_start = datetime.combine(minute_ts.date(), dt_time.min, tzinfo=IST_ZONE)
            range_end = range_start + timedelta(days=1)
            candle_ts = range_start
        else:
            raise ValueError(f"Unsupported derived interval={interval}")

        rows = (
            db.execute(
                select(RawCandle)
                .where(
                    RawCandle.instrument_key == instrument_key,
                    RawCandle.interval == "1minute",
                    RawCandle.ts >= range_start,
                    RawCandle.ts < range_end,
                )
                .order_by(RawCandle.ts.asc())
            )
            .scalars()
            .all()
        )
        if not rows:
            return None
        return CandleRecord(
            instrument_key=instrument_key,
            interval=interval,
            ts=candle_ts,
            open=float(rows[0].open),
            high=max(float(r.high) for r in rows),
            low=min(float(r.low) for r in rows),
            close=float(rows[-1].close),
            volume=float(sum(float(r.volume) for r in rows)),
            oi=rows[-1].oi,
            source="derived_intraday",
        )

    def _thirty_minute_bucket_start(self, ts: datetime) -> datetime:
        session_start = datetime.combine(ts.date(), SESSION_START, tzinfo=IST_ZONE)
        minutes_since_open = max(0, int((ts - session_start).total_seconds() // 60))
        bucket_minutes = (minutes_since_open // 30) * 30
        return session_start + timedelta(minutes=bucket_minutes)

    def _mark_freshness(self, db: Session, source_name: str, status: str, details: dict) -> None:
        row = db.scalar(select(DataFreshness).where(DataFreshness.source_name == source_name))
        if row is None:
            row = DataFreshness(source_name=source_name, last_success_at=datetime.now(IST_ZONE))
            db.add(row)
        row.last_success_at = datetime.now(IST_ZONE)
        row.status = status
        row.details = details

    def _to_candle_record(self, bar: _MinuteBarState) -> CandleRecord:
        return CandleRecord(
            instrument_key=bar.instrument_key,
            interval="1minute",
            ts=bar.ts,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            oi=None,
            source="upstox_ws",
        )

    def _extract_ohlc_records(self, instrument_key: str, payload: dict[str, Any]) -> list[CandleRecord]:
        node = self._extract_full_feed_node(payload)
        if not isinstance(node, dict):
            return []
        market_ohlc = node.get("marketOHLC")
        if not isinstance(market_ohlc, dict):
            return []
        ohlc_rows = market_ohlc.get("ohlc")
        if not isinstance(ohlc_rows, list):
            return []

        out: list[CandleRecord] = []
        oi = self._safe_float(node.get("oi"))
        for row in ohlc_rows:
            if not isinstance(row, dict):
                continue
            interval = str(row.get("interval") or "").strip()
            if interval != "I1":
                continue
            candle_ts = self._parse_epoch_ms(row.get("ts"))
            if candle_ts is None:
                continue
            mapped_interval = "1minute"
            normalized_ts = candle_ts.replace(second=0, microsecond=0)
            open_price = self._safe_float(row.get("open"))
            high_price = self._safe_float(row.get("high"))
            low_price = self._safe_float(row.get("low"))
            close_price = self._safe_float(row.get("close"))
            volume = self._safe_float(row.get("vol"), default=0.0)
            if None in {open_price, high_price, low_price, close_price}:
                continue
            out.append(
                CandleRecord(
                    instrument_key=instrument_key,
                    interval=mapped_interval,
                    ts=normalized_ts,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    oi=oi,
                    source="upstox_ws_ohlc",
                )
            )
        return out

    def _extract_full_feed_node(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        full_feed = payload.get("fullFeed")
        if not isinstance(full_feed, dict):
            return None
        for section in ("indexFF", "marketFF"):
            node = full_feed.get(section)
            if isinstance(node, dict):
                return node
        return None

    def _extract_ltpc(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        ltpc = payload.get("ltpc")
        if isinstance(ltpc, dict):
            return ltpc
        node = self._extract_full_feed_node(payload)
        if isinstance(node, dict) and isinstance(node.get("ltpc"), dict):
            return node["ltpc"]
        return None

    def _extract_order_book_snapshot(
        self,
        instrument_key: str,
        payload: dict[str, Any],
        current_ts: datetime | None,
    ) -> dict[str, Any] | None:
        node = self._extract_full_feed_node(payload)
        if not isinstance(node, dict):
            return None
        market_level = node.get("marketLevel")
        if not isinstance(market_level, dict):
            return None
        levels = market_level.get("bidAskQuote")
        if not isinstance(levels, list) or not levels:
            return None

        bids = []
        asks = []
        for level in levels:
            if not isinstance(level, dict):
                continue
            bid_price = self._safe_float(level.get("bidP"))
            ask_price = self._safe_float(level.get("askP"))
            bid_qty = self._safe_float(level.get("bidQ"), default=0.0) or 0.0
            ask_qty = self._safe_float(level.get("askQ"), default=0.0) or 0.0
            if bid_price is not None:
                bids.append({"price": bid_price, "quantity": bid_qty})
            if ask_price is not None:
                asks.append({"price": ask_price, "quantity": ask_qty})
        if not bids or not asks:
            return None

        best_bid = float(bids[0]["price"])
        best_ask = float(asks[0]["price"])
        if best_bid <= 0 or best_ask <= 0:
            return None
        mid_price = (best_bid + best_ask) / 2.0
        spread_bps = ((best_ask - best_bid) / mid_price * 10000.0) if mid_price > 0 else 0.0
        bid_volume = float(sum(row["quantity"] for row in bids))
        ask_volume = float(sum(row["quantity"] for row in asks))
        total_volume = bid_volume + ask_volume
        depth_imbalance = ((bid_volume - ask_volume) / total_volume) if total_volume > 0 else 0.0
        liquidity_score = min(100.0, total_volume / max(spread_bps, 1.0))
        ltpc = self._extract_ltpc(payload)
        snapshot_ts = self._parse_epoch_ms((ltpc or {}).get("ltt")) or current_ts
        if snapshot_ts is None:
            return None
        return {
            "instrument_key": instrument_key,
            "ts": snapshot_ts,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread_bps": spread_bps,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "depth_imbalance": depth_imbalance,
            "liquidity_score": liquidity_score,
            "depth_data": {"bids": bids, "asks": asks},
        }

    def _parse_epoch_ms(self, raw_value: Any) -> datetime | None:
        try:
            return datetime.fromtimestamp(int(raw_value) / 1000, tz=IST_ZONE)
        except (TypeError, ValueError, OSError):
            return None

    def _safe_float(self, raw_value: Any, default: float | None = None) -> float | None:
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return default

    def _latest_timestamp(self, first: datetime | None, second: datetime | None) -> datetime | None:
        if first is None:
            return second
        if second is None:
            return first
        return second if second > first else first

    def _on_open(self, *_args: Any) -> None:
        logger.info(
            "Upstox websocket connected mode=%s instruments=%s",
            self.settings.upstox_websocket_mode,
            len(self.settings.instrument_keys),
        )

    def _on_error(self, error: Any) -> None:
        logger.error("Upstox websocket error: %s", error)

    def _on_close(self, status_code: Any, message: Any) -> None:
        logger.warning("Upstox websocket closed status=%s message=%s", status_code, message)

    def _on_reconnecting(self, message: Any) -> None:
        logger.warning("Upstox websocket reconnecting: %s", message)

    def _on_auto_reconnect_stopped(self, message: Any) -> None:
        logger.error("Upstox websocket auto reconnect stopped: %s", message)
