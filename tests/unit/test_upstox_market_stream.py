from datetime import datetime
from types import SimpleNamespace

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from data_layer.streamers.upstox_market_stream import UpstoxMarketStream
from db.models import Base, RawCandle
from utils.constants import IST_ZONE


class _FakeStreamer:
    Event = {
        "OPEN": "open",
        "MESSAGE": "message",
        "ERROR": "error",
        "CLOSE": "close",
        "RECONNECTING": "reconnecting",
        "AUTO_RECONNECT_STOPPED": "auto_reconnect_stopped",
    }

    def on(self, *_args, **_kwargs):
        return None

    def auto_reconnect(self, *_args, **_kwargs):
        return None

    def connect(self):
        return None

    def disconnect(self):
        return None


def test_stream_upserts_current_minute_candle_on_each_tick() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
    stream = UpstoxMarketStream(
        settings=SimpleNamespace(
            upstox_websocket_mode="full",
            upstox_websocket_reconnect_interval_seconds=5,
            upstox_websocket_retry_count=5,
            instrument_keys=["NSE_INDEX|Nifty 50"],
        ),
        session_factory=session_factory,
        streamer=_FakeStreamer(),
    )

    instrument_key = "NSE_INDEX|Nifty 50"
    first_tick = datetime(2026, 4, 23, 11, 21, 10, tzinfo=IST_ZONE)
    second_tick = datetime(2026, 4, 23, 11, 21, 42, tzinfo=IST_ZONE)

    stream._update_bar(instrument_key, first_tick, 24000.0)
    stream._flush_pending_records(
        latest_exchange_ts=first_tick,
        received_at=first_tick,
        received_ns=1,
    )

    session = session_factory()
    try:
        first_row = session.scalar(
            select(RawCandle).where(
                RawCandle.instrument_key == instrument_key,
                RawCandle.interval == "1minute",
            )
        )
        assert first_row is not None
        assert first_row.ts.isoformat() == "2026-04-23T11:21:00"
        assert float(first_row.close) == 24000.0
    finally:
        session.close()

    stream._update_bar(instrument_key, second_tick, 24012.5)
    stream._flush_pending_records(
        latest_exchange_ts=second_tick,
        received_at=second_tick,
        received_ns=2,
    )

    session = session_factory()
    try:
        updated_row = session.scalar(
            select(RawCandle).where(
                RawCandle.instrument_key == instrument_key,
                RawCandle.interval == "1minute",
            )
        )
        row_count = session.scalar(
            select(func.count(RawCandle.id)).where(
                RawCandle.instrument_key == instrument_key,
                RawCandle.interval == "1minute",
            )
        )

        assert updated_row is not None
        assert row_count == 1
        assert float(updated_row.high) == 24012.5
        assert float(updated_row.close) == 24012.5
    finally:
        session.close()
