from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from db.models import Base, DataFreshness, RawCandle
from execution_engine.live_service import MarketContext, build_live_price_update, build_technical_signal
from utils.config import Settings
from utils.constants import IST_ZONE


class _FakeDb:
    def __init__(self, *, signal_count: int = 0, latest_signal_ts: datetime | None = None) -> None:
        self.signal_count = signal_count
        self.latest_signal_ts = latest_signal_ts
        self.calls = 0

    def scalar(self, _query):
        self.calls += 1
        if self.calls == 1:
            return self.signal_count
        return self.latest_signal_ts


def _make_rows() -> list[SimpleNamespace]:
    start = datetime(2026, 4, 21, 10, 0, tzinfo=IST_ZONE)
    rows: list[SimpleNamespace] = []
    price = 24000.0
    for index in range(70):
        price += 3.0
        open_price = price - 1.0
        close_price = price + 1.2
        rows.append(
            SimpleNamespace(
                ts=start + timedelta(minutes=index),
                open=open_price,
                high=close_price + 1.4,
                low=open_price - 1.4,
                close=close_price,
                volume=140.0 + index,
            )
        )
    rows.append(
        SimpleNamespace(
            ts=start + timedelta(minutes=70),
            open=rows[-1].close + 2.0,
            high=rows[-1].close + 34.0,
            low=rows[-1].close,
            close=rows[-1].close + 30.0,
            volume=420.0,
        )
    )
    return rows


def _context() -> MarketContext:
    rows = _make_rows()
    latest = rows[-1]
    return MarketContext(
        symbol="Nifty 50",
        instrument_key="NSE_INDEX|Nifty 50",
        latest_price=float(latest.close),
        latest_candle_ts=latest.ts,
        chart_rows=rows[-60:],
        signal_rows=rows,
        technical_context={},
        current_bar={
            "open": float(latest.open),
            "high": float(latest.high),
            "low": float(latest.low),
            "close": float(latest.close),
            "volume": float(latest.volume),
        },
    )


def test_build_technical_signal_returns_disabled_hold_payload() -> None:
    context = _context()
    fake_db = _FakeDb()

    signal = build_technical_signal(
        fake_db,
        context=context,
        now=datetime(2026, 4, 21, 11, 16, tzinfo=IST_ZONE),
    )

    assert signal.action == "HOLD"
    assert signal.bias == "NEUTRAL"
    assert signal.conviction == "disabled"
    assert signal.cooldown_seconds == 0
    assert signal.details["signals_enabled"] is False
    assert any("disabled" in reason.lower() for reason in signal.reasons)


def test_build_technical_signal_skips_guardrails_when_disabled() -> None:
    context = _context()
    fake_db = _FakeDb(
        signal_count=1,
        latest_signal_ts=datetime(2026, 4, 21, 11, 10, tzinfo=IST_ZONE),
    )

    signal = build_technical_signal(
        fake_db,
        context=context,
        now=datetime(2026, 4, 21, 11, 16, tzinfo=IST_ZONE),
    )

    assert signal.action == "HOLD"
    assert signal.cooldown_seconds == 0
    assert signal.details["signals_enabled"] is False
    assert any("data-only mode" in reason.lower() for reason in signal.reasons)


def test_build_chart_payload_exposes_two_year_range_options() -> None:
    from execution_engine.live_service import build_chart_payload

    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        start = datetime(2026, 4, 21, 9, 15, tzinfo=IST_ZONE)
        for index in range(80):
            base = 24000.0 + (index * 2.0)
            session.add(
                RawCandle(
                    instrument_key="NSE_INDEX|Nifty 50",
                    interval="1minute",
                    ts=start + timedelta(minutes=index),
                    open=base,
                    high=base + 4.0,
                    low=base - 4.0,
                    close=base + 1.0,
                    volume=100.0 + index,
                    oi=None,
                    source="test",
                )
            )
        session.commit()

        payload = build_chart_payload(
            session,
            symbol="Nifty 50",
            range_key="2y",
            now=datetime(2026, 4, 22, 10, 0, tzinfo=IST_ZONE),
        )

        range_keys = [item["key"] for item in payload["available_ranges"]]
        assert payload["range"] == "2y"
        assert payload["interval"] == "day"
        assert payload["source_interval"] == "1minute"
        assert payload["is_resampled"] is True
        assert payload["start_date"] == "2024-04-22"
        assert payload["end_date"] == "2026-04-21"
        assert payload["markers"] == []
        assert "2y" in range_keys
        assert payload["candles"]
    finally:
        session.close()


def test_build_chart_payload_uses_latest_available_session_for_one_day_view() -> None:
    from execution_engine.live_service import build_chart_payload

    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        start = datetime(2026, 4, 21, 9, 15, tzinfo=IST_ZONE)
        for index in range(5):
            session.add(
                RawCandle(
                    instrument_key="NSE_INDEX|Nifty 50",
                    interval="1minute",
                    ts=start + timedelta(minutes=index),
                    open=24000.0,
                    high=24002.0,
                    low=23998.0,
                    close=24001.0,
                    volume=100.0,
                    oi=None,
                    source="test",
                )
            )
        session.commit()

        payload = build_chart_payload(
            session,
            symbol="Nifty 50",
            range_key="1d",
            now=datetime(2026, 4, 22, 10, 0, tzinfo=IST_ZONE),
        )

        assert payload["range"] == "1d"
        assert payload["start_date"] == "2026-04-21"
        assert payload["end_date"] == "2026-04-21"
        assert len(payload["candles"]) == 5
        assert payload["markers"] == []
    finally:
        session.close()


def test_build_live_price_update_includes_stream_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(
        "execution_engine.live_service.get_market_stream_runtime_status",
        lambda _settings=None: {
            "owner": "api_process",
            "autostart_enabled": True,
            "autostart_expected": True,
            "running": True,
            "thread_alive": True,
            "last_started_at": "2026-04-23T09:15:00+05:30",
            "last_error": None,
        },
    )

    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        candle_ts = datetime(2026, 4, 23, 9, 15, tzinfo=IST_ZONE)
        session.add(
            RawCandle(
                instrument_key="NSE_INDEX|Nifty 50",
                interval="1minute",
                ts=candle_ts,
                open=24000.0,
                high=24005.0,
                low=23995.0,
                close=24002.0,
                volume=100.0,
                oi=None,
                source="test",
            )
        )
        session.add(
            DataFreshness(
                source_name="upstox_market_stream",
                last_success_at=candle_ts + timedelta(milliseconds=250),
                status="ok",
                details={
                    "latest_exchange_ts": "2026-04-23T09:15:00+05:30",
                    "message_received_at": "2026-04-23T09:15:00.100000+05:30",
                    "write_completed_at": "2026-04-23T09:15:00.250000+05:30",
                    "exchange_timestamp_precision": "milliseconds",
                    "estimated_exchange_to_receive_latency_ns": 100_000_000,
                    "estimated_receive_to_persist_latency_ns": 150_000_000,
                    "estimated_exchange_to_persist_latency_ns": 250_000_000,
                    "candles_flushed": 1,
                    "order_books_flushed": 0,
                    "source": "upstox_ws",
                },
            )
        )
        session.commit()

        payload = build_live_price_update(session, symbol="Nifty 50", settings=Settings(market_stream_autostart=True))

        assert payload["candle"]["x"] == "2026-04-23T09:15:00+05:30"
        assert payload["stream"]["status"] == "ok"
        assert payload["stream"]["latest_exchange_ts"] == "2026-04-23T09:15:00+05:30"
        assert payload["stream"]["estimated_exchange_to_persist_latency_ns"] == 250_000_000
        assert payload["stream"]["runtime"]["running"] is True
        assert payload["stream"]["runtime"]["autostart_expected"] is True
    finally:
        session.close()


def test_settings_should_autostart_market_stream_uses_safe_defaults() -> None:
    base = Settings(market_data_mode="websocket", market_stream_autostart=None)
    sqlite_settings = base.model_copy(update={"database_url_override": "sqlite:///./test.db"})
    postgres_settings = base.model_copy(update={"database_url_override": "postgresql+psycopg://user:pass@localhost/db"})
    forced_off = base.model_copy(
        update={
            "database_url_override": "sqlite:///./test.db",
            "market_stream_autostart": False,
        }
    )

    assert sqlite_settings.should_autostart_market_stream is True
    assert postgres_settings.should_autostart_market_stream is False
    assert forced_off.should_autostart_market_stream is False
