from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.db.models import Base, DataFreshness, ExecutionPosition, OptionQuote, RawCandle, SignalLog
from backend.execution_engine.live_service import (
    MarketContext,
    build_live_price_update,
    build_technical_signal,
    compute_paper_portfolio_metrics,
    latest_option_premium,
    refresh_open_positions_snapshot,
)
from backend.utils.config import Settings
from backend.utils.constants import IST_ZONE


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


def test_fresh_pine_marker_uses_only_current_trading_session(monkeypatch) -> None:
    from backend.execution_engine.live_service import _latest_fresh_pine_marker

    previous_start = datetime(2026, 6, 17, 14, 0, tzinfo=IST_ZONE)
    pre_open_start = datetime(2026, 6, 18, 9, 0, tzinfo=IST_ZONE)
    current_start = datetime(2026, 6, 18, 9, 15, tzinfo=IST_ZONE)
    rows = []
    for index in range(60):
        rows.append(
            SimpleNamespace(
                ts=previous_start + timedelta(minutes=index),
                open=24000.0,
                high=24002.0,
                low=23998.0,
                close=24001.0,
                volume=100.0,
            )
        )
    for index in range(15):
        rows.append(
            SimpleNamespace(
                ts=pre_open_start + timedelta(minutes=index),
                open=24050.0,
                high=24052.0,
                low=24048.0,
                close=24051.0,
                volume=100.0,
            )
        )
    for index in range(60):
        rows.append(
            SimpleNamespace(
                ts=current_start + timedelta(minutes=index),
                open=24100.0,
                high=24102.0,
                low=24098.0,
                close=24101.0,
                volume=100.0,
            )
        )

    captured_rows = []

    def fake_overlay(overlay_rows, **_kwargs):
        captured_rows.extend(overlay_rows)
        return {
            "markers": [
                {
                    "time": (current_start + timedelta(minutes=59)).isoformat(),
                    "text": "BUY",
                }
            ],
            "levels": [],
        }

    monkeypatch.setattr(
        "backend.execution_engine.live_service._build_pine_chart_overlay",
        fake_overlay,
    )

    marker = _latest_fresh_pine_marker(
        rows,
        settings=Settings(),
        candle_ts=current_start + timedelta(minutes=59),
    )

    assert marker is not None
    assert marker["text"] == "BUY"
    assert len(captured_rows) == 60
    assert {row["ts"].date() for row in captured_rows} == {current_start.date()}
    assert min(row["ts"] for row in captured_rows).time().isoformat() == "09:15:00"


def test_serialize_signal_log_exposes_execution_decision() -> None:
    from backend.execution_engine.live_service import _serialize_signal_log

    candle_ts = datetime(2026, 6, 18, 10, 37, tzinfo=IST_ZONE)
    row = SignalLog(
        id=42,
        timestamp=candle_ts,
        trade_date=candle_ts.date(),
        symbol="Nifty 50",
        interval="1minute",
        ml_signal="BUY",
        ml_confidence=0.8,
        pine_signal="BUY",
        ai_score=0.0,
        combined_score=1.0,
        consensus="BUY",
        trade_placed=False,
        skip_reason="No liquid option contract passed the live filter.",
        details={
            "fresh_graph_marker": True,
            "pine_marker_time": candle_ts.isoformat(),
            "pine_marker_text": "BUY",
        },
    )

    payload = _serialize_signal_log(row)

    assert payload["id"] == 42
    assert payload["consensus"] == "BUY"
    assert payload["trade_placed"] is False
    assert payload["fresh_graph_marker"] is True
    assert payload["pine_marker_text"] == "BUY"
    assert payload["skip_reason"] == "No liquid option contract passed the live filter."


def test_chart_payload_attaches_persisted_execution_signal_marker() -> None:
    from backend.execution_engine.live_service import _attach_execution_signal_markers

    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        signal_ts = datetime(2026, 6, 18, 10, 40, tzinfo=IST_ZONE)
        session.add(
            SignalLog(
                timestamp=signal_ts,
                trade_date=signal_ts.date(),
                symbol="Nifty 50",
                interval="1minute",
                ml_signal="SELL",
                ml_confidence=1.0,
                pine_signal="SELL",
                ai_score=0.0,
                combined_score=1.0,
                consensus="SELL",
                trade_placed=True,
                details={"fresh_graph_marker": True},
            )
        )
        session.commit()

        payload = _attach_execution_signal_markers(
            session,
            {
                "candles": [
                    {"x": "2026-06-18T09:15:00+05:30"},
                    {"x": "2026-06-18T10:50:00+05:30"},
                ],
                "markers": [],
            },
            symbol="Nifty 50",
        )

        assert payload["markers"] == [
            {
                "time": "2026-06-18T10:40:00+05:30",
                "position": "aboveBar",
                "color": "#dc2626",
                "shape": "arrowDown",
                "text": "SELL",
                "trade_placed": True,
            }
        ]
    finally:
        session.close()


def test_build_technical_signal_holds_when_vix_too_high(monkeypatch) -> None:
    """VIX ratio >1.40 above its MA must force HOLD regardless of technicals."""
    # VIX=25, MA=17 → ratio≈1.47 (>1.40 threshold)
    monkeypatch.setattr(
        "backend.execution_engine.live_service.get_vix_context",
        lambda _db: (25.0, 17.0, 25.0 / 17.0),
    )
    context = _context()
    fake_db = _FakeDb()

    signal = build_technical_signal(
        fake_db,
        context=context,
        now=datetime(2026, 4, 21, 11, 16, tzinfo=IST_ZONE),
    )

    assert signal.action == "HOLD"
    assert signal.details["vix_level"] == 25.0
    assert signal.details["vix_too_high"] is True
    assert any("vix" in reason.lower() for reason in signal.reasons)


def test_build_technical_signal_holds_during_cooldown(monkeypatch) -> None:
    """When a signal was fired 6 min ago and cooldown is 12 min, action must be HOLD."""
    # VIX=15, MA=15 → ratio=1.0 (neutral, no VIX block)
    monkeypatch.setattr(
        "backend.execution_engine.live_service.get_vix_context",
        lambda _db: (15.0, 15.0, 1.0),
    )
    context = _context()
    # Latest signal 6 minutes ago; default cooldown = 12 minutes → 360 s remaining
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
    assert signal.cooldown_seconds > 0
    assert any("cooldown" in reason.lower() for reason in signal.reasons)


def test_build_chart_payload_forces_one_minute_all_range() -> None:
    from backend.execution_engine.live_service import build_chart_payload

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
        interval_keys = [item["interval"] for item in payload["available_intervals"]]
        assert payload["range"] == "all"
        assert payload["interval"] == "1minute"
        assert payload["source_interval"] == "1minute"
        assert payload["is_resampled"] is False
        assert payload["start_date"] == "2026-04-21"
        assert payload["end_date"] == "2026-04-21"
        assert {marker["text"] for marker in payload["markers"]}.issubset({"BUY", "SELL"})
        assert range_keys == ["all"]
        assert interval_keys == ["1minute"]
        assert len(payload["candles"]) == 80
    finally:
        session.close()


def test_build_chart_payload_ignores_one_day_request() -> None:
    from backend.execution_engine.live_service import build_chart_payload

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

        assert payload["range"] == "all"
        assert payload["interval"] == "1minute"
        assert payload["start_date"] == "2026-04-21"
        assert payload["end_date"] == "2026-04-21"
        assert len(payload["candles"]) == 5
        assert payload["markers"] == []
    finally:
        session.close()


def test_chart_payload_uses_partial_current_session_before_fifty_candles() -> None:
    from backend.execution_engine.live_service import build_chart_payload

    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        instrument_key = "NSE_INDEX|Nifty 50"
        previous_start = datetime(2026, 6, 17, 9, 15, tzinfo=IST_ZONE)
        current_start = datetime(2026, 6, 18, 9, 15, tzinfo=IST_ZONE)
        for index in range(60):
            session.add(
                RawCandle(
                    instrument_key=instrument_key,
                    interval="1minute",
                    ts=previous_start + timedelta(minutes=index),
                    open=24000.0,
                    high=24002.0,
                    low=23998.0,
                    close=24001.0,
                    volume=100.0,
                    source="test",
                )
            )
        for index in range(22):
            session.add(
                RawCandle(
                    instrument_key=instrument_key,
                    interval="1minute",
                    ts=current_start + timedelta(minutes=index),
                    open=24100.0,
                    high=24102.0,
                    low=24098.0,
                    close=24101.0,
                    volume=100.0,
                    source="test",
                )
            )
        session.commit()

        payload = build_chart_payload(
            session,
            symbol="Nifty 50",
            range_key="all",
            now=datetime(2026, 6, 18, 9, 36, tzinfo=IST_ZONE),
        )

        assert payload["latest"] == "2026-06-18T09:36:00+05:30"
        assert len(payload["candles"]) == 82
    finally:
        session.close()


def test_build_live_price_update_includes_stream_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.execution_engine.live_service.get_market_stream_runtime_status",
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


def test_latest_option_premium_uses_synthetic_fallback_when_db_quote_is_stale() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime(2026, 5, 6, tzinfo=IST_ZONE).date()
        session.add(
            RawCandle(
                instrument_key="NSE_INDEX|Nifty 50",
                interval="1minute",
                ts=datetime(2026, 5, 4, 10, 0, tzinfo=IST_ZONE),
                open=24100.0,
                high=24120.0,
                low=24090.0,
                close=24110.0,
                volume=100.0,
                oi=None,
                source="test",
            )
        )
        session.add(
            OptionQuote(
                instrument_key="NSE_FO|STALE",
                underlying_key="NSE_INDEX|Nifty 50",
                underlying_symbol="Nifty 50",
                expiry_date=expiry,
                strike=24100.0,
                option_type="CE",
                ts=datetime(2026, 5, 4, 9, 30, tzinfo=IST_ZONE),
                ltp=201.0,
                bid=200.0,
                ask=202.0,
                volume=100.0,
                oi=1000.0,
                close_price=198.0,
                source="upstox_option_chain",
            )
        )
        session.commit()

        price = latest_option_premium(
            session,
            symbol="Nifty 50",
            expiry_date=expiry,
            strike=24100.0,
            option_type="CE",
            settings=Settings(upstox_access_token="", option_chain_refresh_seconds=4),
        )

        assert price is not None
        assert price != 201.0
        assert price > 0.0
    finally:
        session.close()


def test_refresh_open_positions_snapshot_updates_live_premium_and_pnl() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime(2026, 5, 6, tzinfo=IST_ZONE).date()
        session.add(
            RawCandle(
                instrument_key="NSE_INDEX|Nifty 50",
                interval="1minute",
                ts=datetime(2026, 5, 4, 10, 0, tzinfo=IST_ZONE),
                open=24100.0,
                high=24120.0,
                low=24090.0,
                close=24110.0,
                volume=100.0,
                oi=None,
                source="test",
            )
        )
        position = ExecutionPosition(
            trade_date=datetime(2026, 5, 4, tzinfo=IST_ZONE).date(),
            symbol="Nifty 50",
            interval="1minute",
            strategy_name="test",
            option_type="CE",
            side="BUY",
            expiry_date=expiry,
            strike=24100.0,
            quantity=50,
            status="OPEN",
            entry_price=100.0,
            entry_premium=100.0,
            stop_loss=80.0,
            initial_sl=80.0,
            current_sl=80.0,
            trailing_stop=0.0,
            peak_premium=100.0,
            tsl_active=False,
            take_profit=140.0,
            target_premium=140.0,
            current_price=100.0,
            current_premium=100.0,
            pnl_points=0.0,
            pnl_value=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            metadata_json={"instrument_key": "NSE_FO|24100CE"},
        )
        session.add(position)
        session.commit()

        rows = refresh_open_positions_snapshot(session, settings=Settings(upstox_access_token=""))

        assert len(rows) == 1
        assert rows[0].current_premium is not None
        assert rows[0].current_premium != 100.0
        assert rows[0].unrealized_pnl is not None
        assert rows[0].metadata_json["latest_quote_source"] in {"synthetic_fallback", "synthetic"}
    finally:
        session.close()


def test_compute_paper_portfolio_metrics_derives_balance_from_open_and_closed_positions() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        session.add(
            ExecutionPosition(
                trade_date=datetime(2026, 5, 4, tzinfo=IST_ZONE).date(),
                symbol="Nifty 50",
                interval="1minute",
                strategy_name="test",
                option_type="CE",
                side="BUY",
                expiry_date=datetime(2026, 5, 6, tzinfo=IST_ZONE).date(),
                strike=24100.0,
                quantity=50,
                status="OPEN",
                entry_price=100.0,
                entry_premium=100.0,
                stop_loss=80.0,
                trailing_stop=0.0,
                current_premium=110.0,
                unrealized_pnl=500.0,
                metadata_json={},
            )
        )
        session.add(
            ExecutionPosition(
                trade_date=datetime(2026, 5, 3, tzinfo=IST_ZONE).date(),
                symbol="Bank Nifty",
                interval="1minute",
                strategy_name="test",
                option_type="PE",
                side="BUY",
                expiry_date=datetime(2026, 5, 6, tzinfo=IST_ZONE).date(),
                strike=56000.0,
                quantity=25,
                status="CLOSED",
                entry_price=200.0,
                entry_premium=200.0,
                stop_loss=170.0,
                trailing_stop=0.0,
                exit_premium=220.0,
                realized_pnl=500.0,
                metadata_json={},
            )
        )
        session.commit()

        metrics = compute_paper_portfolio_metrics(session, settings=Settings(execution_capital=100000.0))

        assert metrics["starting_balance"] == 100000.0
        assert metrics["invested_amount"] == 5000.0
        assert metrics["realized_pnl"] == 500.0
        assert metrics["unrealized_pnl"] == 500.0
        assert metrics["available_balance"] == 95500.0
        assert metrics["equity"] == 101000.0
    finally:
        session.close()
