from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.db.models import Base, DataFreshness, ExecutionPosition, OptionQuote, RawCandle, SignalLog
from backend.execution_engine.live_service import (
    MarketContext,
    TechnicalSignal,
    _maybe_refresh_option_chain,
    _option_liquidity_failures,
    build_live_price_update,
    build_option_selection,
    build_technical_signal,
    compute_sandbox_portfolio_metrics,
    latest_option_premium,
    refresh_open_positions_snapshot,
    resolve_live_option_quote,
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


def test_option_expiry_lookup_is_cached_between_fast_ui_polls(monkeypatch) -> None:
    from backend.api.routes.live import _OPTION_EXPIRY_CACHE, _cached_option_expiries

    calls = []

    class FakeCollector:
        def __init__(self, _settings) -> None:
            pass

        def list_expiries(self, underlying_key, max_items):
            calls.append((underlying_key, max_items))
            return [date(2026, 6, 25)]

    _OPTION_EXPIRY_CACHE.clear()
    monkeypatch.setattr("backend.api.routes.live.UpstoxOptionChainCollector", FakeCollector)
    settings = SimpleNamespace(has_market_data_access=True)

    first = _cached_option_expiries("NSE_INDEX|Nifty 50", settings)
    second = _cached_option_expiries("NSE_INDEX|Nifty 50", settings)

    assert first == second == [date(2026, 6, 25)]
    assert calls == [("NSE_INDEX|Nifty 50", 6)]


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


def _option_context(*, latest_price: float = 24110.0) -> MarketContext:
    return MarketContext(
        symbol="Nifty 50",
        instrument_key="NSE_INDEX|Nifty 50",
        latest_price=latest_price,
        latest_candle_ts=datetime.now(IST_ZONE),
        chart_rows=[],
        signal_rows=[],
        technical_context={},
        current_bar={
            "open": latest_price,
            "high": latest_price,
            "low": latest_price,
            "close": latest_price,
            "volume": 0.0,
        },
    )


def _option_signal(*, action: str = "BUY") -> TechnicalSignal:
    return TechnicalSignal(
        symbol="Nifty 50",
        interval="1minute",
        timestamp=datetime.now(IST_ZONE),
        action=action,
        bias=action,
        score=80.0,
        confidence=0.8,
        conviction="high",
        entry_price=24110.0,
        stop_loss=24080.0,
        take_profit=24170.0,
        cooldown_seconds=0,
        max_signals_reached=False,
        reasons=[],
        details={"atr_14": 10.0},
    )


def _add_option_quote(
    session: Session,
    *,
    expiry: date,
    strike: float,
    option_type: str = "CE",
    ts: datetime | None = None,
    ltp: float = 100.0,
    bid: float = 99.0,
    ask: float = 101.0,
    volume: float = 1000.0,
    oi: float = 2000.0,
    source: str = "upstox_option_chain",
) -> None:
    session.add(
        OptionQuote(
            instrument_key=f"NSE_FO|{int(strike)}{option_type}",
            underlying_key="NSE_INDEX|Nifty 50",
            underlying_symbol="Nifty 50",
            expiry_date=expiry,
            strike=strike,
            option_type=option_type,
            ts=ts or datetime.now(IST_ZONE),
            ltp=ltp,
            bid=bid,
            ask=ask,
            volume=volume,
            oi=oi,
            close_price=ltp,
            source=source,
        )
    )


def test_fresh_pine_marker_preserves_state_across_regular_sessions(monkeypatch) -> None:
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
    assert len(captured_rows) == 120
    assert {row["ts"].date() for row in captured_rows} == {
        previous_start.date(),
        current_start.date(),
    }
    assert all(row["ts"].time().isoformat() >= "09:15:00" for row in captured_rows)


def test_pine_rma_matches_tradingview_sma_seed_and_wilder_smoothing() -> None:
    import pandas as pd

    from backend.execution_engine.live_service import _pine_rma

    result = _pine_rma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)

    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(8.0 / 3.0)
    assert result.iloc[4] == pytest.approx(31.0 / 9.0)


def test_pine_marker_calculation_uses_twenty_bar_script_warmup() -> None:
    from backend.execution_engine.live_service import _build_pine_chart_overlay

    start = datetime(2026, 6, 18, 9, 15, tzinfo=IST_ZONE)
    rows = [
        {
            "ts": start + timedelta(minutes=index),
            "open": 24000.0 + index,
            "high": 24003.0 + index,
            "low": 23997.0 + index,
            "close": 24001.0 + index,
            "volume": 100.0,
        }
        for index in range(20)
    ]

    overlay = _build_pine_chart_overlay(
        rows,
        interval="1minute",
        settings=Settings(),
        range_key="1d",
    )

    assert set(overlay) == {"markers", "levels"}
    assert _build_pine_chart_overlay(
        rows[:-1],
        interval="1minute",
        settings=Settings(),
        range_key="1d",
    ) == {"markers": [], "levels": []}


def test_pine_overlay_matches_reference_crossover_sequence() -> None:
    from backend.execution_engine.live_service import _build_pine_chart_overlay

    start = datetime(2026, 6, 15, 9, 15, tzinfo=IST_ZONE)
    closes = (
        [100.0] * 25
        + [100.0 + (3.0 * index) for index in range(1, 11)]
        + [130.0 - (4.0 * index) for index in range(1, 16)]
        + [70.0 + (5.0 * index) for index in range(1, 16)]
    )
    rows = []
    for index, close in enumerate(closes):
        previous_close = closes[index - 1] if index else close
        rows.append(
            {
                "ts": start + timedelta(minutes=index),
                "open": previous_close,
                "high": max(previous_close, close) + 1.0,
                "low": min(previous_close, close) - 1.0,
                "close": close,
                "volume": 100.0,
            }
        )

    overlay = _build_pine_chart_overlay(
        rows,
        interval="1minute",
        settings=Settings(),
        range_key="all",
    )

    assert [(marker["time"], marker["text"]) for marker in overlay["markers"]] == [
        ("2026-06-15T09:44:00+05:30", "BUY"),
        ("2026-06-15T09:56:00+05:30", "SELL"),
        ("2026-06-15T10:12:00+05:30", "BUY"),
    ]
    hidden_overlay = _build_pine_chart_overlay(
        rows,
        interval="1minute",
        settings=Settings(PINE_SIGNAL_SHOW_SIGNALS=False),
        range_key="all",
    )
    assert hidden_overlay
    assert hidden_overlay["markers"] == overlay["markers"]


def test_fresh_pine_marker_is_cached_per_closed_instrument_candle(monkeypatch) -> None:
    from backend.execution_engine.live_service import _PINE_MARKER_CACHE, _latest_fresh_pine_marker

    _PINE_MARKER_CACHE.clear()
    start = datetime(2026, 6, 18, 9, 15, tzinfo=IST_ZONE)
    rows = [
        SimpleNamespace(
            ts=start + timedelta(minutes=index),
            open=24000.0,
            high=24002.0,
            low=23998.0,
            close=24001.0,
            volume=100.0,
        )
        for index in range(20)
    ]
    calls = 0

    def fake_overlay(_rows, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "markers": [{"time": rows[-1].ts.isoformat(), "text": "BUY"}],
            "levels": [],
        }

    monkeypatch.setattr(
        "backend.execution_engine.live_service._build_pine_chart_overlay",
        fake_overlay,
    )

    for _ in range(2):
        marker = _latest_fresh_pine_marker(
            rows,
            settings=Settings(),
            candle_ts=rows[-1].ts,
            instrument_key="NSE_INDEX|Nifty 50",
        )
        assert marker is not None
        assert marker["text"] == "BUY"

    assert calls == 1


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
            "option_selection": {
                "quote_status": "unavailable",
                "quote_source": None,
                "quote_ts": None,
                "quote_age_seconds": None,
                "requested_atm": 24100.0,
                "candidate_diagnostics": [
                    {
                        "strike": 24100.0,
                        "status": "rejected",
                        "rejections": ["Quote is stale."],
                    }
                ],
                "reasons": ["Real Upstox option chain is stale."],
            },
        },
    )

    payload = _serialize_signal_log(row)

    assert payload["id"] == 42
    assert payload["consensus"] == "BUY"
    assert payload["trade_placed"] is False
    assert payload["fresh_graph_marker"] is True
    assert payload["pine_marker_text"] == "BUY"
    assert payload["skip_reason"] == "No liquid option contract passed the live filter."
    assert payload["details"] == row.details
    assert payload["option_selection"] == row.details["option_selection"]
    assert payload["quote_status"] == "unavailable"
    assert payload["requested_atm"] == 24100.0
    assert payload["candidate_diagnostics"][0]["strike"] == 24100.0
    assert payload["selection_reasons"] == ["Real Upstox option chain is stale."]


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
        now=datetime(2026, 4, 21, 11, 12, tzinfo=IST_ZONE),
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
        latest_signal_ts=datetime(2026, 4, 21, 11, 6, tzinfo=IST_ZONE),
    )

    signal = build_technical_signal(
        fake_db,
        context=context,
        now=datetime(2026, 4, 21, 11, 12, tzinfo=IST_ZONE),
    )

    assert signal.action == "HOLD"
    assert signal.cooldown_seconds > 0
    assert any("cooldown" in reason.lower() for reason in signal.reasons)


def test_build_technical_signal_applies_symbol_adx_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.execution_engine.live_service.get_vix_context",
        lambda _db: (15.0, 15.0, 1.0),
    )
    monkeypatch.setattr(
        "backend.execution_engine.live_service._latest_fresh_pine_marker",
        lambda *_args, **_kwargs: {"time": "2026-04-21T11:10:00+05:30", "text": "BUY"},
    )

    signal = build_technical_signal(
        _FakeDb(),
        context=_context(),
        settings=Settings(
            _env_file=None,
            SIGNAL_SYMBOL_PROFILES='{"Nifty 50":{"signal_min_adx":100}}',
        ),
        now=datetime(2026, 4, 21, 11, 12, tzinfo=IST_ZONE),
    )

    assert signal.details["raw_signal"] == "BUY"
    assert signal.action == "HOLD"
    assert signal.details["adx_ok"] is False
    assert any("ADX" in reason for reason in signal.reasons)


def test_build_chart_payload_forces_single_one_minute_mode(monkeypatch) -> None:
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
        marker_time = start + timedelta(minutes=60)
        monkeypatch.setattr(
            "backend.execution_engine.live_service._build_pine_chart_overlay",
            lambda *_args, **_kwargs: {
                "markers": [
                    {
                        "time": marker_time.isoformat(),
                        "position": "belowBar",
                        "color": "#16a34a",
                        "shape": "arrowUp",
                        "text": "BUY",
                    }
                ],
                "levels": [],
            },
        )

        payload = build_chart_payload(
            session,
            symbol="Nifty 50",
            range_key="2y",
            interval_key="1hour",
            now=datetime(2026, 4, 22, 10, 0, tzinfo=IST_ZONE),
        )

        range_keys = [item["key"] for item in payload["available_ranges"]]
        interval_keys = [item["interval"] for item in payload["available_intervals"]]
        assert payload["range"] == "recent"
        assert payload["interval"] == "1minute"
        assert payload["source_interval"] == "1minute"
        assert payload["is_resampled"] is False
        assert payload["start_date"] == "2026-04-15"
        assert payload["end_date"] == "2026-04-21"
        assert payload["markers"] == [
            {
                "time": marker_time.isoformat(),
                "position": "belowBar",
                "color": "#16a34a",
                "shape": "arrowUp",
                "text": "BUY",
            }
        ]
        assert payload["pine_levels"] == []
        assert range_keys == ["recent"]
        assert interval_keys == ["1minute"]
        assert len(payload["candles"]) == 80
    finally:
        session.close()


def test_build_chart_payload_ignores_legacy_one_day_request() -> None:
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

        assert payload["range"] == "recent"
        assert payload["interval"] == "1minute"
        assert payload["start_date"] == "2026-04-15"
        assert payload["end_date"] == "2026-04-21"
        assert len(payload["candles"]) == 5
        assert payload["markers"] == []
    finally:
        session.close()


def test_chart_payload_uses_recent_one_minute_sessions() -> None:
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
        assert payload["range"] == "recent"
        assert payload["interval"] == "1minute"
        assert payload["start_date"] == "2026-06-12"
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


def test_build_option_selection_uses_best_liquid_nearby_real_strike(monkeypatch) -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime.now(IST_ZONE).date() + timedelta(days=5)
        now = datetime.now(IST_ZONE)
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=now,
            volume=100.0,
            oi=250.0,
        )
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24050.0,
            ts=now,
            bid=99.0,
            ask=101.0,
            volume=600.0,
            oi=1200.0,
        )
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24150.0,
            ts=now,
            bid=99.75,
            ask=100.25,
            volume=5000.0,
            oi=10000.0,
        )
        session.commit()
        monkeypatch.setattr(
            "backend.execution_engine.live_service._resolve_expiry",
            lambda **_kwargs: (expiry, [expiry]),
        )
        monkeypatch.setattr(
            "backend.execution_engine.live_service._maybe_refresh_option_chain",
            lambda *_args, **_kwargs: None,
        )

        selection = build_option_selection(
            session,
            context=_option_context(),
            signal=_option_signal(),
                settings=Settings(
                    _env_file=None,
                    execution_mode="live",
                    upstox_access_token="",
                option_chain_refresh_seconds=4,
                option_min_volume=500,
                option_min_oi=1000,
                option_max_spread_pct=0.08,
            ),
        )

        assert selection.signal["action"] == "BUY"
        assert selection.signal["strike"] == 24150.0
        assert selection.signal["quote_source"] == "upstox_option_chain"
        assert selection.signal["quote_age_seconds"] is not None
        diagnostics = selection.signal["candidate_diagnostics"]
        assert len(diagnostics) == 5
        assert next(item for item in diagnostics if item["strike"] == 24150.0)["status"] == "selected"
        atm = next(item for item in diagnostics if item["strike"] == 24100.0)
        assert any("volume" in reason.lower() for reason in atm["rejections"])
        assert any("oi" in reason.lower() for reason in atm["rejections"])
    finally:
        session.close()


def test_build_option_selection_rejects_stale_real_chain(monkeypatch) -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime.now(IST_ZONE).date() + timedelta(days=5)
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=datetime.now(IST_ZONE) - timedelta(minutes=5),
            volume=5000.0,
            oi=10000.0,
        )
        session.commit()
        monkeypatch.setattr(
            "backend.execution_engine.live_service._resolve_expiry",
            lambda **_kwargs: (expiry, [expiry]),
        )
        monkeypatch.setattr(
            "backend.execution_engine.live_service._maybe_refresh_option_chain",
            lambda *_args, **_kwargs: None,
        )

        selection = build_option_selection(
            session,
            context=_option_context(),
            signal=_option_signal(),
            settings=Settings(_env_file=None, upstox_access_token="", option_chain_refresh_seconds=4),
        )

        assert selection.signal["action"] == "HOLD"
        assert selection.signal["quote_status"] == "unavailable"
        assert any("stale" in reason.lower() for reason in selection.signal["reasons"])
        atm = next(
            item
            for item in selection.signal["candidate_diagnostics"]
            if item["strike"] == 24100.0
        )
        assert any("stale" in reason.lower() for reason in atm["rejections"])
    finally:
        session.close()


def test_build_option_selection_keeps_synthetic_chain_display_only(monkeypatch) -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime.now(IST_ZONE).date() + timedelta(days=5)
        monkeypatch.setattr(
            "backend.execution_engine.live_service._resolve_expiry",
            lambda **_kwargs: (expiry, [expiry]),
        )
        monkeypatch.setattr(
            "backend.execution_engine.live_service._maybe_refresh_option_chain",
            lambda *_args, **_kwargs: None,
        )

        selection = build_option_selection(
            session,
            context=_option_context(),
            signal=_option_signal(),
            settings=Settings(_env_file=None, upstox_access_token=""),
        )

        assert selection.chain_source == "synthetic_display_only"
        assert selection.chain_rows
        assert selection.signal["action"] == "HOLD"
        assert selection.signal["quote_status"] == "unavailable"
        assert any("display-only" in reason for reason in selection.signal["reasons"])
    finally:
        session.close()


def test_latest_option_premium_fails_closed_when_db_quote_is_stale() -> None:
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
            settings=Settings(_env_file=None, upstox_access_token="", option_chain_refresh_seconds=4),
        )

        assert price is None
    finally:
        session.close()


def test_latest_option_premium_rejects_fresh_synthetic_quote() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime.now(IST_ZONE).date() + timedelta(days=5)
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=datetime.now(IST_ZONE),
            source="synthetic",
        )
        session.commit()

        price = latest_option_premium(
            session,
            symbol="Nifty 50",
            expiry_date=expiry,
            strike=24100.0,
            option_type="CE",
            settings=Settings(_env_file=None, upstox_access_token=""),
        )

        assert price is None
    finally:
        session.close()


def test_latest_option_premium_prefers_fresh_real_quote_over_newer_synthetic() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime.now(IST_ZONE).date() + timedelta(days=5)
        now = datetime.now(IST_ZONE)
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=now - timedelta(seconds=1),
            ltp=101.0,
            source="upstox_option_chain",
        )
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=now,
            ltp=999.0,
            source="synthetic",
        )
        session.commit()

        price = latest_option_premium(
            session,
            symbol="Nifty 50",
            expiry_date=expiry,
            strike=24100.0,
            option_type="CE",
            settings=Settings(_env_file=None, upstox_access_token="", option_chain_refresh_seconds=4),
        )

        assert price == 101.0
    finally:
        session.close()


def test_latest_option_premium_rejects_non_positive_real_quote() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime.now(IST_ZONE).date() + timedelta(days=5)
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=datetime.now(IST_ZONE),
            ltp=0.0,
            bid=0.0,
            ask=0.0,
        )
        session.commit()

        price = latest_option_premium(
            session,
            symbol="Nifty 50",
            expiry_date=expiry,
            strike=24100.0,
            option_type="CE",
            settings=Settings(_env_file=None, upstox_access_token=""),
        )

        assert price is None
    finally:
        session.close()


def test_latest_option_premium_uses_older_positive_real_quote_over_newer_invalid_real() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime.now(IST_ZONE).date() + timedelta(days=5)
        now = datetime.now(IST_ZONE)
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=now - timedelta(seconds=1),
            ltp=101.0,
        )
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=now,
            ltp=0.0,
            bid=0.0,
            ask=0.0,
        )
        session.commit()

        price = latest_option_premium(
            session,
            symbol="Nifty 50",
            expiry_date=expiry,
            strike=24100.0,
            option_type="CE",
            settings=Settings(_env_file=None, upstox_access_token="", option_chain_refresh_seconds=4),
        )

        assert price == 101.0
    finally:
        session.close()


def test_latest_option_premium_rejects_crossed_real_quote() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime.now(IST_ZONE).date() + timedelta(days=5)
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=datetime.now(IST_ZONE),
            ltp=100.0,
            bid=101.0,
            ask=99.0,
        )
        session.commit()

        price = latest_option_premium(
            session,
            symbol="Nifty 50",
            expiry_date=expiry,
            strike=24100.0,
            option_type="CE",
            settings=Settings(_env_file=None, upstox_access_token=""),
        )

        assert price is None
    finally:
        session.close()


def test_same_minute_partial_snapshot_does_not_refresh_omitted_contract() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = date(2026, 6, 25)
        quote_ts = datetime(2026, 6, 18, 10, 0, 5, tzinfo=IST_ZONE)
        snapshot_ts = datetime(2026, 6, 18, 10, 0, 45, tzinfo=IST_ZONE)
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=quote_ts,
            ltp=101.0,
        )
        session.add(
            DataFreshness(
                source_name="upstox_option_chain:Nifty 50",
                last_success_at=snapshot_ts,
                status="ok",
                details={
                    "expiry_date": expiry.isoformat(),
                    "snapshot_ts": snapshot_ts.isoformat(),
                    "contracts": 1,
                },
            )
        )
        session.commit()

        quote = resolve_live_option_quote(
            session,
            symbol="Nifty 50",
            expiry_date=expiry,
            strike=24100.0,
            option_type="CE",
            settings=Settings(_env_file=None, upstox_access_token="", option_chain_refresh_seconds=4),
            now=snapshot_ts,
        )

        assert quote is None
    finally:
        session.close()


def test_option_chain_refresh_gate_ignores_newer_synthetic_rows(monkeypatch) -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime.now(IST_ZONE).date() + timedelta(days=5)
        now = datetime.now(IST_ZONE)
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=now - timedelta(minutes=5),
            source="upstox_option_chain",
        )
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=now,
            source="synthetic",
        )
        session.commit()
        calls: list[dict] = []

        class _Collector:
            def sync_option_chain(self, _db, **kwargs):
                calls.append(kwargs)

        monkeypatch.setattr(
            "backend.execution_engine.live_service.UpstoxOptionChainCollector",
            _Collector,
        )

        _maybe_refresh_option_chain(
            session,
            symbol="Nifty 50",
            underlying_key="NSE_INDEX|Nifty 50",
            expiry_date=expiry,
            settings=Settings(
                _env_file=None,
                upstox_access_token="token",
                option_chain_refresh_seconds=4,
            ),
        )

        assert len(calls) == 1
        assert calls[0]["expiry_date"] == expiry
    finally:
        session.close()


def test_option_liquidity_rejects_crossed_order_book() -> None:
    failures = _option_liquidity_failures(
        {
            "instrument_key": "NSE_FO|24100CE",
            "ltp": 100.0,
            "bid": 101.0,
            "ask": 99.0,
            "volume": 5000.0,
            "oi": 10000.0,
        },
        Settings(_env_file=None),
    )

    assert any("crossed" in failure.lower() for failure in failures)


def test_refresh_open_positions_snapshot_preserves_pnl_when_real_quote_is_unavailable() -> None:
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
                metadata_json={
                    "execution_mode": "sandbox",
                    "instrument_key": "NSE_FO|24100CE",
                },
        )
        session.add(position)
        session.commit()

        rows = refresh_open_positions_snapshot(
            session,
            settings=Settings(_env_file=None, upstox_access_token=""),
        )

        assert len(rows) == 1
        assert rows[0].current_premium == 100.0
        assert rows[0].unrealized_pnl == 0.0
        assert rows[0].metadata_json["latest_quote_status"] == "unavailable"
        assert rows[0].metadata_json["latest_quote_source"] is None
        assert "P&L was not changed" in rows[0].metadata_json["latest_quote_unavailable_reason"]
    finally:
        session.close()


def test_refresh_open_positions_snapshot_does_not_mark_non_positive_quote() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime.now(IST_ZONE).date() + timedelta(days=5)
        position = ExecutionPosition(
            trade_date=datetime.now(IST_ZONE).date(),
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
            peak_premium=110.0,
            tsl_active=False,
            take_profit=140.0,
            target_premium=140.0,
            current_price=110.0,
            current_premium=110.0,
            pnl_points=10.0,
            pnl_value=500.0,
            realized_pnl=0.0,
            unrealized_pnl=500.0,
                metadata_json={
                    "execution_mode": "sandbox",
                    "instrument_key": "NSE_FO|24100CE",
                },
        )
        session.add(position)
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=datetime.now(IST_ZONE),
            ltp=0.0,
            bid=0.0,
            ask=0.0,
        )
        session.commit()

        rows = refresh_open_positions_snapshot(
            session,
            settings=Settings(_env_file=None, upstox_access_token=""),
        )

        assert rows[0].current_premium == 110.0
        assert rows[0].unrealized_pnl == 500.0
        assert rows[0].metadata_json["latest_quote_status"] == "unavailable"
    finally:
        session.close()


def test_refresh_open_positions_snapshot_uses_fresh_real_quote() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        expiry = datetime.now(IST_ZONE).date() + timedelta(days=5)
        quote_received_at = datetime.now(IST_ZONE)
        position = ExecutionPosition(
            trade_date=datetime.now(IST_ZONE).date(),
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
                metadata_json={
                    "execution_mode": "sandbox",
                    "instrument_key": "NSE_FO|24100CE",
                },
        )
        session.add(position)
        _add_option_quote(
            session,
            expiry=expiry,
            strike=24100.0,
            ts=quote_received_at,
            ltp=112.0,
            bid=111.5,
            ask=112.5,
        )
        session.add(
            DataFreshness(
                source_name="upstox_option_chain:Nifty 50",
                last_success_at=quote_received_at,
                status="ok",
                details={
                    "expiry_date": expiry.isoformat(),
                    "snapshot_ts": quote_received_at.isoformat(),
                    "contracts": 1,
                },
            )
        )
        session.commit()

        rows = refresh_open_positions_snapshot(
            session,
            settings=Settings(_env_file=None, upstox_access_token="", option_chain_refresh_seconds=4),
        )

        assert len(rows) == 1
        assert rows[0].current_premium == 112.0
        assert rows[0].unrealized_pnl == 600.0
        assert rows[0].metadata_json["latest_quote_status"] == "available"
        assert rows[0].metadata_json["latest_quote_source"] == "upstox_option_chain"
        assert rows[0].metadata_json["latest_quote_age_seconds"] is not None
    finally:
        session.close()


def test_compute_sandbox_portfolio_metrics_derives_balance_from_open_and_closed_positions() -> None:
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
                metadata_json={"execution_mode": "sandbox", "latest_quote_status": "unavailable"},
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
                metadata_json={"execution_mode": "sandbox"},
            )
        )
        session.add(
            ExecutionPosition(
                trade_date=datetime(2026, 5, 2, tzinfo=IST_ZONE).date(),
                symbol="SENSEX",
                interval="1minute",
                strategy_name="archived",
                option_type="CE",
                side="BUY",
                expiry_date=datetime(2026, 5, 6, tzinfo=IST_ZONE).date(),
                strike=80000.0,
                quantity=20,
                status="CLOSED",
                entry_price=100.0,
                entry_premium=100.0,
                stop_loss=80.0,
                trailing_stop=0.0,
                exit_premium=200.0,
                realized_pnl=2000.0,
                metadata_json={"execution_mode": "paper"},
            )
        )
        session.commit()

        metrics = compute_sandbox_portfolio_metrics(session, settings=Settings(execution_capital=100000.0))

        assert metrics["starting_balance"] == 100000.0
        assert metrics["invested_amount"] == 5000.0
        assert metrics["realized_pnl"] == 500.0
        assert metrics["unrealized_pnl"] == 500.0
        assert metrics["unpriced_positions_count"] == 1
        assert metrics["unpriced_unrealized_pnl"] == 500.0
        assert metrics["priced_unrealized_pnl"] == 0.0
        assert metrics["available_balance"] == 95500.0
        assert metrics["equity"] == 101000.0
    finally:
        session.close()
