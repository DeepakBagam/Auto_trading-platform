from datetime import date, datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.db.models import Base
from backend.db.models import ExecutionOrder, ExecutionPosition, SignalLog
from backend.execution_engine.broker import BrokerOrderRequest, BrokerOrderResponse, UpstoxBroker
from backend.execution_engine.engine import IntradayOptionsExecutionEngine
from backend.execution_engine.live_service import TechnicalSignal
from backend.utils.app_state import set_runtime_trading_mode
from backend.utils.config import Settings
from backend.utils.constants import IST_ZONE


class RejectingLiveBroker:
    broker_name = "upstox"

    def get_portfolio(self) -> dict:
        return {
            "broker": "upstox",
            "funds": {},
            "positions": [],
            "errors": [{"source": "funds", "status_code": 401, "body": "invalid token"}],
            "status": "warn",
        }

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResponse:
        raise AssertionError("live order placement must not be reached when broker is not ready")

    def cancel_all_pending(self) -> BrokerOrderResponse:
        return BrokerOrderResponse(False, None, "UNUSED", "unused", {})

    def get_order_status(self, order_id: str) -> BrokerOrderResponse:
        return BrokerOrderResponse(False, order_id, "401", "invalid token", {})


class CapturingLiveBroker:
    broker_name = "upstox"

    def __init__(self) -> None:
        self.placed: list[BrokerOrderRequest] = []
        self.modified: list[tuple[str, float | None, float | None]] = []
        self.cancelled: list[str] = []
        self.place_responses: list[BrokerOrderResponse] = []
        self.cancel_response = BrokerOrderResponse(True, "LIVE-1", "CANCELLED", "ok", {"broker": "upstox"})
        self.order_status_calls: list[str] = []
        self.order_status_response = BrokerOrderResponse(
            True,
            "LIVE-1",
            "ACCEPTED",
            "pending",
            {"data": {"status": "open"}},
        )

    def get_portfolio(self) -> dict:
        return {"broker": "upstox", "funds": {"available_margin": 100000}, "positions": [], "errors": [], "status": "ok"}

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResponse:
        self.placed.append(request)
        if self.place_responses:
            return self.place_responses.pop(0)
        return BrokerOrderResponse(True, f"LIVE-{len(self.placed)}", "ACCEPTED", "ok", {"broker": "upstox"})

    def modify_order(
        self, order_id: str, *, trigger_price: float | None = None, price: float | None = None
    ) -> BrokerOrderResponse:
        self.modified.append((order_id, trigger_price, price))
        return BrokerOrderResponse(True, order_id, "MODIFIED", "ok", {"broker": "upstox"})

    def cancel_order(self, order_id: str) -> BrokerOrderResponse:
        self.cancelled.append(order_id)
        return BrokerOrderResponse(
            self.cancel_response.success,
            order_id,
            self.cancel_response.status,
            self.cancel_response.message,
            self.cancel_response.payload,
        )

    def cancel_all_pending(self) -> BrokerOrderResponse:
        return BrokerOrderResponse(True, None, "OK", "ok", {})

    def get_order_status(self, order_id: str) -> BrokerOrderResponse:
        self.order_status_calls.append(order_id)
        return BrokerOrderResponse(
            self.order_status_response.success,
            order_id,
            self.order_status_response.status,
            self.order_status_response.message,
            self.order_status_response.payload,
        )


class CapturingSandboxBroker(CapturingLiveBroker):
    broker_name = "upstox_sandbox"

    def get_portfolio(self) -> dict:
        return {"broker": self.broker_name, "funds": {}, "positions": [], "errors": [], "status": "sandbox"}

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResponse:
        self.placed.append(request)
        if self.place_responses:
            return self.place_responses.pop(0)
        return BrokerOrderResponse(
            True,
            f"SANDBOX-{len(self.placed)}",
            "ACCEPTED",
            "ok",
            {"broker": self.broker_name},
        )

    def modify_order(
        self,
        order_id: str,
        *,
        trigger_price: float | None = None,
        price: float | None = None,
        quantity: int | None = None,
        order_type: str | None = None,
    ) -> BrokerOrderResponse:
        self.modified.append((order_id, trigger_price, price))
        self.modified_quantity = quantity
        self.modified_order_type = order_type
        return BrokerOrderResponse(True, order_id, "MODIFIED", "ok", {"broker": self.broker_name})


class _FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self.ok = True
        self.status_code = 200
        self._payload = payload
        self.content = b"{}"

    def json(self) -> dict:
        return self._payload


class _VerbSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def get(self, url: str, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return _FakeHttpResponse(
            {
                "status": "success",
                "data": [
                    {
                        "order_id": "OID-1",
                        "status": "open pending",
                    }
                ],
            }
        )

    def put(self, url: str, **kwargs):
        self.calls.append(("PUT", url, kwargs))
        return _FakeHttpResponse({"status": "success", "data": {"order_id": "OID-1"}})

    def delete(self, url: str, **kwargs):
        self.calls.append(("DELETE", url, kwargs))
        return _FakeHttpResponse({"status": "success", "data": {"order_id": "OID-1"}})


def _memory_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return Session(engine)


def _live_settings() -> Settings:
    return Settings(
        execution_enabled=True,
        execution_mode="live",
        execution_symbols="Nifty 50",
        upstox_access_token="token",
        execution_accept_external_webhook=False,
    )


def _sandbox_settings() -> Settings:
    return Settings(
        _env_file=None,
        execution_enabled=True,
        execution_mode="sandbox",
        execution_symbols="Nifty 50",
        upstox_sandbox_access_token="sandbox-token",
    )


def _open_position() -> ExecutionPosition:
    return ExecutionPosition(
        trade_date=date(2026, 6, 9),
        symbol="Nifty 50",
        interval="1minute",
        strategy_name="test",
        option_type="CE",
        side="BUY",
        expiry_date=date(2026, 6, 16),
        strike=24150,
        quantity=75,
        status="OPEN",
        entry_price=200,
        entry_premium=200,
        stop_loss=150,
        initial_sl=150,
        current_sl=150,
        trailing_stop=150,
        peak_premium=200,
        tsl_active=False,
        take_profit=320,
        target_premium=320,
        current_price=200,
        current_premium=200,
        opened_at=datetime(2026, 6, 9, 10, 0, tzinfo=IST_ZONE),
        metadata_json={"instrument_key": "NSE_FO|12345", "execution_mode": "live"},
    )


def _dated_position(
    *,
    trade_date: date,
    opened_at: datetime,
    status: str,
    mode: str,
    symbol: str = "Nifty 50",
) -> ExecutionPosition:
    position = _open_position()
    position.trade_date = trade_date
    position.opened_at = opened_at
    position.status = status
    position.symbol = symbol
    position.metadata_json = {**(position.metadata_json or {}), "execution_mode": mode}
    if status == "CLOSED":
        position.closed_at = opened_at
    return position


def test_upstox_modify_uses_put_and_cancel_uses_delete(monkeypatch) -> None:
    broker = UpstoxBroker(base_url="https://api.example.test", access_token="token")
    session = _VerbSession()
    broker.session = session
    monkeypatch.setattr(broker, "_refresh_token_if_available", lambda: None)

    modify = broker.modify_order("OID-1", trigger_price=180.0)
    cancel = broker.cancel_order("OID-1")

    assert modify.success is True
    assert cancel.success is True
    assert session.calls[0][0] == "PUT"
    assert session.calls[0][2]["json"]["order_id"] == "OID-1"
    assert session.calls[1][0] == "DELETE"
    assert session.calls[1][2]["params"] == {"order_id": "OID-1"}


def test_upstox_read_endpoints_support_order_lists_and_report_params(monkeypatch) -> None:
    broker = UpstoxBroker(base_url="https://api.example.test", access_token="token")
    session = _VerbSession()
    broker.session = session
    monkeypatch.setattr(broker, "_refresh_token_if_available", lambda: None)

    order_book = broker.get_order_book()
    history = broker.get_order_history("OID-1")
    pnl = broker.get_trade_pnl_report(
        segment="FO",
        financial_year="2627",
        from_date="23-06-2026",
        to_date="23-06-2026",
    )

    assert order_book["data"][0]["order_id"] == "OID-1"
    assert session.calls[0][1].endswith("/v2/order/retrieve-all")
    assert session.calls[1][1].endswith("/v2/order/history")
    assert session.calls[1][2]["params"] == {"order_id": "OID-1"}
    assert session.calls[2][1].endswith("/v2/trade/profit-loss/data")
    assert session.calls[2][2]["params"]["financial_year"] == "2627"
    assert pnl["status"] == "success"
    assert history["status"] == "success"


def test_sandbox_protected_prices_round_to_configured_tick() -> None:
    service = IntradayOptionsExecutionEngine(
        settings=_sandbox_settings(),
        broker=CapturingSandboxBroker(),
    )

    assert service._sandbox_protected_price(100.0, "BUY") == 101.0
    assert service._sandbox_protected_price(100.0, "SELL") == 99.0
    rounded = service._sandbox_protected_price(123.47, "BUY")
    assert round(rounded / 0.05) * 0.05 == rounded


def test_india_vix_is_never_evaluated_as_an_option_underlying() -> None:
    session = _memory_session()
    service = IntradayOptionsExecutionEngine(
        settings=_sandbox_settings(),
        broker=CapturingSandboxBroker(),
    )
    try:
        assert service._evaluate_symbol(
            session,
            datetime(2026, 6, 22, 10, 0, tzinfo=IST_ZONE),
            "India VIX",
        ) == "skip:unsupported_option_underlying"
    finally:
        session.close()


def test_sandbox_broker_rebuilds_when_runtime_token_changes(monkeypatch) -> None:
    session = _memory_session()
    monkeypatch.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", "old-sandbox-token")
    original = CapturingSandboxBroker()
    replacement = CapturingSandboxBroker()
    service = IntradayOptionsExecutionEngine(settings=_sandbox_settings(), broker=original)
    monkeypatch.setattr(service, "_build_broker", lambda: replacement)
    try:
        monkeypatch.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", "new-sandbox-token")

        assert service._sync_runtime_mode(session) == "sandbox"
        assert service.broker is replacement
    finally:
        session.close()


def test_sandbox_protective_stop_place_modify_and_cancel() -> None:
    session = _memory_session()
    broker = CapturingSandboxBroker()
    service = IntradayOptionsExecutionEngine(settings=_sandbox_settings(), broker=broker)
    position = _open_position()
    position.metadata_json = {**position.metadata_json, "execution_mode": "sandbox"}
    try:
        session.add(position)
        session.flush()

        order = service._place_live_protective_stop(session, position=position)
        assert order is not None
        assert broker.placed[0].order_type == "SL"
        assert broker.placed[0].trigger_price == 150.0
        assert broker.placed[0].price == 148.5

        service._modify_live_protective_stop(session, position=position, trigger_price=180.0)
        assert broker.modified[-1] == ("SANDBOX-1", 180.0, 178.15)
        assert broker.modified_quantity == 75
        assert broker.modified_order_type == "SL"

        cancelled = service._cancel_live_protective_stop(
            session,
            position=position,
            reason="TP_HIT",
            now=datetime(2026, 6, 9, 11, 0, tzinfo=IST_ZONE),
            exit_premium=250.0,
        )
        assert cancelled is True
        assert broker.cancelled == ["SANDBOX-1"]
        assert position.metadata_json["broker_sl_active"] is False
    finally:
        session.close()


def test_sandbox_exit_cancels_stop_and_submits_protected_limit() -> None:
    session = _memory_session()
    broker = CapturingSandboxBroker()
    service = IntradayOptionsExecutionEngine(settings=_sandbox_settings(), broker=broker)
    position = _open_position()
    position.metadata_json = {
        **position.metadata_json,
        "execution_mode": "sandbox",
        "broker_sl_order_id": "SANDBOX-SL",
        "broker_sl_active": True,
        "broker_sl_trigger_price": 150.0,
    }
    try:
        session.add(position)
        session.commit()

        order = service._close_position(
            session,
            position=position,
            now=datetime(2026, 6, 9, 11, 0, tzinfo=IST_ZONE),
            reason="TP_HIT",
            exit_premium=250.0,
        )

        assert order is not None
        assert broker.cancelled == ["SANDBOX-SL"]
        assert broker.placed[-1].side == "SELL"
        assert broker.placed[-1].order_type == "LIMIT"
        assert broker.placed[-1].price == 247.5
        assert position.status == "CLOSED"
        assert position.exit_premium == 250.0
    finally:
        session.close()


def test_live_run_once_blocks_when_broker_is_not_ready() -> None:
    session = _memory_session()
    settings = _live_settings()
    service = IntradayOptionsExecutionEngine(settings=settings, broker=RejectingLiveBroker())
    try:
        set_runtime_trading_mode(session, "live")
        session.commit()

        result = service.run_once(session, now=datetime(2026, 6, 5, 10, 0, tzinfo=IST_ZONE))

        assert result["status"] == "live_broker_not_ready"
        assert result["mode"] == "live"
        assert "invalid token" in result["reason"]
    finally:
        session.close()


def test_signal_candle_claim_blocks_duplicate_workers() -> None:
    session = _memory_session()
    signal = TechnicalSignal(
        symbol="Nifty 50",
        interval="1minute",
        timestamp=datetime(2026, 6, 18, 10, 40, tzinfo=IST_ZONE),
        action="SELL",
        bias="SELL",
        score=100.0,
        confidence=0.95,
        conviction="high",
        entry_price=24070.0,
        stop_loss=24090.0,
        take_profit=24040.0,
        cooldown_seconds=0,
        max_signals_reached=False,
        reasons=["test"],
        details={"strategy_name": "test"},
    )
    first_worker = IntradayOptionsExecutionEngine(settings=Settings())
    second_worker = IntradayOptionsExecutionEngine(settings=Settings())
    try:
        assert first_worker._claim_signal_candle(session, signal=signal) is True
        assert second_worker._claim_signal_candle(session, signal=signal) is False
    finally:
        session.close()


def test_daily_entry_positions_count_open_and_closed_for_active_mode() -> None:
    session = _memory_session()
    service = IntradayOptionsExecutionEngine(settings=Settings(execution_mode="sandbox"))
    trade_date = date(2026, 6, 18)
    try:
        session.add_all(
            [
                _dated_position(
                    trade_date=trade_date,
                    opened_at=datetime(2026, 6, 18, 9, 30, tzinfo=IST_ZONE),
                    status="OPEN",
                    mode="sandbox",
                ),
                _dated_position(
                    trade_date=trade_date,
                    opened_at=datetime(2026, 6, 18, 10, 0, tzinfo=IST_ZONE),
                    status="CLOSED",
                    mode="sandbox",
                ),
                _dated_position(
                    trade_date=trade_date,
                    opened_at=datetime(2026, 6, 18, 10, 30, tzinfo=IST_ZONE),
                    status="CLOSED",
                    mode="live",
                ),
            ]
        )
        session.commit()

        rows = service._entry_positions_for_day(session, trade_date=trade_date)

        assert len(rows) == 2
        assert {row.status for row in rows} == {"OPEN", "CLOSED"}
    finally:
        session.close()


def test_position_and_pnl_queries_are_isolated_by_execution_mode() -> None:
    session = _memory_session()
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=CapturingLiveBroker())
    trade_date = date(2026, 6, 18)
    live_open = _dated_position(
        trade_date=trade_date,
        opened_at=datetime(2026, 6, 18, 9, 30, tzinfo=IST_ZONE),
        status="OPEN",
        mode="live",
    )
    live_closed = _dated_position(
        trade_date=trade_date,
        opened_at=datetime(2026, 6, 18, 10, 0, tzinfo=IST_ZONE),
        status="CLOSED",
        mode="live",
    )
    live_closed.realized_pnl = -500.0
    sandbox_open = _dated_position(
        trade_date=trade_date,
        opened_at=datetime(2026, 6, 18, 10, 30, tzinfo=IST_ZONE),
        status="OPEN",
        mode="sandbox",
    )
    sandbox_closed = _dated_position(
        trade_date=trade_date,
        opened_at=datetime(2026, 6, 18, 11, 0, tzinfo=IST_ZONE),
        status="CLOSED",
        mode="sandbox",
    )
    sandbox_closed.realized_pnl = -9000.0
    try:
        session.add_all([live_open, live_closed, sandbox_open, sandbox_closed])
        session.commit()

        assert service._open_positions(session) == [live_open]
        assert service._daily_realized_pnl(session, trade_date) == -500.0
        service._refresh_daily_summary(session, trade_date)
        session.flush()
        summary = service.daily_report(session, trade_date)
        assert summary["total_trades"] == 1
        assert summary["total_profit"] == -500.0
    finally:
        session.close()


def test_live_reconciliation_ignores_sandbox_pending_positions() -> None:
    session = _memory_session()
    broker = CapturingLiveBroker()
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=broker)
    now = datetime(2026, 6, 18, 15, 0, tzinfo=IST_ZONE)
    sandbox_entry = _dated_position(
        trade_date=now.date(),
        opened_at=now,
        status="ENTRY_PENDING",
        mode="sandbox",
    )
    sandbox_entry.entry_order_id = "SANDBOX-ENTRY"
    sandbox_exit = _dated_position(
        trade_date=now.date(),
        opened_at=now,
        status="EXIT_PENDING",
        mode="sandbox",
    )
    sandbox_exit.metadata_json = {
        **sandbox_exit.metadata_json,
        "exit_order_id": "SANDBOX-EXIT",
    }
    try:
        session.add_all([sandbox_entry, sandbox_exit])
        session.commit()

        entry_result = service._reconcile_pending_entries(session, now)
        exit_result = service._reconcile_pending_exits(session, now)

        assert entry_result["pending_entry_reconciliations"] == 0
        assert exit_result["pending_exit_reconciliations"] == 0
        assert broker.order_status_calls == []
    finally:
        session.close()


def test_signal_cap_and_cooldown_ignore_rejected_signal_attempts() -> None:
    session = _memory_session()
    service = IntradayOptionsExecutionEngine(
        settings=Settings(execution_mode="sandbox", signal_cooldown_minutes=12)
    )
    now = datetime(2026, 6, 18, 10, 20, tzinfo=IST_ZONE)
    try:
        session.add(
            SignalLog(
                timestamp=datetime(2026, 6, 18, 10, 19, tzinfo=IST_ZONE),
                trade_date=now.date(),
                symbol="Nifty 50",
                interval="1minute",
                ml_signal="SELL",
                pine_signal="SELL",
                consensus="SELL",
                trade_placed=False,
                skip_reason="No liquid option contract.",
            )
        )
        session.commit()

        count, cooldown = service._successful_trade_guard(session, now=now, symbol="Nifty 50")

        assert count == 0
        assert cooldown == 0
    finally:
        session.close()


def test_signal_cap_and_cooldown_use_successful_position_entries() -> None:
    session = _memory_session()
    service = IntradayOptionsExecutionEngine(
        settings=Settings(execution_mode="sandbox", signal_cooldown_minutes=12)
    )
    now = datetime(2026, 6, 18, 10, 10, tzinfo=IST_ZONE)
    try:
        session.add(
            _dated_position(
                trade_date=now.date(),
                opened_at=datetime(2026, 6, 18, 10, 5, tzinfo=IST_ZONE),
                status="CLOSED",
                mode="sandbox",
            )
        )
        session.commit()

        count, cooldown = service._successful_trade_guard(session, now=now, symbol="Nifty 50")

        assert count == 1
        assert cooldown == 7 * 60
    finally:
        session.close()


def test_live_sizing_balance_uses_broker_available_margin() -> None:
    session = _memory_session()
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=CapturingLiveBroker())
    try:
        balance, source, metadata = service._available_trading_balance(session)

        assert balance == 100000.0
        assert source == "broker_available_margin"
        assert metadata["broker"] == "upstox"
    finally:
        session.close()


def test_live_sizing_balance_fails_closed_without_broker_margin() -> None:
    session = _memory_session()
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=RejectingLiveBroker())
    try:
        balance, source, metadata = service._available_trading_balance(session)

        assert balance == 0.0
        assert source == "broker_margin_unavailable"
        assert metadata["reason"] == "broker_available_margin_unavailable"
    finally:
        session.close()


def test_partial_live_entry_stays_pending_and_cancels_remaining_quantity() -> None:
    session = _memory_session()
    broker = CapturingLiveBroker()
    broker.order_status_response = BrokerOrderResponse(
        True,
        "ENTRY-1",
        "ACCEPTED",
        "ok",
        {
            "data": {
                "status": "open",
                "filled_quantity": 25,
                "pending_quantity": 50,
                "average_price": 201.0,
            }
        },
    )
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=broker)
    now = datetime(2026, 6, 18, 10, 5, tzinfo=IST_ZONE)
    position = _dated_position(
        trade_date=now.date(),
        opened_at=now,
        status="ENTRY_PENDING",
        mode="live",
    )
    position.entry_order_id = "ENTRY-1"
    position.metadata_json = {
        **(position.metadata_json or {}),
        "entry_order_id": "ENTRY-1",
        "entry_requested_quantity": 75,
    }
    try:
        session.add(position)
        session.flush()
        session.add(
            ExecutionOrder(
                position_id=position.id,
                trade_date=now.date(),
                symbol=position.symbol,
                order_kind="ENTRY",
                side="BUY",
                quantity=75,
                status="ACCEPTED",
                broker_name="upstox",
                broker_order_id="ENTRY-1",
                response_json={},
            )
        )
        session.commit()

        result = service._reconcile_pending_entries(session, now)

        session.refresh(position)
        assert position.status == "ENTRY_PENDING"
        assert position.quantity == 75
        assert position.metadata_json["entry_filled_quantity"] == 25
        assert broker.cancelled == ["ENTRY-1"]
        assert broker.placed == []
        assert result["pending_entry_reconciliations"] == 1
    finally:
        session.close()


def test_cancelled_partial_entry_opens_only_filled_quantity_with_protection() -> None:
    session = _memory_session()
    broker = CapturingLiveBroker()
    broker.order_status_response = BrokerOrderResponse(
        True,
        "ENTRY-1",
        "ACCEPTED",
        "ok",
        {
            "data": {
                "status": "cancelled",
                "filled_quantity": 25,
                "pending_quantity": 0,
                "average_price": 201.0,
            }
        },
    )
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=broker)
    now = datetime(2026, 6, 18, 10, 6, tzinfo=IST_ZONE)
    position = _dated_position(
        trade_date=now.date(),
        opened_at=now,
        status="ENTRY_PENDING",
        mode="live",
    )
    position.entry_order_id = "ENTRY-1"
    position.metadata_json = {
        **(position.metadata_json or {}),
        "entry_order_id": "ENTRY-1",
        "entry_requested_quantity": 75,
    }
    try:
        session.add(position)
        session.flush()
        session.add(
            ExecutionOrder(
                position_id=position.id,
                trade_date=now.date(),
                symbol=position.symbol,
                order_kind="ENTRY",
                side="BUY",
                quantity=75,
                status="ACCEPTED",
                broker_name="upstox",
                broker_order_id="ENTRY-1",
                response_json={},
            )
        )
        session.commit()

        result = service._reconcile_pending_entries(session, now)

        session.refresh(position)
        assert position.status == "OPEN"
        assert position.quantity == 25
        assert position.entry_premium == 201.0
        assert len(broker.placed) == 1
        assert broker.placed[0].side == "SELL"
        assert broker.placed[0].order_type == "SL-M"
        assert broker.placed[0].qty == 25
        assert result["reconciled_entries"] == 1
    finally:
        session.close()


def test_force_squareoff_defers_sandbox_exit_without_real_quote(monkeypatch) -> None:
    session = _memory_session()
    service = IntradayOptionsExecutionEngine(
        settings=Settings(execution_mode="sandbox"),
    )
    now = datetime(2026, 6, 18, 15, 15, tzinfo=IST_ZONE)
    position = _dated_position(
        trade_date=now.date(),
        opened_at=datetime(2026, 6, 18, 10, 0, tzinfo=IST_ZONE),
        status="OPEN",
        mode="sandbox",
    )
    try:
        session.add(position)
        session.commit()
        monkeypatch.setattr("backend.execution_engine.engine.latest_option_premium", lambda *args, **kwargs: None)

        result = service._force_square_off(session, now, "FORCE_SQUAREOFF")

        session.refresh(position)
        assert position.status == "OPEN"
        assert position.exit_premium is None
        assert position.metadata_json["exit_deferred"] is True
        assert result["square_off_closed"] == 0
        assert result["square_off_deferred"] == 1
        assert session.query(ExecutionOrder).count() == 0
    finally:
        session.close()


def test_force_squareoff_marks_live_exit_for_reconciliation_without_fill(monkeypatch) -> None:
    session = _memory_session()
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=CapturingLiveBroker())
    now = datetime(2026, 6, 18, 15, 15, tzinfo=IST_ZONE)
    position = _dated_position(
        trade_date=now.date(),
        opened_at=datetime(2026, 6, 18, 10, 0, tzinfo=IST_ZONE),
        status="OPEN",
        mode="live",
    )
    try:
        session.add(position)
        session.commit()
        monkeypatch.setattr("backend.execution_engine.engine.latest_option_premium", lambda *args, **kwargs: None)

        result = service._force_square_off(session, now, "FORCE_SQUAREOFF")

        session.refresh(position)
        order = session.query(ExecutionOrder).one()
        assert position.status == "EXIT_PENDING"
        assert position.exit_premium is None
        assert position.realized_pnl is None
        assert position.metadata_json["reconciliation_needed"] is True
        assert order.exit_premium is None
        assert result["square_off_reconciliation_pending"] == 1
    finally:
        session.close()


def test_pending_live_exit_reconciles_confirmed_fill() -> None:
    session = _memory_session()
    broker = CapturingLiveBroker()
    broker.order_status_response = BrokerOrderResponse(
        True,
        "LIVE-EXIT",
        "ACCEPTED",
        "ok",
        {"data": {"status": "complete", "average_price": 220.0}},
    )
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=broker)
    now = datetime(2026, 6, 18, 15, 16, tzinfo=IST_ZONE)
    position = _dated_position(
        trade_date=now.date(),
        opened_at=datetime(2026, 6, 18, 10, 0, tzinfo=IST_ZONE),
        status="EXIT_PENDING",
        mode="live",
    )
    position.metadata_json = {
        **(position.metadata_json or {}),
        "exit_order_id": "LIVE-EXIT",
        "reconciliation_needed": True,
    }
    try:
        session.add(position)
        session.flush()
        session.add(
            ExecutionOrder(
                position_id=position.id,
                trade_date=now.date(),
                symbol=position.symbol,
                order_kind="EXIT",
                side="SELL",
                quantity=position.quantity,
                status="ACCEPTED",
                broker_name="upstox",
                broker_order_id="LIVE-EXIT",
                response_json={},
            )
        )
        session.commit()

        result = service._reconcile_pending_exits(session, now)

        session.refresh(position)
        order = session.query(ExecutionOrder).one()
        assert position.status == "CLOSED"
        assert position.exit_premium == 220.0
        assert position.realized_pnl == 1500.0
        assert order.status == "COMPLETE"
        assert order.exit_premium == 220.0
        assert result["reconciled_exits"] == 1
    finally:
        session.close()


def test_pending_live_exit_reopens_after_broker_rejection() -> None:
    session = _memory_session()
    broker = CapturingLiveBroker()
    broker.order_status_response = BrokerOrderResponse(
        True,
        "LIVE-EXIT",
        "ACCEPTED",
        "ok",
        {"data": {"status": "rejected"}},
    )
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=broker)
    now = datetime(2026, 6, 18, 15, 16, tzinfo=IST_ZONE)
    position = _dated_position(
        trade_date=now.date(),
        opened_at=datetime(2026, 6, 18, 10, 0, tzinfo=IST_ZONE),
        status="EXIT_PENDING",
        mode="live",
    )
    position.metadata_json = {
        **(position.metadata_json or {}),
        "exit_order_id": "LIVE-EXIT",
        "reconciliation_needed": True,
        "broker_sl_active": False,
    }
    try:
        session.add(position)
        session.flush()
        session.add(
            ExecutionOrder(
                position_id=position.id,
                trade_date=now.date(),
                symbol=position.symbol,
                order_kind="EXIT",
                side="SELL",
                quantity=position.quantity,
                status="ACCEPTED",
                broker_name="upstox",
                broker_order_id="LIVE-EXIT",
                response_json={},
            )
        )
        session.commit()

        result = service._reconcile_pending_exits(session, now)

        session.refresh(position)
        assert position.status == "OPEN"
        assert position.metadata_json["exit_order_failed"] is True
        assert position.metadata_json["broker_sl_active"] is True
        assert len(broker.placed) == 1
        assert broker.placed[0].order_type == "SL-M"
        assert result["reopened_exits"] == 1
    finally:
        session.close()


def test_partial_cancelled_exit_reopens_only_residual_quantity() -> None:
    session = _memory_session()
    broker = CapturingLiveBroker()
    broker.order_status_response = BrokerOrderResponse(
        True,
        "LIVE-EXIT",
        "ACCEPTED",
        "ok",
        {
            "data": {
                "status": "cancelled",
                "filled_quantity": 25,
                "pending_quantity": 0,
                "average_price": 220.0,
            }
        },
    )
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=broker)
    now = datetime(2026, 6, 18, 15, 16, tzinfo=IST_ZONE)
    position = _dated_position(
        trade_date=now.date(),
        opened_at=datetime(2026, 6, 18, 10, 0, tzinfo=IST_ZONE),
        status="EXIT_PENDING",
        mode="live",
    )
    position.metadata_json = {
        **(position.metadata_json or {}),
        "entry_filled_quantity": 75,
        "exit_order_id": "LIVE-EXIT",
        "exit_requested_quantity": 75,
        "exit_base_realized_pnl": 0.0,
        "reconciliation_needed": True,
        "broker_sl_active": False,
    }
    try:
        session.add(position)
        session.flush()
        session.add(
            ExecutionOrder(
                position_id=position.id,
                trade_date=now.date(),
                symbol=position.symbol,
                order_kind="EXIT",
                side="SELL",
                quantity=75,
                status="ACCEPTED",
                broker_name="upstox",
                broker_order_id="LIVE-EXIT",
                response_json={},
            )
        )
        session.commit()

        result = service._reconcile_pending_exits(session, now)

        session.refresh(position)
        assert position.status == "OPEN"
        assert position.quantity == 50
        assert position.realized_pnl == 500.0
        assert len(broker.placed) == 1
        assert broker.placed[0].order_type == "SL-M"
        assert broker.placed[0].qty == 50
        assert result["reopened_exits"] == 1
    finally:
        session.close()


def test_market_exit_is_blocked_until_protective_stop_cancel_is_confirmed() -> None:
    session = _memory_session()
    broker = CapturingLiveBroker()
    broker.cancel_response = BrokerOrderResponse(False, "SL-1", "500", "cancel failed", {})
    broker.order_status_response = BrokerOrderResponse(
        True,
        "SL-1",
        "ACCEPTED",
        "ok",
        {"data": {"status": "trigger pending", "filled_quantity": 0}},
    )
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=broker)
    now = datetime(2026, 6, 18, 15, 0, tzinfo=IST_ZONE)
    position = _open_position()
    position.metadata_json = {
        **position.metadata_json,
        "broker_sl_order_id": "SL-1",
        "broker_sl_active": True,
    }
    try:
        session.add(position)
        session.commit()

        order = service._close_position(
            session,
            position=position,
            now=now,
            reason="MANUAL",
            exit_premium=210.0,
        )
        session.commit()

        session.refresh(position)
        assert order is None
        assert position.status == "EXIT_PENDING"
        assert position.metadata_json["exit_reconciliation_source"] == "protective_stop_cancel"
        assert broker.cancelled == ["SL-1"]
        assert broker.placed == []
    finally:
        session.close()


def test_failed_market_exit_restores_protective_stop_after_confirmed_cancel() -> None:
    session = _memory_session()
    broker = CapturingLiveBroker()
    broker.order_status_response = BrokerOrderResponse(
        True,
        "SL-1",
        "ACCEPTED",
        "ok",
        {"data": {"status": "cancelled", "filled_quantity": 0}},
    )
    broker.place_responses = [
        BrokerOrderResponse(False, None, "400", "exit rejected", {}),
        BrokerOrderResponse(True, "SL-2", "ACCEPTED", "ok", {}),
    ]
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=broker)
    now = datetime(2026, 6, 18, 15, 0, tzinfo=IST_ZONE)
    position = _open_position()
    position.metadata_json = {
        **position.metadata_json,
        "broker_sl_order_id": "SL-1",
        "broker_sl_active": True,
    }
    try:
        session.add(position)
        session.commit()

        service._close_position(
            session,
            position=position,
            now=now,
            reason="MANUAL",
            exit_premium=210.0,
        )
        session.commit()

        session.refresh(position)
        assert [request.order_type for request in broker.placed] == ["MARKET", "SL-M"]
        assert position.status == "OPEN"
        assert position.metadata_json["broker_sl_active"] is True
        assert position.metadata_json["broker_sl_order_id"] == "SL-2"
    finally:
        session.close()


def test_exit_claim_allows_only_one_worker() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    writer = Session(engine)
    first = Session(engine)
    second = Session(engine)
    now = datetime(2026, 6, 18, 15, 0, tzinfo=IST_ZONE)
    try:
        position = _open_position()
        writer.add(position)
        writer.commit()
        first_position = first.get(ExecutionPosition, position.id)
        second_position = second.get(ExecutionPosition, position.id)
        first_service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=CapturingLiveBroker())
        second_service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=CapturingLiveBroker())

        assert first_service._claim_exit(first, first_position, now, "MANUAL") is True
        assert second_service._claim_exit(second, second_position, now, "MANUAL") is False
    finally:
        first.close()
        second.close()
        writer.close()


def test_fresh_exit_submitting_state_is_not_reopened_without_order_identity() -> None:
    session = _memory_session()
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=CapturingLiveBroker())
    now = datetime(2026, 6, 18, 15, 0, tzinfo=IST_ZONE)
    position = _dated_position(
        trade_date=now.date(),
        opened_at=now,
        status="EXIT_SUBMITTING",
        mode="live",
    )
    try:
        session.add(position)
        session.commit()

        result = service._reconcile_pending_exits(session, now)

        session.refresh(position)
        assert position.status == "EXIT_SUBMITTING"
        assert position.metadata_json["reconciliation_last_error"] == "missing_exit_order_id"
        assert result["pending_exit_reconciliations"] == 1
    finally:
        session.close()


def test_live_order_placement_does_not_retry_ambiguous_failure() -> None:
    session = _memory_session()
    broker = CapturingLiveBroker()
    broker.place_responses = [
        BrokerOrderResponse(False, None, "AMBIGUOUS", "timeout", {}),
        BrokerOrderResponse(True, "DUPLICATE", "ACCEPTED", "ok", {}),
    ]
    service = IntradayOptionsExecutionEngine(
        settings=_live_settings().model_copy(update={"order_retry_attempts": 3}),
        broker=broker,
    )
    request = BrokerOrderRequest(
        instrument_key="NSE_FO|12345",
        option_type="CE",
        strike=24150,
        expiry_date="2026-06-16",
        side="BUY",
        qty=75,
    )
    try:
        response = service._place_order_with_retry(
            session,
            request=request,
            action="test",
            resource_id="test",
        )

        assert response.status == "AMBIGUOUS"
        assert len(broker.placed) == 1
    finally:
        session.close()


def test_live_entry_places_broker_side_protective_sl() -> None:
    session = _memory_session()
    broker = CapturingLiveBroker()
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=broker)
    try:
        position = _open_position()
        session.add(position)
        session.flush()

        service._place_live_protective_stop(session, position=position)

        assert len(broker.placed) == 1
        request = broker.placed[0]
        assert request.side == "SELL"
        assert request.order_type == "SL-M"
        assert request.trigger_price == 150
        assert position.metadata_json["broker_sl_order_id"] == "LIVE-1"
        assert position.metadata_json["broker_sl_active"] is True
    finally:
        session.close()


def test_live_tsl_update_modifies_broker_side_stop() -> None:
    session = _memory_session()
    broker = CapturingLiveBroker()
    service = IntradayOptionsExecutionEngine(settings=_live_settings(), broker=broker)
    try:
        position = _open_position()
        position.metadata_json = {
            **position.metadata_json,
            "broker_sl_order_id": "LIVE-1",
            "broker_sl_active": True,
            "broker_sl_trigger_price": 150,
        }
        session.add(position)
        session.flush()

        service._modify_live_protective_stop(session, position=position, trigger_price=180)

        assert broker.modified == [("LIVE-1", 180, None)]
        assert position.metadata_json["broker_sl_trigger_price"] == 180
        assert position.metadata_json["broker_sl_active"] is True
    finally:
        session.close()
