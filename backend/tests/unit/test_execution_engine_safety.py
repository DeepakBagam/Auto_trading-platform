from datetime import date, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.db.models import Base
from backend.db.models import ExecutionPosition
from backend.execution_engine.broker import BrokerOrderRequest, BrokerOrderResponse
from backend.execution_engine.engine import IntradayOptionsExecutionEngine
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


class CapturingLiveBroker:
    broker_name = "upstox"

    def __init__(self) -> None:
        self.placed: list[BrokerOrderRequest] = []
        self.modified: list[tuple[str, float | None, float | None]] = []
        self.cancelled: list[str] = []

    def get_portfolio(self) -> dict:
        return {"broker": "upstox", "funds": {"available_margin": 100000}, "positions": [], "errors": [], "status": "ok"}

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResponse:
        self.placed.append(request)
        return BrokerOrderResponse(True, f"LIVE-{len(self.placed)}", "ACCEPTED", "ok", {"broker": "upstox"})

    def modify_order(
        self, order_id: str, *, trigger_price: float | None = None, price: float | None = None
    ) -> BrokerOrderResponse:
        self.modified.append((order_id, trigger_price, price))
        return BrokerOrderResponse(True, order_id, "MODIFIED", "ok", {"broker": "upstox"})

    def cancel_order(self, order_id: str) -> BrokerOrderResponse:
        self.cancelled.append(order_id)
        return BrokerOrderResponse(True, order_id, "CANCELLED", "ok", {"broker": "upstox"})

    def cancel_all_pending(self) -> BrokerOrderResponse:
        return BrokerOrderResponse(True, None, "OK", "ok", {})


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
