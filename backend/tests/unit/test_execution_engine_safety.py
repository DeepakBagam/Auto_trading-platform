from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.db.models import Base
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


def test_live_run_once_blocks_when_broker_is_not_ready() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    settings = Settings(
        execution_enabled=True,
        execution_mode="live",
        execution_symbols="Nifty 50",
        upstox_access_token="token",
        execution_accept_external_webhook=False,
    )
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
