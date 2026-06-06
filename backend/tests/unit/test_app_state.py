from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.db.models import Base, ExecutionPosition
from backend.execution_engine.live_service import compute_paper_portfolio_metrics
from backend.utils.app_state import get_runtime_trading_mode, reset_paper_account, set_runtime_trading_mode
from backend.utils.config import Settings
from backend.utils.constants import IST_ZONE


def test_runtime_trading_mode_persists_in_app_settings() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        assert get_runtime_trading_mode(session, settings=Settings(execution_mode="paper")) == "paper"
        set_runtime_trading_mode(session, "live")
        session.commit()
        assert get_runtime_trading_mode(session, settings=Settings(execution_mode="paper")) == "live"
    finally:
        session.close()


def test_reset_paper_account_excludes_positions_before_reset() -> None:
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
                trade_date=datetime(2026, 5, 1, tzinfo=IST_ZONE).date(),
                symbol="Nifty 50",
                interval="1minute",
                strategy_name="test",
                option_type="CE",
                side="BUY",
                expiry_date=datetime(2026, 5, 6, tzinfo=IST_ZONE).date(),
                strike=24100.0,
                quantity=50,
                status="CLOSED",
                entry_price=100.0,
                entry_premium=100.0,
                stop_loss=80.0,
                trailing_stop=0.0,
                realized_pnl=500.0,
                opened_at=datetime(2026, 5, 1, 10, 0, tzinfo=IST_ZONE),
                metadata_json={"execution_mode": "paper"},
            )
        )
        session.commit()
        reset_paper_account(session, starting_balance=200000.0, clear_open_positions=False)
        session.commit()

        metrics = compute_paper_portfolio_metrics(session, settings=Settings(execution_capital=100000.0))

        assert metrics["starting_balance"] == 200000.0
        assert metrics["realized_pnl"] == 0.0
        assert metrics["available_balance"] == 200000.0
    finally:
        session.close()
