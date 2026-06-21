import os
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.db.models import Base, ExecutionPosition
from backend.execution_engine.live_service import compute_sandbox_portfolio_metrics
from backend.utils.app_state import (
    apply_runtime_execution_settings,
    get_runtime_execution_settings,
    get_runtime_trading_mode,
    reset_sandbox_account,
    set_runtime_execution_settings,
    set_runtime_trading_mode,
)
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
        assert get_runtime_trading_mode(session, settings=Settings(execution_mode="sandbox")) == "sandbox"
        set_runtime_trading_mode(session, "live")
        session.commit()
        assert get_runtime_trading_mode(session, settings=Settings(execution_mode="sandbox")) == "live"
    finally:
        session.close()


def test_legacy_runtime_paper_mode_migrates_to_disabled_sandbox() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        from backend.utils.app_state import set_setting_value

        set_setting_value(session, "runtime_trading_mode", "paper")
        set_runtime_execution_settings(session, {"execution_enabled": True})
        session.commit()

        assert get_runtime_trading_mode(session, settings=Settings(_env_file=None)) == "sandbox"
        assert get_runtime_execution_settings(session)["execution_enabled"] is False
    finally:
        session.close()


def test_upstox_token_persists_to_runtime_env_and_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UPSTOX_ACCESS_TOKEN", raising=False)
    (tmp_path / ".env").write_text("OTHER_KEY=value\nUPSTOX_ACCESS_TOKEN=old-token\n", encoding="utf-8")
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        set_runtime_execution_settings(session, {"upstox_access_token": "  fresh-token  "})
        session.commit()

        runtime = get_runtime_execution_settings(session)

        assert runtime["upstox_access_token"] == "fresh-token"
        assert os.environ["UPSTOX_ACCESS_TOKEN"] == "fresh-token"
        assert (tmp_path / ".env").read_text(encoding="utf-8") == (
            "OTHER_KEY=value\nUPSTOX_ACCESS_TOKEN=fresh-token\n"
        )
    finally:
        session.close()


def test_sandbox_token_persists_separately_from_live_token(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UPSTOX_SANDBOX_ACCESS_TOKEN", raising=False)
    (tmp_path / ".env").write_text("UPSTOX_ACCESS_TOKEN=live-token\n", encoding="utf-8")
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        set_runtime_execution_settings(
            session,
            {"upstox_sandbox_access_token": " sandbox-token "},
        )
        session.commit()

        assert os.environ["UPSTOX_SANDBOX_ACCESS_TOKEN"] == "sandbox-token"
        env_text = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "UPSTOX_ACCESS_TOKEN=live-token" in env_text
        assert "UPSTOX_SANDBOX_ACCESS_TOKEN=sandbox-token" in env_text
    finally:
        session.close()


def test_signal_max_per_day_applies_from_runtime_settings() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        set_runtime_execution_settings(session, {"signal_max_per_day": 4})
        session.commit()
        settings = Settings(_env_file=None, signal_max_per_day=2)

        apply_runtime_execution_settings(session, settings)

        assert get_runtime_execution_settings(session)["signal_max_per_day"] == 4
        assert settings.signal_max_per_day == 4
    finally:
        session.close()


def test_reset_sandbox_account_excludes_positions_before_reset() -> None:
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
                metadata_json={"execution_mode": "sandbox"},
            )
        )
        session.commit()
        reset_sandbox_account(session, starting_balance=200000.0, clear_open_positions=False)
        session.commit()

        metrics = compute_sandbox_portfolio_metrics(session, settings=Settings(execution_capital=100000.0))

        assert metrics["starting_balance"] == 200000.0
        assert metrics["realized_pnl"] == 0.0
        assert metrics["available_balance"] == 200000.0
    finally:
        session.close()
