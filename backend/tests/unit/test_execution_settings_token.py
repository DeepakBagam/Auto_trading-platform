from datetime import date

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api.routes.execution import (
    RuntimeSettingsRequest,
    _check_upstox_profile,
    _serializable_settings,
    delete_position,
    update_runtime_settings,
)
from backend.db.models import Base, ExecutionPosition
from backend.utils.config import Settings


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def test_check_upstox_profile_reports_ok(monkeypatch) -> None:
    captured = {}

    def fake_get(url, *, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Response(
            200,
            {"data": {"user_name": "Trader", "email": "trader@example.test", "broker": "upstox"}},
        )

    monkeypatch.setattr("backend.api.routes.execution._requests.get", fake_get)

    out = _check_upstox_profile(Settings(_env_file=None, upstox_base_url="https://example.test"), "token")

    assert out["status"] == "ok"
    assert out["user_name"] == "Trader"
    assert captured["url"] == "https://example.test/v2/user/profile"
    assert captured["headers"]["Authorization"] == "Bearer token"
    assert captured["timeout"] == 5


def test_check_upstox_profile_reports_expired_token(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.api.routes.execution._requests.get",
        lambda *args, **kwargs: _Response(401),
    )

    out = _check_upstox_profile(Settings(_env_file=None), "expired-token")

    assert out["status"] == "error"
    assert "expired" in out["detail"]


def test_serializable_settings_exposes_signal_max_per_day() -> None:
    out = _serializable_settings(Settings(_env_file=None, signal_max_per_day=4))

    assert out["signal_max_per_day"] == 4


def test_runtime_settings_rejects_zero_signal_max_per_day() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        with pytest.raises(HTTPException, match="Successful trades per symbol must be at least 1"):
            update_runtime_settings(RuntimeSettingsRequest(signal_max_per_day=0), session)
    finally:
        session.close()


def test_runtime_settings_rejects_india_vix_for_option_execution() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        with pytest.raises(HTTPException, match="has no supported Upstox option contracts"):
            update_runtime_settings(
                RuntimeSettingsRequest(execution_symbols=["Nifty 50", "India VIX"]),
                session,
            )
    finally:
        session.close()


@pytest.mark.parametrize("status", ["ENTRY_PENDING", "EXIT_PENDING"])
def test_delete_position_rejects_pending_reconciliation(status: str) -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        position = ExecutionPosition(
            trade_date=date(2026, 6, 18),
            symbol="Nifty 50",
            interval="1minute",
            strategy_name="test",
            option_type="CE",
            expiry_date=date(2026, 6, 25),
            strike=25000.0,
            quantity=75,
            status=status,
            entry_price=100.0,
            stop_loss=75.0,
            trailing_stop=75.0,
            metadata_json={"reconciliation_needed": True},
        )
        session.add(position)
        session.commit()

        with pytest.raises(HTTPException) as exc_info:
            delete_position(position.id, session)

        assert exc_info.value.status_code == 409
        assert "broker reconciliation is pending" in exc_info.value.detail
        assert session.get(ExecutionPosition, position.id) is position
    finally:
        session.close()
