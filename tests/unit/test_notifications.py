from datetime import date, datetime
from types import SimpleNamespace

from utils.constants import IST_ZONE
from utils.notifications import build_order_notification_message, smtp_ready


def _settings(**overrides):
    base = dict(
        env="test",
        smtp_enabled=True,
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username="user",
        smtp_password="pass",
        smtp_from_email="bot@example.com",
        smtp_recipients=["ops@example.com"],
        smtp_use_tls=True,
        smtp_use_ssl=False,
        smtp_timeout_seconds=20,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_smtp_ready_requires_core_fields() -> None:
    assert smtp_ready(_settings())
    assert not smtp_ready(_settings(smtp_enabled=False))
    assert not smtp_ready(_settings(smtp_host=""))
    assert not smtp_ready(_settings(smtp_recipients=[]))


def test_build_order_notification_message_contains_pnl_and_contract() -> None:
    message = build_order_notification_message(
        {
            "symbol": "Nifty 50",
            "order_kind": "EXIT",
            "side": "SELL",
            "status": "FILLED",
            "quantity": 65,
            "strike_price": 24000.0,
            "option_type": "CE",
            "trade_date": date.fromisoformat("2026-04-16"),
            "expiry_date": date.fromisoformat("2026-04-23"),
            "price": 142.5,
            "trigger_price": 118.0,
            "entry_premium": 121.0,
            "exit_premium": 142.5,
            "current_sl": 130.0,
            "target_premium": 150.0,
            "realized_pnl": 1397.5,
            "unrealized_pnl": 0.0,
            "broker_name": "paper",
            "broker_order_id": "paper-123",
            "created_at": datetime.fromisoformat("2026-04-16T10:15:00+05:30"),
            "position_status": "CLOSED",
            "position_opened_at": datetime.fromisoformat("2026-04-16T09:31:00+05:30"),
            "position_closed_at": datetime.fromisoformat("2026-04-16T10:15:00+05:30"),
            "exit_reason": "TARGET",
            "consensus_reason": "Combined score 0.79",
        },
        settings=_settings(),
    )

    assert message is not None
    assert "Nifty 50 EXIT SELL FILLED" in message["Subject"]
    body_part = message.get_body(preferencelist=("plain",))
    body = body_part.get_content() if body_part is not None else message.get_content()
    assert "Contract: 24000.00 CE" in body
    assert "Realized P&L: +1397.50" in body
    assert "Order Time (IST): 2026-04-16 10:15:00 IST" in body
