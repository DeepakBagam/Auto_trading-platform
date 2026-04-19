from datetime import date, datetime
from types import SimpleNamespace

from utils.email_reports import build_daily_summary_message, build_gap_report_message


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


class _FakeDb:
    def __init__(self, *, get_value=None, execute_rows=None, scalar_values=None):
        self._get_value = get_value
        self._execute_rows = list(execute_rows or [])
        self._scalar_values = list(scalar_values or [])

    def get(self, _model, _key):
        return self._get_value

    def execute(self, _query):
        rows = self._execute_rows.pop(0) if self._execute_rows else []
        return _FakeResult(rows)

    def scalar(self, _query):
        return self._scalar_values.pop(0) if self._scalar_values else None


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
        execution_symbol_list=["Nifty 50"],
        instrument_keys=["NSE_INDEX|Nifty 50"],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_gap_report_message_contains_gap_table(monkeypatch) -> None:
    monkeypatch.setattr("utils.email_reports.is_trading_day", lambda *_args, **_kwargs: True)
    fake_db = _FakeDb(
        scalar_values=[24210.5],
        execute_rows=[
            [
                SimpleNamespace(open=24300.0, close=24366.9),
            ]
        ],
    )

    message = build_gap_report_message(
        fake_db,
        report_date=date.fromisoformat("2026-04-17"),
        settings=_settings(),
    )

    assert message is not None
    assert "Market Gap Report" in message["Subject"]
    body_part = message.get_body(preferencelist=("plain",))
    body = body_part.get_content() if body_part is not None else message.get_content()
    assert "Nifty 50: prev_close=24210.50 open=24300.00 gap=+89.50 latest=24366.90" in body


def test_build_daily_summary_message_contains_trades_orders_and_signals() -> None:
    trade_date = date.fromisoformat("2026-04-17")
    fake_db = _FakeDb(
        get_value=SimpleNamespace(
            date=trade_date,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            total_pnl=845.0,
            max_profit_trade=845.0,
            max_loss_trade=0.0,
            win_rate=100.0,
            is_green=True,
        ),
        scalar_values=[2, 3],
        execute_rows=[
            [
                SimpleNamespace(
                    symbol="Nifty 50",
                    strike=24350.0,
                    option_type="CE",
                    quantity=65,
                    entry_premium=121.5,
                    entry_price=121.5,
                    exit_premium=134.5,
                    current_premium=134.5,
                    current_price=134.5,
                    current_sl=116.0,
                    initial_sl=116.0,
                    stop_loss=116.0,
                    tsl_active=True,
                    target_premium=145.0,
                    take_profit=145.0,
                    realized_pnl=845.0,
                    unrealized_pnl=0.0,
                    pnl_value=845.0,
                    opened_at=datetime.fromisoformat("2026-04-17T10:04:00+05:30"),
                    exit_reason="TARGET",
                )
            ],
            [
                SimpleNamespace(
                    created_at=datetime.fromisoformat("2026-04-17T10:04:00+05:30"),
                    symbol="Nifty 50",
                    strike_price=24350.0,
                    option_type="CE",
                    order_kind="ENTRY",
                    side="BUY",
                    quantity=65,
                    price=121.5,
                    current_sl=116.0,
                    target_premium=145.0,
                    realized_pnl=None,
                    unrealized_pnl=0.0,
                    status="FILLED",
                )
            ],
            [
                SimpleNamespace(
                    timestamp=datetime.fromisoformat("2026-04-17T10:03:00+05:30"),
                    symbol="Nifty 50",
                    ml_signal="BUY",
                    pine_signal="BUY",
                    ai_score=72.4,
                    combined_score=0.78,
                    consensus="BUY",
                    skip_reason=None,
                    trade_placed=True,
                )
            ],
        ],
    )

    message = build_daily_summary_message(fake_db, trade_date=trade_date, settings=_settings())

    assert message is not None
    assert "Daily Trading Summary" in message["Subject"]
    body_part = message.get_body(preferencelist=("plain",))
    body = body_part.get_content() if body_part is not None else message.get_content()
    assert "Total Trades: 1" in body
    assert "Nifty 50 24350 CE qty=65 entry=121.50 exit=134.50 pnl=+845.00" in body
    assert "Orders Logged: 2" in body
    assert "Signals Logged: 3" in body
