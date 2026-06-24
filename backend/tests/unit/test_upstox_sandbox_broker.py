from types import SimpleNamespace

import upstox_client

from backend.execution_engine.broker import BrokerOrderRequest, UpstoxSandboxBroker
from backend.utils.config import (
    Settings,
    read_runtime_upstox_access_token,
    read_runtime_upstox_sandbox_access_token,
)


class FakeOrderApi:
    def __init__(self) -> None:
        self.placed = []
        self.modified = []
        self.cancelled = []

    def place_order(self, body):
        self.placed.append(body)
        return SimpleNamespace(to_dict=lambda: {"data": {"order_ids": ["SANDBOX-1"]}})

    def modify_order(self, body):
        self.modified.append(body)
        return SimpleNamespace(to_dict=lambda: {"data": {"order_id": body.order_id}})

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return SimpleNamespace(to_dict=lambda: {"data": {"order_id": order_id}})


class TimeoutOrderApi(FakeOrderApi):
    def place_order(self, body):
        self.placed.append(body)
        raise TimeoutError("response lost")


class FakeReadResponse:
    def __init__(self, payload, *, ok=True, status_code=200) -> None:
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.content = b"{}"
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeReadSession:
    def __init__(self) -> None:
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeReadResponse(
            {
                "status": "success",
                "data": [{"order_id": "SANDBOX-1", "status": "open pending"}],
            }
        )


def test_sandbox_broker_is_pinned_to_upstox_sandbox_and_maps_product() -> None:
    broker = UpstoxSandboxBroker(access_token="sandbox-token")
    fake = FakeOrderApi()
    broker.order_api = fake

    response = broker.place_order(
        BrokerOrderRequest(
            instrument_key="NSE_FO|12345",
            option_type="CE",
            strike=24000.0,
            expiry_date="2026-06-25",
            side="BUY",
            qty=75,
            order_type="LIMIT",
            price=101.0,
            product="MIS",
        )
    )

    assert broker.configuration.sandbox is True
    assert broker.configuration.order_host == "https://api-sandbox.upstox.com"
    assert response.success is True
    assert response.order_id == "SANDBOX-1"
    assert fake.placed[0].product == "I"
    assert fake.placed[0].instrument_token == "NSE_FO|12345"


def test_sandbox_broker_repins_hosts_when_sdk_ignores_sandbox_flag(monkeypatch) -> None:
    real_configuration = upstox_client.Configuration

    def legacy_configuration(*_args, **_kwargs):
        configuration = real_configuration()
        configuration.sandbox = False
        configuration.host = "https://api.upstox.com"
        configuration.order_host = "https://api-hft.upstox.com"
        return configuration

    monkeypatch.setattr(upstox_client, "Configuration", legacy_configuration)

    broker = UpstoxSandboxBroker(access_token="sandbox-token")

    assert broker.configuration.sandbox is True
    assert broker.configuration.host == "https://api-sandbox.upstox.com"
    assert broker.configuration.order_host == "https://api-sandbox.upstox.com"


def test_sandbox_broker_modify_and_cancel_use_v3_adapter_contract() -> None:
    broker = UpstoxSandboxBroker(access_token="sandbox-token")
    fake = FakeOrderApi()
    broker.order_api = fake

    modified = broker.modify_order(
        "SANDBOX-1",
        trigger_price=90.0,
        price=89.0,
        quantity=75,
        order_type="SL",
    )
    cancelled = broker.cancel_order("SANDBOX-1")

    assert modified.success is True
    assert fake.modified[0].quantity == 75
    assert fake.modified[0].trigger_price == 90.0
    assert fake.modified[0].price == 89.0
    assert cancelled.status == "CANCELLED"
    assert fake.cancelled == ["SANDBOX-1"]


def test_live_and_sandbox_tokens_never_fall_back_to_each_other(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "live-token")
    monkeypatch.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", "sandbox-token")
    settings = Settings(
        _env_file=None,
        upstox_access_token="settings-live",
        upstox_sandbox_access_token="settings-sandbox",
    )

    assert read_runtime_upstox_access_token(settings) == "live-token"
    assert read_runtime_upstox_sandbox_access_token(settings) == "sandbox-token"

    monkeypatch.delenv("UPSTOX_SANDBOX_ACCESS_TOKEN")
    settings.upstox_sandbox_access_token = ""
    assert read_runtime_upstox_sandbox_access_token(settings) == ""


def test_sandbox_mutating_timeout_is_ambiguous_and_not_retried() -> None:
    broker = UpstoxSandboxBroker(access_token="sandbox-token")
    fake = TimeoutOrderApi()
    broker.order_api = fake

    response = broker.place_order(
        BrokerOrderRequest(
            instrument_key="NSE_FO|12345",
            option_type="CE",
            strike=24000.0,
            expiry_date="2026-06-25",
            side="BUY",
            qty=75,
            order_type="LIMIT",
            price=101.0,
        )
    )

    assert response.success is False
    assert response.status == "AMBIGUOUS"
    assert len(fake.placed) == 1


def test_sandbox_read_endpoints_are_pinned_to_sandbox_host() -> None:
    broker = UpstoxSandboxBroker(access_token="sandbox-token")
    session = FakeReadSession()
    broker.session = session

    order_book = broker.get_order_book()
    history = broker.get_order_history("SANDBOX-1")
    status = broker.get_order_status("SANDBOX-1")

    assert order_book["data"][0]["order_id"] == "SANDBOX-1"
    assert session.calls[0][0] == "https://api-sandbox.upstox.com/v2/order/retrieve-all"
    assert session.calls[1][0] == "https://api-sandbox.upstox.com/v2/order/history"
    assert session.calls[1][1]["params"] == {"order_id": "SANDBOX-1"}
    assert session.calls[2][0] == "https://api-sandbox.upstox.com/v2/order/details"
    assert status.success is True
    assert history["status"] == "success"


def test_sandbox_trade_pnl_report_is_explicitly_unsupported() -> None:
    broker = UpstoxSandboxBroker(access_token="sandbox-token")

    payload = broker.get_trade_pnl_report(
        segment="FO",
        financial_year="2627",
        from_date="23-06-2026",
        to_date="23-06-2026",
    )

    assert payload["status"] == "error"
    assert payload["errors"][0]["status"] == "UNSUPPORTED"
