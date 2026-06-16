from types import SimpleNamespace

from backend.data_layer.collectors.upstox_option_chain import UpstoxOptionChainCollector


def test_collector_accepts_explicit_settings(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("UPSTOX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("UPSTOX_TOKEN_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    settings = SimpleNamespace(
        upstox_base_url="https://example.test/",
        market_data_access_token="market-token",
        upstox_access_token="execution-token",
    )

    collector = UpstoxOptionChainCollector(settings=settings)

    assert collector.settings is settings
    assert collector.base_url == "https://example.test"
    assert collector.headers["Authorization"] == "Bearer execution-token"


def test_collector_prefers_runtime_access_token(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("UPSTOX_ACCESS_TOKEN", "runtime-token")
    monkeypatch.delenv("UPSTOX_TOKEN_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    settings = SimpleNamespace(
        upstox_base_url="https://example.test/",
        market_data_access_token="market-token",
        upstox_access_token="stale-settings-token",
        upstox_analytics_token="analytics-token",
    )

    collector = UpstoxOptionChainCollector(settings=settings)

    assert collector.headers["Authorization"] == "Bearer runtime-token"
