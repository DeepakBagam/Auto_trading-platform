import pytest

from backend.utils.config import Settings


def test_live_execution_requires_credentials_and_symbols() -> None:
    with pytest.raises(ValueError, match="UPSTOX_ACCESS_TOKEN"):
        Settings(
            execution_enabled=True,
            execution_mode="live",
            execution_symbols="",
            upstox_access_token="",
            execution_accept_external_webhook=False,
        )


def test_live_execution_external_webhook_requires_secret() -> None:
    with pytest.raises(ValueError, match="PINE_WEBHOOK_SECRET"):
        Settings(
            execution_enabled=True,
            execution_mode="live",
            execution_symbols="Nifty 50",
            upstox_access_token="token",
            pine_webhook_secret="",
            execution_accept_external_webhook=True,
        )


def test_sandbox_mode_does_not_treat_live_token_as_sandbox_token() -> None:
    settings = Settings(
        _env_file=None,
        execution_enabled=True,
        execution_mode="sandbox",
        upstox_access_token="live-token",
        upstox_sandbox_access_token="",
    )

    assert settings.upstox_sandbox_access_token == ""


def test_legacy_paper_mode_migrates_to_disabled_sandbox() -> None:
    settings = Settings(execution_enabled=True, execution_mode="paper")

    assert settings.execution_mode == "sandbox"
    assert settings.execution_enabled is False


def test_execution_symbols_exclude_india_vix_from_market_data_fallback() -> None:
    settings = Settings(
        _env_file=None,
        execution_symbols="",
        upstox_instrument_keys="NSE_INDEX|Nifty 50,NSE_INDEX|Nifty Bank,NSE_INDEX|India VIX",
    )

    assert settings.execution_symbol_list == ["Nifty 50", "Nifty Bank"]


def test_signal_profile_resolves_normalized_symbol_name() -> None:
    settings = Settings(
        _env_file=None,
        SIGNAL_SYMBOL_PROFILES='{"Nifty Bank":{"signal_min_adx":18,"entry_window_start":"09:25"}}',
    )

    assert settings.signal_profile_for_symbol("Bank Nifty") == {
        "signal_min_adx": 18,
        "entry_window_start": "09:25",
    }
