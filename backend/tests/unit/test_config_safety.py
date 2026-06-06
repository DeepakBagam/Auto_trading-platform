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


def test_paper_execution_allows_missing_live_credentials() -> None:
    settings = Settings(
        execution_enabled=True,
        execution_mode="paper",
        upstox_access_token="",
        pine_webhook_secret="",
    )

    assert settings.execution_mode == "paper"
