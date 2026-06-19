from datetime import date, datetime
from types import SimpleNamespace

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.data_layer.collectors.upstox_option_chain import UpstoxOptionChainCollector
from backend.db.models import Base, DataFreshness, OptionQuote
from backend.utils.constants import IST_ZONE


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


def test_collector_skips_non_positive_ltp_and_does_not_mark_empty_snapshot_successful(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("UPSTOX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("UPSTOX_TOKEN_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    settings = SimpleNamespace(
        upstox_base_url="https://example.test/",
        market_data_access_token="market-token",
        upstox_access_token="execution-token",
        upstox_analytics_token="",
    )
    collector = UpstoxOptionChainCollector(settings=settings)
    expiry = date(2026, 6, 25)
    snapshot_ts = datetime(2026, 6, 18, 10, 0, 12, 345678, tzinfo=IST_ZONE)

    with Session(engine) as session:
        result = collector.persist_option_chain(
            session,
            underlying_key="NSE_INDEX|Nifty 50",
            underlying_symbol="Nifty 50",
            expiry_date=expiry,
            fetched_at=snapshot_ts,
            chain_rows=[
                {
                    "strike_price": 24100,
                    "call_options": {
                        "instrument_key": "NSE_FO|ZERO",
                        "market_data": {"ltp": 0},
                    },
                    "put_options": {
                        "instrument_key": "NSE_FO|MISSING",
                        "market_data": {},
                    },
                }
            ],
        )

        assert result == {"inserted": 0, "updated": 0, "rows": 1, "skipped": 2}
        assert session.scalar(select(OptionQuote)) is None
        assert session.scalar(select(DataFreshness)) is None


def test_collector_persists_exact_snapshot_timestamp_for_valid_contracts(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("UPSTOX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("UPSTOX_TOKEN_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    collector = UpstoxOptionChainCollector(
        settings=SimpleNamespace(
            upstox_base_url="https://example.test/",
            market_data_access_token="market-token",
            upstox_access_token="execution-token",
            upstox_analytics_token="",
        )
    )
    expiry = date(2026, 6, 25)
    snapshot_ts = datetime(2026, 6, 18, 10, 0, 12, 345678, tzinfo=IST_ZONE)

    with Session(engine) as session:
        collector.persist_option_chain(
            session,
            underlying_key="NSE_INDEX|Nifty 50",
            underlying_symbol="Nifty 50",
            expiry_date=expiry,
            fetched_at=snapshot_ts,
            chain_rows=[
                {
                    "strike_price": 24100,
                    "call_options": {
                        "instrument_key": "NSE_FO|24100CE",
                        "market_data": {"ltp": 100, "bid_price": 99, "ask_price": 101},
                    },
                }
            ],
        )

        quote = session.scalar(select(OptionQuote))
        freshness = session.scalar(select(DataFreshness))
        assert quote is not None
        assert quote.ts == snapshot_ts.replace(tzinfo=None)
        assert freshness is not None
        assert freshness.details["snapshot_ts"] == snapshot_ts.isoformat()
        assert freshness.details["contracts"] == 1
