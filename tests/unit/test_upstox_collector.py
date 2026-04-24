from datetime import date, datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from data_layer.collectors.upstox_collector import UpstoxCollector
from db.models import Base, RawCandle
from utils.constants import IST_ZONE


def test_ingest_historical_batch_merges_live_intraday_feed(monkeypatch) -> None:
    monkeypatch.setattr(
        "data_layer.collectors.upstox_collector.get_settings",
        lambda: SimpleNamespace(
            upstox_base_url="https://api.upstox.com",
            upstox_access_token="token",
            instrument_keys=["NSE_INDEX|Nifty 50"],
        ),
    )

    collector = UpstoxCollector()
    seen_calls: list[tuple[str, str]] = []

    def fake_historical(instrument_key: str, interval: str, from_date, to_date):
        seen_calls.append(("historical", interval))
        return [] if interval == "1minute" else [f"hist-{interval}"]

    def fake_intraday(instrument_key: str, interval: str):
        seen_calls.append(("intraday", interval))
        return [f"live-{interval}"]

    def fake_persist(db, records, update_existing=False):
        del update_existing
        return len(records)

    monkeypatch.setattr(collector, "fetch_historical_candles", fake_historical)
    monkeypatch.setattr(collector, "fetch_intraday_candles", fake_intraday)
    monkeypatch.setattr(collector, "persist", fake_persist)
    monkeypatch.setattr(
        collector,
        "rebuild_derived_from_one_minute",
        lambda db, instrument_key, from_date, to_date: {"30minute": 0, "day": 0},
    )

    summary = collector.ingest_historical_batch(db=object(), days_back=1)

    assert summary["NSE_INDEX|Nifty 50:1minute"] == 1
    assert seen_calls == [
        ("historical", "1minute"),
        ("intraday", "1minute"),
    ]


def test_rebuild_derived_from_one_minute_updates_latest_day(monkeypatch) -> None:
    monkeypatch.setattr(
        "data_layer.collectors.upstox_collector.get_settings",
        lambda: SimpleNamespace(
            upstox_base_url="https://api.upstox.com",
            upstox_access_token="token",
            instrument_keys=["NSE_INDEX|Nifty 50"],
        ),
    )

    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        for session_start in (
            datetime(2026, 4, 21, 9, 15, tzinfo=IST_ZONE),
            datetime(2026, 4, 22, 9, 15, tzinfo=IST_ZONE),
        ):
            for minute in range(375):
                price = 100.0 + minute
                session.add(
                    RawCandle(
                        instrument_key="NSE_INDEX|Nifty 50",
                        interval="1minute",
                        ts=session_start + timedelta(minutes=minute),
                        open=price,
                        high=price + 2.0,
                        low=price - 1.0,
                        close=price + 1.0,
                        volume=10.0,
                        oi=None,
                        source="test",
                    )
                )
        session.commit()

        collector = UpstoxCollector()
        summary = collector.rebuild_derived_from_one_minute(
            session,
            instrument_key="NSE_INDEX|Nifty 50",
            from_date=date(2026, 4, 21),
            to_date=date(2026, 4, 22),
        )

        latest_30m = session.scalar(
            select(func.max(RawCandle.ts)).where(
                RawCandle.instrument_key == "NSE_INDEX|Nifty 50",
                RawCandle.interval == "30minute",
            )
        )
        latest_day = session.scalar(
            select(func.max(RawCandle.ts)).where(
                RawCandle.instrument_key == "NSE_INDEX|Nifty 50",
                RawCandle.interval == "day",
            )
        )
        count_30m = session.scalar(
            select(func.count(RawCandle.id)).where(
                RawCandle.instrument_key == "NSE_INDEX|Nifty 50",
                RawCandle.interval == "30minute",
            )
        )

        assert summary == {"30minute": 26, "day": 2}
        assert latest_30m.date().isoformat() == "2026-04-22"
        assert latest_day.date().isoformat() == "2026-04-22"
        assert count_30m == 26
    finally:
        session.close()


def test_normalize_response_1minute_canonicalizes_ohlc(monkeypatch) -> None:
    monkeypatch.setattr(
        "data_layer.collectors.upstox_collector.get_settings",
        lambda: SimpleNamespace(
            upstox_base_url="https://api.upstox.com",
            upstox_access_token="token",
            instrument_keys=["NSE_INDEX|Nifty 50"],
        ),
    )
    collector = UpstoxCollector()

    rows = [
        # open=100, high=98, low=105, close=102 is invalid; normalization must fix high/low.
        ["2026-04-10T09:15:00+05:30", 100.0, 98.0, 105.0, 102.0, 1234.0, 0.0]
    ]
    out = collector._normalize_response(rows, "NSE_INDEX|Nifty 50", "1minute")
    assert len(out) == 1
    record = out[0]
    assert record.ts.isoformat() == "2026-04-10T09:15:00+05:30"
    assert record.open == 100.0
    assert record.close == 102.0
    assert record.high == 105.0
    assert record.low == 98.0


def test_history_window_start_uses_calendar_years(monkeypatch) -> None:
    monkeypatch.setattr(
        "data_layer.collectors.upstox_collector.get_settings",
        lambda: SimpleNamespace(
            upstox_base_url="https://api.upstox.com",
            upstox_access_token="token",
            instrument_keys=["NSE_INDEX|Nifty 50"],
            history_retention_years=2,
        ),
    )

    collector = UpstoxCollector()

    assert collector.history_window_start(date(2026, 4, 21)) == date(2024, 4, 21)


def test_enforce_retention_window_prunes_old_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        "data_layer.collectors.upstox_collector.get_settings",
        lambda: SimpleNamespace(
            upstox_base_url="https://api.upstox.com",
            upstox_access_token="token",
            instrument_keys=["NSE_INDEX|Nifty 50"],
            history_retention_years=2,
        ),
    )

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
            RawCandle(
                instrument_key="NSE_INDEX|Nifty 50",
                interval="1minute",
                ts=datetime(2024, 4, 20, 9, 15, tzinfo=IST_ZONE),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=10.0,
                oi=None,
                source="test",
            )
        )
        session.add(
            RawCandle(
                instrument_key="NSE_INDEX|Nifty 50",
                interval="1minute",
                ts=datetime(2024, 4, 21, 9, 15, tzinfo=IST_ZONE),
                open=102.0,
                high=103.0,
                low=101.0,
                close=102.5,
                volume=10.0,
                oi=None,
                source="test",
            )
        )
        session.commit()

        collector = UpstoxCollector()
        summary = collector.enforce_retention_window(session, as_of=date(2026, 4, 21))

        remaining_count = session.scalar(select(func.count(RawCandle.id)))
        remaining_oldest = session.scalar(select(func.min(RawCandle.ts)))
        assert summary["cutoff_date"] == "2024-04-21"
        assert summary["deleted"]["raw_candles"] == 1
        assert remaining_count == 1
        assert remaining_oldest.date().isoformat() == "2024-04-21"
    finally:
        session.close()
