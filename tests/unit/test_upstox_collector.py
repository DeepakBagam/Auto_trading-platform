from types import SimpleNamespace

from data_layer.collectors.upstox_collector import UpstoxCollector


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

    summary = collector.ingest_historical_batch(db=object(), days_back=1)

    assert summary["NSE_INDEX|Nifty 50:1minute"] == 1
    assert seen_calls == [
        ("historical", "1minute"),
        ("intraday", "1minute"),
    ]

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
