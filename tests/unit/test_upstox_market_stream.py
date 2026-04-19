from datetime import datetime
from types import SimpleNamespace

from data_layer.streamers.upstox_market_stream import UpstoxMarketStream
from utils.constants import IST_ZONE


class _FakeStreamer:
    Event = {
        "OPEN": "open",
        "MESSAGE": "message",
        "ERROR": "error",
        "CLOSE": "close",
        "RECONNECTING": "reconnecting",
        "AUTO_RECONNECT_STOPPED": "autoReconnectStopped",
    }

    def on(self, *_args, **_kwargs):
        return None

    def auto_reconnect(self, *_args, **_kwargs):
        return None

    def connect(self):
        return None

    def disconnect(self):
        return None


def _epoch_ms(ts: str) -> str:
    return str(int(datetime.fromisoformat(ts).timestamp() * 1000))


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        upstox_access_token="token",
        instrument_keys=["NSE_INDEX|Nifty 50"],
        upstox_websocket_mode="ltpc",
        upstox_websocket_reconnect_interval_seconds=5,
        upstox_websocket_retry_count=10,
    )


def test_market_stream_aggregates_ticks_into_closed_1m_candle() -> None:
    stream = UpstoxMarketStream(settings=_settings(), streamer=_FakeStreamer())
    persisted = []
    stream._persist_candle_record = persisted.append  # type: ignore[method-assign]

    stream.handle_market_data(
        {
            "feeds": {
                "NSE_INDEX|Nifty 50": {
                    "ltpc": {"ltp": 100.0, "ltt": _epoch_ms("2026-04-08T09:15:05+05:30")}
                }
            },
            "currentTs": _epoch_ms("2026-04-08T09:15:05+05:30"),
        }
    )
    stream.handle_market_data(
        {
            "feeds": {
                "NSE_INDEX|Nifty 50": {
                    "ltpc": {"ltp": 102.5, "ltt": _epoch_ms("2026-04-08T09:15:45+05:30")}
                }
            },
            "currentTs": _epoch_ms("2026-04-08T09:15:45+05:30"),
        }
    )
    stream.flush_closed_candles(datetime(2026, 4, 8, 9, 16, tzinfo=IST_ZONE))

    assert len(persisted) == 1
    candle = persisted[0]
    assert candle.instrument_key == "NSE_INDEX|Nifty 50"
    assert candle.interval == "1minute"
    assert candle.ts == datetime(2026, 4, 8, 9, 15, tzinfo=IST_ZONE)
    assert candle.open == 100.0
    assert candle.high == 102.5
    assert candle.low == 100.0
    assert candle.close == 102.5
    assert candle.source == "upstox_ws"


def test_market_stream_extracts_ltpc_from_nested_full_feed() -> None:
    stream = UpstoxMarketStream(settings=_settings(), streamer=_FakeStreamer())

    ltpc = stream._extract_ltpc(
        {
            "fullFeed": {
                "indexFF": {
                    "ltpc": {"ltp": 23123.65, "ltt": _epoch_ms("2026-04-08T09:20:00+05:30")}
                }
            }
        }
    )

    assert ltpc == {"ltp": 23123.65, "ltt": _epoch_ms("2026-04-08T09:20:00+05:30")}


def test_market_stream_extracts_ohlc_records_from_full_feed() -> None:
    stream = UpstoxMarketStream(settings=_settings(), streamer=_FakeStreamer())

    records = stream._extract_ohlc_records(
        "NSE_INDEX|Nifty 50",
        {
            "fullFeed": {
                "marketFF": {
                    "marketOHLC": {
                        "ohlc": [
                            {
                                "interval": "I1",
                                "open": 23120.0,
                                "high": 23155.0,
                                "low": 23110.0,
                                "close": 23150.0,
                                "vol": "2500",
                                "ts": _epoch_ms("2026-04-08T09:16:00+05:30"),
                            },
                        ]
                    },
                    "oi": 12345,
                }
            }
        },
    )

    assert len(records) == 1
    assert records[0].interval == "1minute"
    assert records[0].ts == datetime(2026, 4, 8, 9, 16, tzinfo=IST_ZONE)
    assert records[0].volume == 2500.0
    assert records[0].oi == 12345.0
    assert records[0].source == "upstox_ws_ohlc"
