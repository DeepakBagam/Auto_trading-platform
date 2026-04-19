import pandas as pd

from data_layer.processors.candle_resampler import resample_candles


def test_resample_1m_to_30m():
    ts = pd.date_range("2026-01-01 09:15:00+00:00", periods=60, freq="1min")
    df = pd.DataFrame(
        {
            "ts": ts,
            "open": [100.0] * 60,
            "high": [101.0] * 60,
            "low": [99.0] * 60,
            "close": [100.5] * 60,
            "volume": [10.0] * 60,
        }
    )
    out = resample_candles(df, "30minute")
    assert len(out) == 2
    assert out["volume"].sum() == 600.0
