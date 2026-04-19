import pandas as pd


def test_next_day_label_shift():
    df = pd.DataFrame(
        {
            "session_date": pd.date_range("2026-01-01", periods=3, freq="D"),
            "close": [100.0, 105.0, 102.0],
            "open": [99.0, 104.0, 101.0],
            "high": [101.0, 106.0, 103.0],
            "low": [98.0, 103.0, 100.0],
        }
    )
    df["next_close"] = df["close"].shift(-1)
    assert df.loc[0, "next_close"] == 105.0
    assert pd.isna(df.loc[2, "next_close"])
