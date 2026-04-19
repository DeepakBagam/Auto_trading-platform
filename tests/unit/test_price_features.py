import pandas as pd

from feature_engine.price_features import build_price_features


def test_build_price_features_includes_imported_technical_columns() -> None:
    periods = 80
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=periods, freq="D"),
            "open": [100 + i * 0.5 for i in range(periods)],
            "high": [101 + i * 0.5 for i in range(periods)],
            "low": [99 + i * 0.5 for i in range(periods)],
            "close": [100.2 + i * 0.55 for i in range(periods)],
            "volume": [1000 + i * 10 for i in range(periods)],
        }
    )

    out = build_price_features(df)

    for col in [
        "macd_hist",
        "kc_mid",
        "kc_upper",
        "kc_lower",
        "body_pct_range",
        "upper_wick_pct",
        "lower_wick_pct",
        "pattern_marubozu",
        "pattern_hanging_man",
        "pattern_shooting_star",
        "pattern_spinning_top",
        "pattern_engulfing",
    ]:
        assert col in out.columns


def test_build_price_features_treats_zero_volume_indexes_as_neutral() -> None:
    periods = 40
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=periods, freq="h"),
            "open": [100 + i * 0.2 for i in range(periods)],
            "high": [101 + i * 0.2 for i in range(periods)],
            "low": [99 + i * 0.2 for i in range(periods)],
            "close": [100.1 + i * 0.25 for i in range(periods)],
            "volume": [0 for _ in range(periods)],
        }
    )

    out = build_price_features(df)

    assert out["vwap"].equals(out["close"])
    assert (out["volume_ratio_20"] == 1.0).all()
