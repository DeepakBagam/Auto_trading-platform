from __future__ import annotations

from datetime import datetime

import pandas as pd

from backend.execution_engine.scalping_strategy import (
    ScalpingConfig,
    build_scalping_features,
    detect_scalp_signal,
)
from backend.utils.constants import IST_ZONE


def _base_candles() -> pd.DataFrame:
    rows: list[dict[str, float | datetime]] = []
    ts = datetime(2026, 4, 21, 9, 15, tzinfo=IST_ZONE)
    price = 24000.0
    for i in range(80):
        drift = 0.8 if i < 70 else 1.2
        open_ = price
        close = price + drift
        high = close + 0.6
        low = open_ - 0.4
        rows.append(
            {
                "ts": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000 + (i % 5) * 100,
            }
        )
        price = close
        ts += pd.Timedelta(minutes=1)
    return pd.DataFrame(rows)


def test_detect_scalp_signal_triggers_high_grade_micro_momentum() -> None:
    candles = _base_candles()
    for idx in range(76, 79):
        candles.loc[idx, "open"] = candles.loc[idx - 1, "close"] - 0.5
        candles.loc[idx, "close"] = candles.loc[idx, "open"] + 1.5
        candles.loc[idx, "high"] = candles.loc[idx, "close"] + 1.2
        candles.loc[idx, "low"] = candles.loc[idx, "open"] - 1.5
        candles.loc[idx, "volume"] = 2600
    candles.loc[79, "open"] = candles.loc[78, "close"] + 0.4
    candles.loc[79, "close"] = candles.loc[79, "open"] + 3.8
    candles.loc[79, "high"] = candles.loc[79, "close"] + 0.3
    candles.loc[79, "low"] = candles.loc[79, "open"] - 0.8
    candles.loc[79, "volume"] = 4500

    cfg = ScalpingConfig(
        min_momentum_score=60.0,
        min_pullback_score=60.0,
        min_atr_points=1.0,
        max_vwap_extension_atr=20.0,
        momentum_adx_max=100.0,
        momentum_rsi_long=50.0,
    )
    features = build_scalping_features(candles, cfg)
    signal = detect_scalp_signal(
        features.set_index("ts"),
        now=features.iloc[-1]["ts"].to_pydatetime(),
        trades_today=0,
        last_exit_ts=None,
        config=cfg,
    )

    assert signal.action == "BUY"
    assert signal.setup == "MICRO_MOMENTUM"
    assert signal.score >= 60.0


def test_detect_scalp_signal_skips_overextended_breakout() -> None:
    candles = _base_candles()
    for idx in range(76, 80):
        candles.loc[idx, "open"] = candles.loc[idx - 1, "close"] + 2.5
        candles.loc[idx, "close"] = candles.loc[idx, "open"] + 4.0
        candles.loc[idx, "high"] = candles.loc[idx, "close"] + 0.2
        candles.loc[idx, "low"] = candles.loc[idx, "open"] - 0.1
        candles.loc[idx, "volume"] = 2200

    features = build_scalping_features(candles, ScalpingConfig())
    signal = detect_scalp_signal(
        features.set_index("ts"),
        now=features.iloc[-1]["ts"].to_pydatetime(),
        trades_today=0,
        last_exit_ts=None,
        config=ScalpingConfig(),
    )

    assert signal.action == "HOLD"
    assert signal.reasons[0] in {
        "atr_too_low",
        "vwap_extension_too_large",
        "no_scalp_setup",
        "chop_noise",
    }
