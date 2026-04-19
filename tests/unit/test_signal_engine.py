from prediction_engine.signal_engine import build_trade_signal


def test_build_trade_signal_returns_buy_for_strong_1m_setup() -> None:
    out = build_trade_signal(
        symbol="Nifty 50",
        interval="1minute",
        direction="BUY",
        confidence=0.84,
        latest_price=24000.0,
        predicted_price=24110.0,
        pred_high=24140.0,
        pred_low=23940.0,
        pred_interval_close={"p10": 23960.0, "p90": 24160.0},
        technical_context={
            "open": 24010.0,
            "rsi_14": 58.0,
            "macd": 0.8,
            "macd_signal": 0.4,
            "macd_hist": 0.25,
            "macd_hist_delta_1": 0.08,
            "ema_9": 24045.0,
            "ema_21": 23990.0,
            "ema_50": 23960.0,
            "ema_21_slope_3": 14.0,
            "atr_14": 55.0,
            "close": 24050.0,
            "body_pct_range": 0.55,
        },
    )
    assert out.action == "BUY"
    assert out.conviction == "high"
    assert out.risk_reward_ratio is not None


def test_build_trade_signal_returns_hold_when_primary_indicator_gate_blocks() -> None:
    out = build_trade_signal(
        symbol="Nifty 50",
        interval="1minute",
        direction="BUY",
        confidence=0.80,
        latest_price=24000.0,
        predicted_price=24120.0,
        pred_high=24150.0,
        pred_low=23980.0,
        pred_interval_close={"p10": 23990.0, "p90": 24180.0},
        technical_context={
            "open": 24020.0,
            "rsi_14": 74.0,
            "macd": 0.6,
            "macd_signal": 0.5,
            "macd_hist": 0.2,
            "macd_hist_delta_1": 0.06,
            "ema_9": 24030.0,
            "ema_21": 24010.0,
            "ema_50": 23990.0,
            "ema_21_slope_3": 10.0,
            "atr_14": 48.0,
            "close": 24005.0,
            "body_pct_range": 0.20,
        },
    )
    assert out.action == "HOLD"
    assert any("No Pine-style BUY strategy passed" in r or "No Pine-style BUY strategy confirmation" in r for r in out.reasons)


def test_build_trade_signal_returns_hold_when_confidence_is_too_low() -> None:
    out = build_trade_signal(
        symbol="Nifty 50",
        interval="1minute",
        direction="BUY",
        confidence=0.41,
        latest_price=24000.0,
        predicted_price=24120.0,
        pred_high=24150.0,
        pred_low=23980.0,
        pred_interval_close={"p10": 23990.0, "p90": 24180.0},
    )
    assert out.action == "HOLD"
    assert any("ML confidence below BUY threshold" in r for r in out.reasons)


def test_build_trade_signal_blocks_buy_for_india_vix_profile() -> None:
    out = build_trade_signal(
        symbol="India VIX",
        interval="1minute",
        direction="BUY",
        confidence=0.86,
        latest_price=16.0,
        predicted_price=16.3,
        pred_high=16.4,
        pred_low=15.8,
        pred_interval_close={"p10": 15.9, "p90": 16.5},
        technical_context={
            "open": 15.95,
            "rsi_14": 62.0,
            "macd": 0.12,
            "macd_signal": 0.05,
            "macd_hist": 0.07,
            "macd_hist_delta_1": 0.02,
            "ema_9": 16.08,
            "ema_21": 15.98,
            "ema_50": 15.85,
            "ema_21_slope_3": 0.06,
            "close": 16.12,
            "body_pct_range": 0.42,
        },
    )
    assert out.action == "HOLD"
    assert any("blocks BUY entries" in r for r in out.reasons)


def test_build_trade_signal_returns_sell_for_india_vix_short_bias() -> None:
    out = build_trade_signal(
        symbol="India VIX",
        interval="1minute",
        direction="SELL",
        confidence=0.78,
        latest_price=16.0,
        predicted_price=15.72,
        pred_high=16.08,
        pred_low=15.68,
        pred_interval_close={"p10": 15.70, "p90": 16.02},
        technical_context={
            "open": 16.02,
            "high": 16.04,
            "low": 15.71,
            "close": 15.76,
            "vwap": 15.90,
            "rsi_14": 39.0,
            "macd": -0.08,
            "macd_signal": -0.03,
            "macd_hist": -0.05,
            "macd_hist_delta_1": -0.02,
            "ema_9": 15.84,
            "ema_21": 15.90,
            "ema_50": 15.98,
            "ema_21_slope_3": -0.04,
            "atr_14": 0.18,
            "body_pct_range": 0.42,
            "upper_wick_pct": 0.10,
        },
    )
    assert out.action == "SELL"
    assert any("SELL strategy confirmations" in r for r in out.reasons)


def test_build_trade_signal_requires_short_interval_alignment_when_present() -> None:
    out = build_trade_signal(
        symbol="Nifty 50",
        interval="1minute",
        direction="BUY",
        confidence=0.84,
        latest_price=24000.0,
        predicted_price=24110.0,
        pred_high=24140.0,
        pred_low=23940.0,
        pred_interval_close={"p10": 23960.0, "p90": 24160.0},
        technical_context={
            "open": 24010.0,
            "high": 24070.0,
            "low": 23995.0,
            "close": 24050.0,
            "vwap": 24000.0,
            "rsi_14": 58.0,
            "macd": 0.8,
            "macd_signal": 0.4,
            "macd_hist": 0.25,
            "macd_hist_delta_1": 0.08,
            "ema_9": 24045.0,
            "ema_21": 23990.0,
            "ema_50": 23960.0,
            "ema_21_slope_3": 14.0,
            "atr_14": 55.0,
            "body_pct_range": 0.55,
            "mtf_3m_action": "NEUTRAL",
            "mtf_5m_action": "SELL",
        },
    )
    assert out.action == "HOLD"
    assert any(
        "alignment is missing" in r
        or "opposite-side score is stronger" in r
        or "No Pine-style BUY strategy confirmation passed" in r
        for r in out.reasons
    )


def test_build_trade_signal_derives_direction_from_forecast_when_model_is_hold() -> None:
    out = build_trade_signal(
        symbol="Nifty 50",
        interval="1minute",
        direction="HOLD",
        confidence=0.84,
        latest_price=24000.0,
        predicted_price=24120.0,
        pred_high=24160.0,
        pred_low=23940.0,
        pred_interval_close={"p10": 23970.0, "p90": 24180.0},
        technical_context={
            "open": 24010.0,
            "high": 24080.0,
            "low": 23995.0,
            "close": 24055.0,
            "vwap": 24005.0,
            "rsi_14": 58.0,
            "macd": 0.8,
            "macd_signal": 0.4,
            "macd_hist": 0.25,
            "macd_hist_delta_1": 0.08,
            "ema_9": 24045.0,
            "ema_21": 23990.0,
            "ema_50": 23960.0,
            "ema_21_slope_3": 14.0,
            "atr_14": 55.0,
            "body_pct_range": 0.55,
        },
    )
    assert out.action == "BUY"
