from datetime import datetime
from types import SimpleNamespace

from prediction_engine.signal_engine import TradeSignal
from prediction_engine.consensus_engine import get_consensus_signal


def test_consensus_uses_local_pine_fallback_when_webhook_missing(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-04-17T10:05:00+05:30")
    settings = SimpleNamespace(
        ml_buy_threshold=0.62,
        ml_sell_threshold=0.62,
        ml_min_expected_move=80.0,
        pine_signal_max_age_seconds=60,
        ai_quality_minimum=65.0,
        combined_score_threshold=0.65,
        entry_window_start="09:20",
        entry_window_end="13:30",
        execution_capital=500000.0,
        execution_max_daily_loss_pct=0.05,
        execution_max_simultaneous_trades=2,
        execution_max_daily_trades=5,
    )

    monkeypatch.setattr(
        "prediction_engine.consensus_engine.predict_single",
        lambda **_kwargs: SimpleNamespace(
            direction="BUY",
            confidence=0.82,
            confidence_score=0.82,
            pred_close=24110.0,
            pred_high=24140.0,
            pred_low=23960.0,
            pred_interval=SimpleNamespace(model_dump=lambda: {"close": {"p10": 23980.0, "p90": 24160.0}}),
        ),
    )
    monkeypatch.setattr("prediction_engine.consensus_engine.latest_price_for_symbol", lambda *_args, **_kwargs: 24000.0)
    monkeypatch.setattr(
        "prediction_engine.consensus_engine.technical_context_for_symbol",
        lambda *_args, **_kwargs: {
            "open": 24010.0,
            "high": 24060.0,
            "low": 23995.0,
            "close": 24055.0,
            "vwap": 24005.0,
            "rsi_14": 59.0,
            "macd": 0.9,
            "macd_signal": 0.4,
            "macd_hist": 0.5,
            "macd_hist_delta_1": 0.12,
            "ema_9": 24040.0,
            "ema_21": 23990.0,
            "ema_50": 23950.0,
            "ema_21_slope_3": 12.0,
            "atr_14": 52.0,
            "body_pct_range": 0.48,
        },
    )
    monkeypatch.setattr("prediction_engine.consensus_engine.recent_news_sentiment_for_symbol", lambda *_args, **_kwargs: 0.35)
    monkeypatch.setattr("prediction_engine.consensus_engine._latest_pine_signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("prediction_engine.consensus_engine.latest_vix_level", lambda *_args, **_kwargs: 15.0)
    monkeypatch.setattr("prediction_engine.consensus_engine._recent_prediction_accuracy", lambda *_args, **_kwargs: 0.62)
    monkeypatch.setattr("prediction_engine.consensus_engine._daily_realized_pnl", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr("prediction_engine.consensus_engine._open_positions_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("prediction_engine.consensus_engine._daily_trades_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        "prediction_engine.consensus_engine.score_trade_intelligence",
        lambda **_kwargs: SimpleNamespace(score=76.0, reasons=["AI passed"]),
    )

    result = get_consensus_signal(
        db=SimpleNamespace(),
        symbol="Nifty 50",
        interval="1minute",
        now=now,
        settings=settings,
        persist=False,
    )

    assert result.pine_signal == "BUY"
    assert result.details["pine_source"] == "local_fallback"


def test_consensus_reinterprets_hold_model_signal_from_expected_move(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-04-17T10:05:00+05:30")
    settings = SimpleNamespace(
        ml_buy_threshold=0.62,
        ml_sell_threshold=0.62,
        ml_min_expected_move=80.0,
        pine_signal_max_age_seconds=60,
        ai_quality_minimum=65.0,
        combined_score_threshold=0.65,
        entry_window_start="09:20",
        entry_window_end="13:30",
        execution_capital=500000.0,
        execution_max_daily_loss_pct=0.05,
        execution_max_simultaneous_trades=1,
        execution_max_daily_trades=5,
    )

    monkeypatch.setattr(
        "prediction_engine.consensus_engine.predict_single",
        lambda **_kwargs: SimpleNamespace(
            direction="SELL",
            confidence=0.74,
            confidence_score=0.74,
            pred_close=23880.0,
            pred_high=23930.0,
            pred_low=23820.0,
            pred_interval=None,
        ),
    )
    monkeypatch.setattr("prediction_engine.consensus_engine.latest_price_for_symbol", lambda *_args, **_kwargs: 24000.0)
    monkeypatch.setattr(
        "prediction_engine.consensus_engine.technical_context_for_symbol",
        lambda *_args, **_kwargs: {
            "close": 24020.0,
            "vwap": 24015.0,
            "rsi_14": 47.0,
            "macd_hist": -0.2,
            "ema_9": 24010.0,
            "ema_21": 23990.0,
            "ema_50": 23970.0,
            "atr_14": 42.0,
        },
    )
    monkeypatch.setattr("prediction_engine.consensus_engine.recent_news_sentiment_for_symbol", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr("prediction_engine.consensus_engine._latest_pine_signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "prediction_engine.consensus_engine.local_pine_signal",
        lambda *_args, **_kwargs: {
            "action": "NEUTRAL",
            "buy_strategy_names": [],
            "sell_strategy_names": [],
            "market_regime": "range",
        },
    )
    monkeypatch.setattr(
        "prediction_engine.consensus_engine.build_trade_signal",
        lambda **_kwargs: TradeSignal(
            action="HOLD",
            conviction="medium",
            expected_return_pct=-0.005,
            expected_move_points=-120.0,
            stop_loss=None,
            take_profit=None,
            risk_reward_ratio=None,
            technical_score=0.0,
            reasons=["No Pine-style SELL strategy passed"],
        ),
    )
    monkeypatch.setattr("prediction_engine.consensus_engine.actionable_direction_from_forecast", lambda **_kwargs: "SELL")
    monkeypatch.setattr("prediction_engine.consensus_engine.latest_vix_level", lambda *_args, **_kwargs: 16.0)
    monkeypatch.setattr("prediction_engine.consensus_engine._recent_prediction_accuracy", lambda *_args, **_kwargs: 0.55)
    monkeypatch.setattr("prediction_engine.consensus_engine._daily_realized_pnl", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr("prediction_engine.consensus_engine._open_positions_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("prediction_engine.consensus_engine._daily_trades_count", lambda *_args, **_kwargs: 0)

    captured: list[str] = []

    def _score_stub(**kwargs):
        captured.append(kwargs["signal_action"])
        return SimpleNamespace(score=58.0, reasons=["Neutral setup"])

    monkeypatch.setattr("prediction_engine.consensus_engine.score_trade_intelligence", _score_stub)

    result = get_consensus_signal(
        db=SimpleNamespace(),
        symbol="Nifty 50",
        interval="1minute",
        now=now,
        settings=settings,
        persist=False,
    )

    assert "SELL" in captured
    assert result.details["forecast_action"] == "SELL"
    assert result.ml_signal == "SELL"


def test_consensus_allows_pine_led_trade_when_ml_is_weak(monkeypatch) -> None:
    now = datetime.fromisoformat("2026-04-16T11:40:00+05:30")
    settings = SimpleNamespace(
        ml_buy_threshold=0.62,
        ml_sell_threshold=0.62,
        ml_min_expected_move=80.0,
        pine_signal_max_age_seconds=60,
        ai_quality_minimum=65.0,
        combined_score_threshold=0.65,
        entry_window_start="09:20",
        entry_window_end="13:30",
        execution_capital=500000.0,
        execution_max_daily_loss_pct=0.05,
        execution_max_simultaneous_trades=1,
        execution_max_daily_trades=5,
    )

    monkeypatch.setattr(
        "prediction_engine.consensus_engine.predict_single",
        lambda **_kwargs: SimpleNamespace(
            direction="HOLD",
            confidence=0.58,
            confidence_score=0.58,
            pred_close=24020.0,
            pred_high=24040.0,
            pred_low=23970.0,
            pred_interval=None,
        ),
    )
    monkeypatch.setattr("prediction_engine.consensus_engine.latest_price_for_symbol", lambda *_args, **_kwargs: 24000.0)
    monkeypatch.setattr(
        "prediction_engine.consensus_engine.technical_context_for_symbol",
        lambda *_args, **_kwargs: {
            "close": 24035.0,
            "open": 24005.0,
            "high": 24040.0,
            "low": 23995.0,
            "vwap": 24010.0,
            "rsi_14": 57.0,
            "macd": 0.4,
            "macd_signal": 0.2,
            "macd_hist": 0.22,
            "macd_hist_delta_1": 0.05,
            "ema_9": 24028.0,
            "ema_21": 24012.0,
            "ema_50": 23990.0,
            "ema_21_slope_3": 8.0,
            "atr_14": 44.0,
            "body_pct_range": 0.42,
            "mtf_3m_action": "BUY",
            "mtf_5m_action": "BUY",
        },
    )
    monkeypatch.setattr("prediction_engine.consensus_engine.recent_news_sentiment_for_symbol", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr("prediction_engine.consensus_engine._latest_pine_signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "prediction_engine.consensus_engine.local_pine_signal",
        lambda *_args, **_kwargs: {
            "action": "BUY",
            "buy_strategy_names": ["trend_continuation"],
            "sell_strategy_names": [],
            "market_regime": "trend",
        },
    )
    monkeypatch.setattr(
        "prediction_engine.consensus_engine.build_trade_signal",
        lambda **_kwargs: TradeSignal(
            action="HOLD",
            conviction="medium",
            expected_return_pct=0.0008,
            expected_move_points=20.0,
            stop_loss=None,
            take_profit=None,
            risk_reward_ratio=None,
            technical_score=0.0,
            reasons=["ML is weak but Pine is aligned"],
        ),
    )
    monkeypatch.setattr("prediction_engine.consensus_engine.latest_vix_level", lambda *_args, **_kwargs: 14.0)
    monkeypatch.setattr("prediction_engine.consensus_engine._recent_prediction_accuracy", lambda *_args, **_kwargs: 0.55)
    monkeypatch.setattr("prediction_engine.consensus_engine._daily_realized_pnl", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr("prediction_engine.consensus_engine._open_positions_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("prediction_engine.consensus_engine._daily_trades_count", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        "prediction_engine.consensus_engine.score_trade_intelligence",
        lambda **_kwargs: SimpleNamespace(score=38.0, reasons=["Pine-led AI passed"]),
    )

    result = get_consensus_signal(
        db=SimpleNamespace(),
        symbol="Nifty 50",
        interval="1minute",
        now=now,
        settings=settings,
        persist=False,
    )

    assert result.pine_signal == "BUY"
    assert result.details["model_trade_signal"] == "HOLD"
    assert result.consensus == "BUY"
