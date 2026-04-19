from datetime import date, datetime
from types import SimpleNamespace

from api.routes.signal import signal


class _FakeDb:
    def __init__(self, scalar_values):
        self._scalar_values = list(scalar_values)

    def scalar(self, _query):
        if self._scalar_values:
            return self._scalar_values.pop(0)
        return None


def _prediction(**overrides):
    base = dict(
        symbol="Nifty 50",
        interval="1minute",
        prediction_mode="standard",
        source_interval="1minute",
        target_session_date=date.fromisoformat("2026-04-11"),
        target_ts=datetime.fromisoformat("2026-04-11T10:01:00+05:30"),
        pred_open=24020.0,
        pred_high=24150.0,
        pred_low=23950.0,
        pred_close=24110.0,
        direction="BUY",
        confidence=0.82,
        direction_prob_calibrated=0.88,
        confidence_score=0.82,
        confidence_bucket="high",
        pred_interval=SimpleNamespace(
            model_dump=lambda: {
                "close": {"p10": 23980.0, "p50": 24110.0, "p90": 24180.0},
            }
        ),
        model_family="meta_v3",
        calibration_version="cal_test",
        confidence_components={},
        model_version="meta_v3_test",
        feature_cutoff_ist=datetime.fromisoformat("2026-04-10T15:30:00+05:30"),
        generated_at=datetime.fromisoformat("2026-04-10T15:31:00+05:30"),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_signal_route_returns_actionable_buy(monkeypatch) -> None:
    monkeypatch.setattr("api.routes.signal.predict_single", lambda **_kwargs: _prediction())
    monkeypatch.setattr(
        "api.routes.signal.technical_context_for_symbol",
        lambda *_args, **_kwargs: {
            "open": 24010.0,
            "close": 24050.0,
            "ema_9": 24040.0,
            "ema_21": 24000.0,
            "ema_50": 23970.0,
            "ema_21_slope_3": 12.0,
            "rsi_14": 60.0,
            "macd": 1.2,
            "macd_signal": 0.7,
            "macd_hist": 0.5,
            "macd_hist_delta_1": 0.14,
            "body_pct_range": 0.52,
        },
    )
    out = signal(symbol="Nifty 50", interval="1m", prediction_mode="standard", db=_FakeDb([24000.0]))

    assert out.action == "BUY"
    assert out.latest_price == 24000.0
    assert out.predicted_price == 24110.0
    assert out.risk_reward_ratio is not None


def test_signal_route_falls_back_to_hold_for_weak_setup(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.routes.signal.predict_single",
        lambda **_kwargs: _prediction(
            direction="SELL",
            pred_close=23990.0,
            pred_high=24080.0,
            pred_low=23940.0,
            confidence_score=0.70,
            confidence_bucket="medium",
            pred_interval=SimpleNamespace(
                model_dump=lambda: {
                    "close": {"p10": 23950.0, "p50": 23990.0, "p90": 24090.0},
                }
            ),
        ),
    )
    monkeypatch.setattr("api.routes.signal.technical_context_for_symbol", lambda *_args, **_kwargs: None)
    out = signal(symbol="Nifty 50", interval="1minute", prediction_mode="standard", db=_FakeDb([24000.0]))

    assert out.action == "HOLD"
    assert any("Expected move too small" in r or "strategy confirmation" in r for r in out.reasons)
