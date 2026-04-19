from datetime import date, datetime
from types import SimpleNamespace

from api.routes.options import options_signal


class _FakeDb:
    def scalar(self, _query):
        return None

    def add(self, _row) -> None:
        return None

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def execute(self, _query):
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))


def _prediction(**overrides):
    base = dict(
        symbol="Nifty 50",
        interval="1minute",
        prediction_mode="session_close",
        source_interval="1minute",
        target_session_date=date.fromisoformat("2026-04-11"),
        target_ts=datetime.fromisoformat("2026-04-11T15:30:00+05:30"),
        pred_open=24100.0,
        pred_high=24210.0,
        pred_low=24060.0,
        pred_close=24190.0,
        direction="BUY",
        confidence=0.72,
        direction_prob_calibrated=0.72,
        confidence_score=0.72,
        confidence_bucket="medium",
        pred_interval=None,
        model_family="intraday_v1",
        calibration_version=None,
        confidence_components={},
        model_version="intraday_v1_test",
        feature_cutoff_ist=datetime.fromisoformat("2026-04-11T15:28:00+05:30"),
        generated_at=datetime.fromisoformat("2026-04-11T15:28:30+05:30"),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_options_signal_returns_buy_ce_with_risk_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.routes.options.get_settings",
        lambda: SimpleNamespace(upstox_access_token="", instrument_keys=[]),
    )
    monkeypatch.setattr("api.routes.options.predict_single", lambda **_kwargs: _prediction())
    monkeypatch.setattr("api.routes.options.latest_price_for_symbol", lambda *_args, **_kwargs: 24125.0)
    monkeypatch.setattr("api.routes.options.technical_context_for_symbol", lambda *_args, **_kwargs: {})

    out = options_signal(
        symbol="Nifty 50",
        interval="1minute",
        prediction_mode="session_close",
        expiry_date=None,
        strike_mode="auto",
        manual_strike=None,
        allow_option_writing=False,
        db=_FakeDb(),
    )

    assert out.symbol == "Nifty 50"
    assert out.option_signal.action in {"BUY", "HOLD"}
    assert out.trade_intelligence is not None
    if out.option_signal.action == "BUY":
        assert out.option_signal.option_type == "CE"
        assert out.option_signal.stop_loss is not None
        assert out.option_signal.trailing_stop_loss is not None
    assert len(out.chain) > 0
