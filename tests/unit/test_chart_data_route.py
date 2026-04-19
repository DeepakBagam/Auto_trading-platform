from datetime import date, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.routes.data import _build_execution_status, chart_data


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


class _FakeDb:
    def __init__(self, candle_rows, scalar_values=None):
        self._candle_rows = candle_rows
        self._scalar_values = list(scalar_values or [])

    def execute(self, _query):
        return _FakeResult(self._candle_rows)

    def scalar(self, _query):
        if self._scalar_values:
            return self._scalar_values.pop(0)
        return None


def _candle(ts: str, close: float):
    return SimpleNamespace(
        ts=datetime.fromisoformat(ts),
        open=close - 10.0,
        high=close + 10.0,
        low=close - 20.0,
        close=close,
        volume=1000.0,
    )


def _prediction():
    return SimpleNamespace(
        symbol="SENSEX",
        interval="1minute",
        prediction_mode="standard",
        source_interval="1minute",
        target_session_date=date.fromisoformat("2026-04-08"),
        target_ts=datetime.fromisoformat("2026-04-08T09:16:00+05:30"),
        pred_open=74645.0,
        pred_high=74710.0,
        pred_low=74610.0,
        pred_close=74690.0,
        direction="BUY",
        confidence=0.71,
        confidence_score=0.71,
        confidence_bucket="medium",
        direction_prob_calibrated=0.71,
        pred_interval=None,
        model_family="intraday_v1",
        calibration_version=None,
        confidence_components={},
        model_version="intraday_v1_test",
        feature_cutoff_ist=datetime.fromisoformat("2026-04-08T09:15:00+05:30"),
        generated_at=datetime.fromisoformat("2026-04-08T09:15:10+05:30"),
    )


def test_chart_data_returns_hidden_predictions_with_execution_status(monkeypatch) -> None:
    fake_db = _FakeDb(
        [
            _candle("2026-04-08T09:15:00+05:30", 74623.35),
            _candle("2026-04-08T09:16:00+05:30", 74631.10),
        ],
        scalar_values=[
            74631.10,
            None,
            None,
        ],
    )
    monkeypatch.setattr(
        "api.routes.data.get_settings",
        lambda: SimpleNamespace(upstox_access_token=""),
    )
    monkeypatch.setattr(
        "api.routes.data._resolve_instrument_key",
        lambda db, symbol: ("NSE_INDEX|SENSEX", "SENSEX"),
    )
    monkeypatch.setattr("api.routes.data.predict_single", lambda **_kwargs: _prediction())

    payload = chart_data(
        symbol="SENSEX",
        interval="1minute",
        prediction_target_mode="standard",
        candles_limit=200,
        predictions_limit=50,
        include_historical_predictions=False,
        db=fake_db,
    )

    assert payload["interval"] == "1minute"
    assert payload["prediction_target_mode"] == "standard"
    assert payload["predicted"] == []
    assert "signal" in payload
    assert "markers" in payload
    assert "execution_status" in payload
    assert "indicator_matrix" in payload
    assert "analytics" in payload
    assert "model_variance" in payload
    assert "trade_intelligence" in payload
    assert "freshness" in payload
    assert payload["execution_status"]["consensus_state"] in {"missing_pine", "non_trade_signal"}


def test_chart_data_rejects_non_1m_interval(monkeypatch) -> None:
    fake_db = _FakeDb(
        [_candle("2026-04-08T09:15:00+05:30", 74623.35)],
        scalar_values=[74623.35, None, None],
    )
    monkeypatch.setattr(
        "api.routes.data.get_settings",
        lambda: SimpleNamespace(upstox_access_token=""),
    )
    monkeypatch.setattr(
        "api.routes.data._resolve_instrument_key",
        lambda db, symbol: ("NSE_INDEX|SENSEX", "SENSEX"),
    )
    monkeypatch.setattr("api.routes.data.predict_single", lambda **_kwargs: _prediction())

    with pytest.raises(HTTPException) as exc:
        chart_data(
            symbol="SENSEX",
            interval="30minute",
            prediction_target_mode="standard",
            candles_limit=200,
            predictions_limit=50,
            include_historical_predictions=False,
            db=fake_db,
        )
    assert exc.value.status_code == 422


def test_chart_data_keeps_confluence_markers_even_with_existing_markers(monkeypatch) -> None:
    fake_db = _FakeDb(
        [
            _candle("2026-04-08T09:15:00+05:30", 74623.35),
            _candle("2026-04-08T09:16:00+05:30", 74631.10),
        ],
        scalar_values=[74631.10, None, None],
    )
    monkeypatch.setattr("api.routes.data.get_settings", lambda: SimpleNamespace(upstox_access_token=""))
    monkeypatch.setattr("api.routes.data._resolve_instrument_key", lambda db, symbol: ("NSE_INDEX|SENSEX", "SENSEX"))
    monkeypatch.setattr("api.routes.data.predict_single", lambda **_kwargs: _prediction())
    monkeypatch.setattr("api.routes.data._build_trade_markers_from_positions", lambda *_args, **_kwargs: [{"time": "2026-04-08T09:15:00+05:30", "shape": "circle", "text": "ENTRY"}])
    monkeypatch.setattr("api.routes.data._build_signal_markers_from_logs", lambda *_args, **_kwargs: [{"time": "2026-04-08T09:16:00+05:30", "shape": "circle", "text": "SIG"}])
    monkeypatch.setattr("api.routes.data._build_confluence_markers_from_actual", lambda *_args, **_kwargs: [{"time": "2026-04-08T09:16:00+05:30", "shape": "arrowUp", "text": "PINE BUY 2TF"}])

    payload = chart_data(
        symbol="SENSEX",
        interval="1minute",
        prediction_target_mode="standard",
        candles_limit=200,
        predictions_limit=50,
        include_historical_predictions=False,
        db=fake_db,
    )

    marker_texts = {marker["text"] for marker in payload["markers"]}
    assert "ENTRY" in marker_texts
    assert "SIG" in marker_texts
    assert "PINE BUY 2TF" in marker_texts


def test_chart_data_converts_aware_utc_candle_timestamps_to_ist(monkeypatch) -> None:
    fake_db = _FakeDb(
        [
            _candle("2026-04-08T03:45:00+00:00", 74623.35),
            _candle("2026-04-08T03:46:00+00:00", 74631.10),
        ],
        scalar_values=[74631.10, None, None],
    )
    monkeypatch.setattr("api.routes.data.get_settings", lambda: SimpleNamespace(upstox_access_token=""))
    monkeypatch.setattr("api.routes.data._resolve_instrument_key", lambda db, symbol: ("NSE_INDEX|SENSEX", "SENSEX"))
    monkeypatch.setattr("api.routes.data.predict_single", lambda **_kwargs: _prediction())

    payload = chart_data(
        symbol="SENSEX",
        interval="1minute",
        prediction_target_mode="standard",
        candles_limit=200,
        predictions_limit=50,
        include_historical_predictions=False,
        db=fake_db,
    )

    actual_times = {row["x"] for row in payload["actual"]}
    assert "2026-04-08T09:15:00+05:30" in actual_times
    assert "2026-04-08T09:16:00+05:30" in actual_times


def test_execution_status_ignores_stale_signal_log() -> None:
    stale_log = SimpleNamespace(
        timestamp=datetime.fromisoformat("2026-04-08T09:16:00+05:30"),
        consensus="BUY",
        pine_signal="BUY",
        ml_signal="BUY",
        trade_placed=True,
        skip_reason=None,
    )
    fake_db = _FakeDb([], scalar_values=[stale_log, None])

    payload = _build_execution_status(
        fake_db,
        symbol="SENSEX",
        interval="1minute",
        candle_ts=datetime.fromisoformat("2026-04-17T15:29:00+05:30"),
        signal_action="BUY",
    )

    assert payload["consensus_state"] == "non_trade_signal"
    assert payload["audit_executed"] is False
