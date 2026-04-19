from datetime import date, datetime
from types import SimpleNamespace

from api.routes.predict import _prediction_is_stale, _select_intraday_prediction
from utils.constants import IST_ZONE


def _row(ts: str):
    return SimpleNamespace(target_ts=datetime.fromisoformat(ts), generated_at=datetime.now(IST_ZONE))


def test_select_intraday_prediction_prefers_future_bar_when_target_not_set() -> None:
    rows = [
        _row("2026-04-07T09:46:00+05:30"),
        _row("2026-04-07T09:47:00+05:30"),
        _row("2026-04-07T09:48:00+05:30"),
    ]
    now = datetime.fromisoformat("2026-04-07T09:46:30+05:30")
    selected = _select_intraday_prediction(rows, now=now, target_date=None)
    assert selected is not None
    assert selected.target_ts == datetime.fromisoformat("2026-04-07T09:47:00+05:30")


def test_select_intraday_prediction_honors_target_date() -> None:
    rows = [
        _row("2026-04-07T15:29:00+05:30"),
        _row("2026-04-08T09:15:00+05:30"),
    ]
    now = datetime.fromisoformat("2026-04-07T15:30:00+05:30")
    selected = _select_intraday_prediction(rows, now=now, target_date=date.fromisoformat("2026-04-08"))
    assert selected is not None
    assert selected.target_ts == datetime.fromisoformat("2026-04-08T09:15:00+05:30")


def test_prediction_is_stale_when_target_bar_is_past() -> None:
    row = SimpleNamespace(
        target_ts=datetime.fromisoformat("2026-04-07T09:15:00+05:30"),
        generated_at=datetime.fromisoformat("2026-04-07T09:10:00+05:30"),
        metadata_json={},
    )
    now = datetime.fromisoformat("2026-04-07T09:16:00+05:30")
    assert _prediction_is_stale(row, latest_candle_ts=None, now=now) is True


def test_prediction_is_stale_when_newer_candle_exists_than_prediction_source() -> None:
    row = SimpleNamespace(
        target_ts=datetime.fromisoformat("2026-04-08T09:15:00+05:30"),
        generated_at=datetime.fromisoformat("2026-04-07T15:29:00+05:30"),
        metadata_json={"source_candle_ts": "2026-04-07T15:29:00+05:30"},
    )
    now = datetime.fromisoformat("2026-04-07T15:29:30+05:30")
    latest_candle = datetime.fromisoformat("2026-04-07T15:30:00+05:30")
    assert _prediction_is_stale(row, latest_candle_ts=latest_candle, now=now) is True
