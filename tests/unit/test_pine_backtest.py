from datetime import datetime

import pandas as pd

from backtesting.engine import run_pine_signal_backtest


class _FakeDb:
    pass


def test_pine_backtest_handles_missing_candles(monkeypatch) -> None:
    monkeypatch.setattr("backtesting.engine._load_intraday_frame", lambda *_args, **_kwargs: pd.DataFrame())
    out = run_pine_signal_backtest(_FakeDb(), symbol="Nifty 50")
    assert out["status"] == "no_candles"


def test_pine_backtest_closes_trade_on_opposite_signal(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "ts": [
                datetime.fromisoformat("2026-04-08T09:20:00+05:30"),
                datetime.fromisoformat("2026-04-08T09:21:00+05:30"),
                datetime.fromisoformat("2026-04-08T09:22:00+05:30"),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 103.0, 103.0],
            "low": [99.0, 100.0, 100.0],
            "close": [100.0, 102.0, 101.0],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )
    signal_frame = frame.copy()
    signal_frame["buy_signal"] = [True, False, False]
    signal_frame["sell_signal"] = [False, False, True]
    signal_frame["force_exit"] = [False, False, False]

    monkeypatch.setattr("backtesting.engine._load_intraday_frame", lambda *_args, **_kwargs: frame)
    monkeypatch.setattr("backtesting.engine._pine_signals", lambda _frame, **_kwargs: signal_frame)

    out = run_pine_signal_backtest(_FakeDb(), symbol="Nifty 50", initial_capital=100000.0)

    assert out["status"] == "ok"
    assert out["trades"] == 2
    assert out["wins"] == 1
    assert out["final_equity"] > 100000.0
