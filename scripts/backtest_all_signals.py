"""
Point-in-time backtest for ML, Pine, AI, and consensus signals.

This script uses stored intraday predictions plus later realized candles.
It does not synthesize predictions from future prices.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from backtesting.signal_backtest import run_signal_backtest_suite
from db.connection import get_db
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _print_signal_block(name: str, payload: dict) -> None:
    print(f"\n{name}:")
    print(f"  Total Signals: {payload['total_signals']}")
    print(f"  Buy Signals: {payload['buy_signals']}")
    print(f"  Sell Signals: {payload['sell_signals']}")
    print(f"  Accuracy: {payload['accuracy']}%")
    print(f"  Win Rate: {payload['win_rate']}%")
    print(f"  Avg Move Points: {payload['avg_move_points']}")
    print(f"  Total Move Points: {payload['total_move_points']}")
    print(f"  Max Drawdown: {payload.get('max_drawdown_pct', 0.0)}%")


def main() -> None:
    db = get_db()
    settings = get_settings()
    symbols = [key.split("|", 1)[1] if "|" in key else key for key in settings.instrument_keys]
    if not symbols:
        raise RuntimeError("No configured symbols found in UPSTOX_INSTRUMENT_KEYS.")

    suite = run_signal_backtest_suite(db, symbols)
    results = suite["symbols"]
    print("\n" + "=" * 90)
    print("POINT-IN-TIME SIGNAL BACKTEST")
    print("=" * 90)
    print("Method: stored intraday predictions + later realized candles")
    print("Optimization: iterative realism tuning on train split, reported on holdout split")

    print("\nSuite Summary:")
    print(json.dumps(suite["summary"], indent=2))

    for symbol, payload in results.items():
        print("\n" + "-" * 90)
        print(f"{symbol}")
        print("-" * 90)
        if payload.get("status") != "ok":
            print(json.dumps(payload, indent=2))
            continue
        print(f"Train Records: {payload['train_records']}")
        print(f"Test Records: {payload['test_records']}")
        print(f"Window: {payload['date_from']} -> {payload['date_to']}")
        print(f"Thresholds: {json.dumps(payload['thresholds'])}")
        if payload.get("tuning_history"):
            last = payload["tuning_history"][-1]
            print(f"Tuning Iterations: {len(payload['tuning_history'])}")
            print(f"Last Train Realism Score: {last.get('realism_score')}")
        _print_signal_block("ML Signal", payload["ml_signal"])
        _print_signal_block("Pine Signal", payload["pine_signal"])
        _print_signal_block("AI Signal", payload["ai_signal"])
        _print_signal_block("Consensus Signal", payload["consensus_signal"])

    db.close()


if __name__ == "__main__":
    main()
