"""Consensus accuracy validation using the point-in-time signal backtest."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.signal_backtest import run_signal_backtest_for_symbols
from db.connection import get_db
from utils.config import get_settings


def main() -> None:
    db = get_db()
    settings = get_settings()
    symbols = [key.split("|", 1)[1] if "|" in key else key for key in settings.instrument_keys]
    results = run_signal_backtest_for_symbols(db, symbols)

    print("\n" + "=" * 80)
    print("CONSENSUS ACCURACY VALIDATION")
    print("=" * 80)
    for symbol, payload in results.items():
        consensus = payload.get("consensus_signal", {})
        print(f"\n{symbol}:")
        print(f"  Status: {payload.get('status')}")
        print(f"  Holdout Records: {payload.get('test_records', 0)}")
        print(f"  Total Signals: {consensus.get('total_signals', 0)}")
        print(f"  Accuracy: {consensus.get('accuracy', 0.0)}%")
        print(f"  Win Rate: {consensus.get('win_rate', 0.0)}%")
        print(f"  Total Move Points: {consensus.get('total_move_points', 0.0)}")
        print(f"  Thresholds: {payload.get('thresholds', {})}")
    db.close()


if __name__ == "__main__":
    main()
