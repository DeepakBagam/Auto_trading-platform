"""Quick entrypoint for the point-in-time signal backtest."""

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
    print(results)
    db.close()


if __name__ == "__main__":
    main()
