import json
import sqlite3
import subprocess
from pathlib import Path
import uuid


def _create_test_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE raw_candles (id INTEGER PRIMARY KEY, interval TEXT)")
        conn.execute("CREATE TABLE option_trade_signals (id INTEGER PRIMARY KEY, interval TEXT)")
        conn.executemany("INSERT INTO raw_candles(interval) VALUES (?)", [("1minute",), ("30minute",), ("day",)])
        conn.executemany("INSERT INTO option_trade_signals(interval) VALUES (?)", [("1minute",), ("day",)])
        conn.commit()
    finally:
        conn.close()


def test_purge_script_dry_run_outputs_expected_shape() -> None:
    db_path = Path.cwd() / f"tmp_purge_test_{uuid.uuid4().hex}.db"
    try:
        _create_test_db(db_path)
        out = subprocess.run(
            ["python", "backend/scripts/purge_non_1m_data.py", "--db-path", str(db_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(out.stdout)
        assert payload["mode"] == "dry_run"
        assert ["1minute", 1] in payload["before"]["raw_candles"]
        assert "predictions_daily_total" not in payload["before"]
    finally:
        if db_path.exists():
            db_path.unlink()
