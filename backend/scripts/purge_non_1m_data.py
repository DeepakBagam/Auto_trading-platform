try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import argparse
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


def _count_by_interval(conn: sqlite3.Connection, table: str) -> list[tuple[str, int]]:
    rows = conn.execute(
        "SELECT interval, COUNT(*) AS c FROM " + table + " GROUP BY interval ORDER BY interval"
    ).fetchall()
    return [(str(r[0]), int(r[1])) for r in rows]


def _count_total(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute("SELECT COUNT(*) FROM " + table).fetchone()
    return int((row or [0])[0] or 0)


def _snapshot_counts(conn: sqlite3.Connection) -> dict:
    out: dict[str, object] = {}
    out["raw_candles"] = _count_by_interval(conn, "raw_candles")
    out["predictions_intraday"] = _count_by_interval(conn, "predictions_intraday")
    out["predictions_daily_total"] = _count_total(conn, "predictions_daily")
    out["option_trade_signals"] = _count_by_interval(conn, "option_trade_signals")
    return out


def _create_backup(db_path: Path, backup_path: Path) -> None:
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(db_path, backup_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Purge non-1minute interval rows from SQLite DB.")
    parser.add_argument("--db-path", default="trading.db", help="SQLite database path")
    parser.add_argument("--apply", action="store_true", help="Apply deletions (default is dry-run)")
    parser.add_argument("--backup-path", default="", help="Backup file path used with --apply")
    parser.add_argument("--no-vacuum", action="store_true", help="Skip VACUUM after apply")
    args = parser.parse_args()

    db_path = Path(args.db_path).resolve()
    if not db_path.exists():
        raise RuntimeError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        before = _snapshot_counts(conn)
        if not args.apply:
            print(
                json.dumps(
                    {
                        "mode": "dry_run",
                        "db_path": str(db_path),
                        "before": before,
                        "planned_deletes": {
                            "raw_candles_non_1m": "DELETE interval != '1minute'",
                            "predictions_intraday_non_1m": "DELETE interval != '1minute'",
                            "predictions_daily_all": "DELETE all rows",
                            "option_trade_signals_non_1m": "DELETE interval != '1minute'",
                        },
                    },
                    indent=2,
                )
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = (
            Path(args.backup_path).resolve()
            if args.backup_path.strip()
            else db_path.with_suffix(db_path.suffix + f".backup_{timestamp}")
        )
        _create_backup(db_path, backup_path)

        deleted: dict[str, int] = {}
        deleted["raw_candles_non_1m"] = conn.execute(
            "DELETE FROM raw_candles WHERE interval <> '1minute'"
        ).rowcount
        deleted["predictions_intraday_non_1m"] = conn.execute(
            "DELETE FROM predictions_intraday WHERE interval <> '1minute'"
        ).rowcount
        deleted["predictions_daily_all"] = conn.execute("DELETE FROM predictions_daily").rowcount
        deleted["option_trade_signals_non_1m"] = conn.execute(
            "DELETE FROM option_trade_signals WHERE interval <> '1minute'"
        ).rowcount
        conn.commit()

        if not args.no_vacuum:
            conn.execute("VACUUM")

        after = _snapshot_counts(conn)
        print(
            json.dumps(
                {
                    "mode": "apply",
                    "db_path": str(db_path),
                    "backup_path": str(backup_path),
                    "deleted_rows": deleted,
                    "before": before,
                    "after": after,
                    "vacuum": (not args.no_vacuum),
                },
                indent=2,
            )
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
