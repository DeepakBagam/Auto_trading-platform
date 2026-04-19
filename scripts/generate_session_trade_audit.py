from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from backtesting.engine import run_pine_signal_backtest
from db.connection import SessionLocal
from utils.config import get_settings
from utils.constants import IST_ZONE


def _default_symbols() -> list[str]:
    settings = get_settings()
    return list(settings.execution_symbol_list or ["Nifty 50", "Bank Nifty", "SENSEX", "India VIX"])


def _render_markdown(report: dict) -> str:
    lines = [
        "# Session Trade Audit",
        "",
        f"- Trade date: {report['trade_date']}",
        f"- Generated at: {report['generated_at']}",
        "",
        "## Direction To Option Mapping",
        "",
        "- Underlying `BUY` direction maps to `BUY CE` in the live/paper execution engine.",
        "- Underlying `SELL` direction maps to `BUY PE` in the live/paper execution engine.",
        "- This audit replays underlying Pine-style directional entries and exits; it does not reconstruct historical option-premium fills.",
        "",
        "## Session Summary",
        "",
        "| Symbol | Trades | Wins | Losses | Win Rate | Gross Return | Final Equity | Max DD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for symbol, payload in report["symbols"].items():
        lines.append(
            f"| {symbol} | {payload.get('trades', 0)} | {payload.get('wins', 0)} | {payload.get('losses', 0)} | "
            f"{payload.get('win_rate_pct', 0.0)}% | {payload.get('gross_return_pct', 0.0)}% | "
            f"{payload.get('final_equity', 0.0)} | {payload.get('max_drawdown_pct', 0.0)}% |"
        )

    for symbol, payload in report["symbols"].items():
        lines.extend(
            [
                "",
                f"## {symbol}",
                "",
                f"- Signals: {payload.get('signals', 0)}",
                f"- Trades: {payload.get('trades', 0)}",
                f"- Win rate: {payload.get('win_rate_pct', 0.0)}%",
                f"- Gross return: {payload.get('gross_return_pct', 0.0)}%",
                "",
            ]
        )
        trade_log = list(payload.get("trade_log") or [])
        if not trade_log:
            lines.append("No trades were generated for this symbol on the requested date.")
            continue
        lines.extend(
            [
                "| Entry Time | Exit Time | Side | Engine Option Mapping | Entry | Exit | PnL Points | Return % | Exit Reason |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for trade in trade_log:
            side = str(trade.get("side") or "-")
            option_mapping = "BUY CE" if side == "BUY" else ("BUY PE" if side == "SELL" else "-")
            lines.append(
                f"| {trade.get('entry_time', '-')} | {trade.get('exit_time', '-')} | {side} | {option_mapping} | "
                f"{trade.get('entry_price', '-')} | {trade.get('exit_price', '-')} | {trade.get('pnl_points', '-')} | "
                f"{trade.get('return_pct', '-')} | {trade.get('exit_reason', '-')} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a one-session Pine trade audit report.")
    parser.add_argument("--date", required=True, help="Trade date in YYYY-MM-DD format")
    parser.add_argument("--symbols", nargs="*", default=None, help="Override symbol list")
    parser.add_argument("--output-dir", default="logs/backtests", help="Directory for the audit report")
    args = parser.parse_args()

    trade_date = date.fromisoformat(str(args.date))
    symbols = list(args.symbols or _default_symbols())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(IST_ZONE)

    report = {
        "trade_date": trade_date.isoformat(),
        "generated_at": generated_at.isoformat(),
        "symbols": {},
    }

    db = SessionLocal()
    try:
        for symbol in symbols:
            report["symbols"][symbol] = run_pine_signal_backtest(
                db,
                symbol=symbol,
                start_date=trade_date,
                end_date=trade_date,
            )
    finally:
        db.close()

    stamp = generated_at.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"session_trade_audit_{trade_date.isoformat()}_{stamp}.json"
    md_path = output_dir / f"session_trade_audit_{trade_date.isoformat()}_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
