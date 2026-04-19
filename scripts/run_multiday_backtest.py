"""Multi-day backtest runner with validation."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from backtesting.enhanced_engine import run_multi_day_backtest
from db.connection import SessionLocal
from utils.config import get_settings
from utils.constants import IST_ZONE


def validate_results(results: dict) -> dict:
    """Validate backtest results against production criteria."""
    validation = {
        \"passed\": True,
        \"checks\": [],
    }
    
    trades_per_session = results.get(\"trades_per_session\", 0)
    avg_win_rate = results.get(\"avg_win_rate_pct\", 0)
    exit_reasons = results.get(\"exit_reasons\", {})
    total_trades = results.get(\"total_trades\", 0)
    
    # Check 1: Trades per session >= 8
    if trades_per_session >= 8:
        validation[\"checks\"].append({
            \"name\": \"Trades per session\",
            \"status\": \"PASS\",
            \"value\": trades_per_session,
            \"threshold\": \">=8\",
        })
    else:
        validation[\"checks\"].append({
            \"name\": \"Trades per session\",
            \"status\": \"FAIL\",
            \"value\": trades_per_session,
            \"threshold\": \">=8\",
        })
        validation[\"passed\"] = False
    
    # Check 2: Win rate 55-65%
    if 55 <= avg_win_rate <= 65:
        validation[\"checks\"].append({
            \"name\": \"Win rate\",
            \"status\": \"PASS\",
            \"value\": f\"{avg_win_rate:.1f}%\",
            \"threshold\": \"55-65%\",
        })
    else:
        validation[\"checks\"].append({
            \"name\": \"Win rate\",
            \"status\": \"WARNING\" if 50 <= avg_win_rate < 70 else \"FAIL\",
            \"value\": f\"{avg_win_rate:.1f}%\",
            \"threshold\": \"55-65%\",
        })
        if avg_win_rate < 50 or avg_win_rate > 70:
            validation[\"passed\"] = False
    
    # Check 3: Exit distribution (EOD should not dominate)
    eod_exits = exit_reasons.get(\"EOD\", 0) + exit_reasons.get(\"FORCE_EXIT\", 0)
    eod_pct = (eod_exits / total_trades * 100) if total_trades > 0 else 0
    
    if eod_pct < 50:
        validation[\"checks\"].append({
            \"name\": \"EOD exit percentage\",
            \"status\": \"PASS\",
            \"value\": f\"{eod_pct:.1f}%\",
            \"threshold\": \"<50%\",
        })
    else:
        validation[\"checks\"].append({
            \"name\": \"EOD exit percentage\",
            \"status\": \"WARNING\",
            \"value\": f\"{eod_pct:.1f}%\",
            \"threshold\": \"<50%\",
        })
    
    # Check 4: Proper exit distribution
    sl_exits = exit_reasons.get(\"SL\", 0)
    tsl_exits = exit_reasons.get(\"TSL\", 0)
    target_exits = exit_reasons.get(\"TARGET\", 0)
    proper_exits = sl_exits + tsl_exits + target_exits
    proper_exit_pct = (proper_exits / total_trades * 100) if total_trades > 0 else 0
    
    if proper_exit_pct >= 30:
        validation[\"checks\"].append({
            \"name\": \"Proper exit usage\",
            \"status\": \"PASS\",
            \"value\": f\"{proper_exit_pct:.1f}%\",
            \"threshold\": \">=30%\",
        })
    else:
        validation[\"checks\"].append({
            \"name\": \"Proper exit usage\",
            \"status\": \"FAIL\",
            \"value\": f\"{proper_exit_pct:.1f}%\",
            \"threshold\": \">=30%\",
        })
        validation[\"passed\"] = False
    
    # Check 5: Positive PnL across symbols
    positive_symbols = 0
    total_symbols = 0
    for symbol, result in results.get(\"symbol_results\", {}).items():
        if result.get(\"status\") == \"ok\":
            total_symbols += 1
            if result.get(\"gross_return_pct\", 0) > 0:
                positive_symbols += 1
    
    if positive_symbols >= total_symbols * 0.6:
        validation[\"checks\"].append({
            \"name\": \"Profitable symbols\",
            \"status\": \"PASS\",
            \"value\": f\"{positive_symbols}/{total_symbols}\",
            \"threshold\": \">=60%\",
        })
    else:
        validation[\"checks\"].append({
            \"name\": \"Profitable symbols\",
            \"status\": \"WARNING\",
            \"value\": f\"{positive_symbols}/{total_symbols}\",
            \"threshold\": \">=60%\",
        })
    
    return validation


def render_report(results: dict, validation: dict) -> str:
    \"\"\"Render markdown report.\"\"\"
    lines = [
        \"# Multi-Day Backtest Report\",
        \"\",
        f\"**Period**: {results['start_date']} to {results['end_date']}\",
        f\"**Generated**: {datetime.now(IST_ZONE).isoformat()}\",
        \"\",
        \"## Overall Performance\",
        \"\",
        f\"- Total Trades: {results['total_trades']}\",
        f\"- Trades per Session: {results['trades_per_session']:.1f}\",
        f\"- Win Rate: {results['avg_win_rate_pct']:.1f}%\",
        f\"- Wins: {results['total_wins']}\",
        f\"- Losses: {results['total_losses']}\",
        \"\",
        \"## Exit Distribution\",
        \"\",
        \"| Exit Reason | Count | Percentage |\",
        \"| --- | ---: | ---: |\",
    ]
    
    total = results[\"total_trades\"]
    for reason, count in sorted(results[\"exit_reasons\"].items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total > 0 else 0
        lines.append(f\"| {reason} | {count} | {pct:.1f}% |\")
    
    lines.extend([
        \"\",
        \"## Symbol Performance\",
        \"\",
        \"| Symbol | Trades | Win Rate | Return | Max DD | Sharpe |\",
        \"| --- | ---: | ---: | ---: | ---: | ---: |\",
    ])
    
    for symbol, result in results[\"symbol_results\"].items():
        if result.get(\"status\") == \"ok\":
            lines.append(
                f\"| {symbol} | {result['trades']} | {result['win_rate_pct']:.1f}% | \"\n                f\"{result['gross_return_pct']:+.2f}% | {result['max_drawdown_pct']:.2f}% | \"\n                f\"{result['sharpe']:.2f} |\"\n            )
    
    lines.extend([
        \"\",
        \"## Validation Results\",
        \"\",
        f\"**Status**: {'✅ PASSED' if validation['passed'] else '❌ FAILED'}\",
        \"\",
        \"| Check | Status | Value | Threshold |\",
        \"| --- | --- | --- | --- |\",
    ])
    
    for check in validation[\"checks\"]:
        status_icon = \"✅\" if check[\"status\"] == \"PASS\" else (\"⚠️\" if check[\"status\"] == \"WARNING\" else \"❌\")
        lines.append(
            f\"| {check['name']} | {status_icon} {check['status']} | {check['value']} | {check['threshold']} |\"\n        )
    
    lines.append(\"\")
    return \"\\n\".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=\"Run multi-day backtest with validation\")
    parser.add_argument(\"--start-date\", required=True, help=\"Start date (YYYY-MM-DD)\")
    parser.add_argument(\"--end-date\", required=True, help=\"End date (YYYY-MM-DD)\")
    parser.add_argument(\"--symbols\", nargs=\"*\", default=None, help=\"Symbols to test\")
    parser.add_argument(\"--output-dir\", default=\"logs/backtests\", help=\"Output directory\")
    args = parser.parse_args()
    
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    
    settings = get_settings()
    symbols = args.symbols or list(settings.execution_symbol_list or [\"Nifty 50\", \"Bank Nifty\", \"SENSEX\"])
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f\"Running backtest from {start_date} to {end_date}...\")
    print(f\"Symbols: {', '.join(symbols)}\")
    
    db = SessionLocal()
    try:
        results = run_multi_day_backtest(
            db,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        db.close()
    
    validation = validate_results(results)
    
    # Save results
    timestamp = datetime.now(IST_ZONE).strftime(\"%Y%m%d_%H%M%S\")
    json_path = output_dir / f\"multiday_backtest_{start_date}_{end_date}_{timestamp}.json\"
    md_path = output_dir / f\"multiday_backtest_{start_date}_{end_date}_{timestamp}.md\"
    
    results[\"validation\"] = validation
    json_path.write_text(json.dumps(results, indent=2), encoding=\"utf-8\")
    md_path.write_text(render_report(results, validation), encoding=\"utf-8\")
    
    print(f\"\\n{'='*60}\")
    print(\"BACKTEST COMPLETE\")
    print(f\"{'='*60}\")
    print(f\"Total Trades: {results['total_trades']}\")
    print(f\"Trades/Session: {results['trades_per_session']:.1f}\")
    print(f\"Win Rate: {results['avg_win_rate_pct']:.1f}%\")
    print(f\"\\nValidation: {'✅ PASSED' if validation['passed'] else '❌ FAILED'}\")
    print(f\"\\nReports saved:\")
    print(f\"  JSON: {json_path}\")
    print(f\"  MD:   {md_path}\")


if __name__ == \"__main__\":
    main()
