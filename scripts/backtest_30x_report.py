from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from backtesting.signal_backtest import _actual_direction, _point_in_time_dataset, _row_technical_context, run_signal_backtest
from db.connection import SessionLocal
from execution_engine.ai_intelligence import score_trade_intelligence
from prediction_engine.signal_engine import actionable_direction_from_forecast
from utils.config import get_settings
from utils.constants import IST_ZONE


@dataclass(frozen=True, slots=True)
class SweepCombo:
    min_confidence: float
    min_move_pct: float
    min_ai_score: float


def _combo_grid() -> list[SweepCombo]:
    return [
        SweepCombo(min_confidence=min_conf, min_move_pct=min_move_pct, min_ai_score=min_ai)
        for min_conf, min_move_pct, min_ai in product(
            [0.10, 0.15, 0.20, 0.25, 0.30],
            [0.0004, 0.0008, 0.0012],
            [30.0, 40.0],
        )
    ]


def _precompute_rows(dataset) -> list[dict]:
    cached_rows: list[dict] = []
    for row in dataset.itertuples(index=False):
        row_dict = row._asdict()
        latest_price = float(row_dict["ref_close"])
        predicted_price = float(row_dict["pred_close"])
        technical_context = _row_technical_context(row_dict)
        forecast_action = actionable_direction_from_forecast(
            direction=str(row_dict["direction"]),
            latest_price=latest_price,
            predicted_price=predicted_price,
            technical_context=technical_context,
        )
        if forecast_action not in {"BUY", "SELL"}:
            continue

        ai = score_trade_intelligence(
            signal_action=forecast_action,
            confidence=float(row_dict["confidence"]),
            expected_return_pct=(predicted_price - latest_price) / max(abs(latest_price), 1e-9),
            technical_context=technical_context,
            now=row_dict["ref_ts"] if isinstance(row_dict["ref_ts"], datetime) else datetime.now(IST_ZONE),
        )
        _, actual_move_points = _actual_direction(row_dict)
        cached_rows.append(
            {
                "confidence": float(row_dict["confidence"]),
                "move_pct": abs(predicted_price - latest_price) / max(abs(latest_price), 1e-9),
                "ai_score": float(ai.score),
                "forecast_action": forecast_action,
                "actual_move_points": float(actual_move_points),
            }
        )
    return cached_rows


def _sweep_symbol(symbol: str, dataset, combos: list[SweepCombo]) -> dict:
    if dataset.empty:
        return {
            "symbol": symbol,
            "status": "no_backtest_data",
            "rows": 0,
            "combos_evaluated": len(combos),
            "results": [],
        }

    cached_rows = _precompute_rows(dataset)
    results: list[dict] = []
    for index, combo in enumerate(combos, start=1):
        trades = 0
        wins = 0
        losses = 0
        total_points = 0.0
        buy_trades = 0
        sell_trades = 0

        for row in cached_rows:
            if float(row["confidence"]) < combo.min_confidence or float(row["move_pct"]) < combo.min_move_pct:
                continue
            if float(row["ai_score"]) < combo.min_ai_score:
                continue

            pnl_points = (
                float(row["actual_move_points"])
                if row["forecast_action"] == "BUY"
                else -float(row["actual_move_points"])
            )
            trades += 1
            total_points += pnl_points
            if row["forecast_action"] == "BUY":
                buy_trades += 1
            else:
                sell_trades += 1
            if pnl_points > 0:
                wins += 1
            else:
                losses += 1

        win_rate = (wins / trades * 100.0) if trades else 0.0
        avg_points = (total_points / trades) if trades else 0.0
        results.append(
            {
                "combo_id": index,
                "combo": asdict(combo),
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "win_rate_pct": round(win_rate, 2),
                "avg_points": round(avg_points, 2),
                "total_points": round(total_points, 2),
            }
        )

    ranked = sorted(
        results,
        key=lambda item: (
            float(item["win_rate_pct"]),
            float(item["total_points"]),
            int(item["trades"]),
        ),
        reverse=True,
    )
    return {
        "symbol": symbol,
        "status": "ok",
        "rows": int(len(dataset)),
        "combos_evaluated": len(combos),
        "results": ranked,
    }


def _qualified_best(results: list[dict], min_trades: int) -> dict | None:
    for row in results:
        if int(row["trades"]) >= min_trades:
            return row
    return None


def _format_combo_label(combo: dict) -> str:
    return (
        f"conf>={float(combo['min_confidence']):.2f}, "
        f"move>={float(combo['min_move_pct']) * 100.0:.2f}%, "
        f"ai>={float(combo['min_ai_score']):.0f}"
    )


def _render_markdown(report: dict) -> str:
    lines = [
        "# 30x Backtest Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Symbols: {', '.join(report['symbols'])}",
        f"- Combos evaluated per symbol: {report['combos_per_symbol']}",
        f"- Qualified trade floor: {report['qualified_trade_floor']}",
        "",
    ]
    if report["production_baseline"]:
        lines.extend(
            [
                "## Production Baseline",
                "",
                "| Symbol | Trades | Win Rate | Total Move Points | Status |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for symbol, payload in report["production_baseline"].items():
            consensus = payload.get("consensus_signal", {})
            lines.append(
                f"| {symbol} | {consensus.get('total_signals', 0)} | {consensus.get('win_rate', 0.0)}% | "
                f"{consensus.get('total_move_points', 0.0)} | {payload.get('status', '-') } |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "## Production Baseline",
                "",
                "Skipped for this run. Use `--include-production-baseline` to execute the slower strict baseline.",
                "",
            ]
        )

    lines.extend(
        [
            "## 30-Combo Sweep",
            "",
            "| Symbol | Best Raw Win Rate | Best Qualified Win Rate | Notes |",
            "| --- | ---: | ---: | --- |",
        ]
    )

    for symbol, payload in report["sweep"].items():
        if payload.get("status") != "ok":
            lines.append(f"| {symbol} | - | - | no_backtest_data |")
            continue
        best_raw = payload["results"][0] if payload["results"] else None
        best_qualified = payload.get("best_qualified")
        notes: list[str] = []
        if best_raw is not None:
            notes.append(f"raw {_format_combo_label(best_raw['combo'])}")
        if best_qualified is not None:
            notes.append(f"qualified {_format_combo_label(best_qualified['combo'])}")
        lines.append(
            f"| {symbol} | "
            f"{best_raw['win_rate_pct'] if best_raw else 0.0}% ({best_raw['trades'] if best_raw else 0} trades) | "
            f"{best_qualified['win_rate_pct'] if best_qualified else 0.0}% ({best_qualified['trades'] if best_qualified else 0} trades) | "
            f"{' ; '.join(notes) if notes else '-'} |"
        )

    top_aggregate = report.get("top_aggregate_combos", [])
    if top_aggregate:
        lines.extend(
            [
                "",
                "## Top Aggregate Combos",
                "",
                "| Rank | Combo | Trades | Win Rate | Total Points |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for rank, row in enumerate(top_aggregate, start=1):
            lines.append(
                f"| {rank} | {_format_combo_label(row['combo'])} | {row['trades']} | "
                f"{row['win_rate_pct']}% | {row['total_points']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 30-combo intraday backtest sweep and write a report.")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Override symbol list. Defaults to configured execution symbols.",
    )
    parser.add_argument(
        "--qualified-trade-floor",
        type=int,
        default=30,
        help="Minimum trade count for the qualified best-win-rate result.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/backtests",
        help="Directory where JSON and Markdown reports should be written.",
    )
    parser.add_argument(
        "--include-production-baseline",
        action="store_true",
        help="Run the slower strict production baseline in addition to the 30-combo sweep.",
    )
    args = parser.parse_args()

    settings = get_settings()
    symbols = list(args.symbols or settings.execution_symbol_list or ["Nifty 50", "Bank Nifty", "SENSEX"])
    combos = _combo_grid()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(IST_ZONE)

    db = SessionLocal()
    try:
        production_baseline = (
            {symbol: run_signal_backtest(db, symbol) for symbol in symbols}
            if args.include_production_baseline
            else {}
        )
        sweep: dict[str, dict] = {}
        aggregate_rows: list[dict] = []
        for combo_index, combo in enumerate(combos, start=1):
            aggregate_rows.append(
                {
                    "combo_id": combo_index,
                    "combo": asdict(combo),
                    "trades": 0,
                    "wins": 0,
                    "total_points": 0.0,
                }
            )

        for symbol in symbols:
            dataset = _point_in_time_dataset(db, symbol)
            payload = _sweep_symbol(symbol, dataset, combos)
            if payload.get("status") == "ok":
                payload["best_qualified"] = _qualified_best(payload["results"], int(args.qualified_trade_floor))
                for row in payload["results"]:
                    agg = aggregate_rows[int(row["combo_id"]) - 1]
                    agg["trades"] += int(row["trades"])
                    agg["wins"] += int(row["wins"])
                    agg["total_points"] += float(row["total_points"])
            sweep[symbol] = payload

        top_aggregate: list[dict] = []
        for row in aggregate_rows:
            trades = int(row["trades"])
            wins = int(row["wins"])
            win_rate = (wins / trades * 100.0) if trades else 0.0
            top_aggregate.append(
                {
                    "combo_id": row["combo_id"],
                    "combo": row["combo"],
                    "trades": trades,
                    "wins": wins,
                    "win_rate_pct": round(win_rate, 2),
                    "total_points": round(float(row["total_points"]), 2),
                }
            )
        top_aggregate.sort(
            key=lambda item: (
                float(item["win_rate_pct"]),
                float(item["total_points"]),
                int(item["trades"]),
            ),
            reverse=True,
        )

        report = {
            "generated_at": generated_at.isoformat(),
            "symbols": symbols,
            "combos_per_symbol": len(combos),
            "qualified_trade_floor": int(args.qualified_trade_floor),
            "production_baseline": production_baseline,
            "sweep": sweep,
            "top_aggregate_combos": top_aggregate[:10],
        }

        stamp = generated_at.strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"backtest_30x_report_{stamp}.json"
        md_path = output_dir / f"backtest_30x_report_{stamp}.md"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text(_render_markdown(report), encoding="utf-8")
        print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
