"""Backtest the independent intraday scalping strategy.

The existing directional strategy is not imported or modified here. This script
tests a separate 1-minute scalper with 5-15 minute exits, fixed point stops, a
cooldown, and a daily trade cap.

Usage:
    python backend/scripts/backtest_scalping_strategy.py --symbol "Nifty 50" --days 60
    python backend/scripts/backtest_scalping_strategy.py --symbol "Nifty 50" --days 120 --optimize
"""
try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import and_, select

from backend.db.connection import SessionLocal
from backend.db.models import RawCandle
from backend.execution_engine.scalping_strategy import (
    ScalpSignal,
    ScalpingConfig,
    build_scalping_features,
    detect_scalp_signal,
)
from backend.utils.constants import IST_ZONE
from backend.utils.symbols import instrument_key_filter


@dataclass(slots=True)
class ScalpTrade:
    trade_id: int
    date: date
    entry_ts: datetime
    action: str
    setup: str
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_ts: datetime
    exit_price: float
    exit_reason: str
    pnl_points: float
    holding_minutes: int
    mfe_points: float
    mae_points: float
    is_win: bool
    score: float


def _as_ist(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(IST_ZONE) if value.tzinfo else value.replace(tzinfo=IST_ZONE)
    out = pd.Timestamp(value).to_pydatetime()
    return out.astimezone(IST_ZONE) if out.tzinfo else out.replace(tzinfo=IST_ZONE)


def _load_candles(symbol: str, days: int) -> pd.DataFrame:
    cutoff = datetime.now(IST_ZONE) - timedelta(days=days + 5)
    db = SessionLocal()
    try:
        rows = (
            db.execute(
                select(RawCandle)
                .where(
                    and_(
                        instrument_key_filter(RawCandle.instrument_key, symbol),
                        RawCandle.interval == "1minute",
                        RawCandle.ts >= cutoff,
                    )
                )
                .order_by(RawCandle.ts.asc())
            )
            .scalars()
            .all()
        )
    finally:
        db.close()
    if not rows:
        raise ValueError(f"No 1-minute candles found for symbol '{symbol}'")
    frame = pd.DataFrame(
        {
            "ts": [_as_ist(row.ts) for row in rows],
            "open": [float(row.open) for row in rows],
            "high": [float(row.high) for row in rows],
            "low": [float(row.low) for row in rows],
            "close": [float(row.close) for row in rows],
            "volume": [float(row.volume or 0.0) for row in rows],
        }
    )
    frame = frame.drop_duplicates(subset=["ts"]).sort_values("ts")
    frame.set_index("ts", inplace=True)
    return frame


def _move(action: str, entry: float, price: float) -> float:
    mult = 1 if action == "BUY" else -1
    return round((price - entry) * mult, 2)


def _price_from_offset(action: str, entry: float, points: float) -> float:
    mult = 1 if action == "BUY" else -1
    return round(entry + (mult * points), 2)


def simulate_scalps(
    candles: pd.DataFrame,
    features: pd.DataFrame,
    config: ScalpingConfig,
) -> list[ScalpTrade]:
    trades: list[ScalpTrade] = []
    trades_today: dict[date, int] = {}
    pnl_today: dict[date, float] = {}
    consecutive_losses = 0
    last_exit_ts: datetime | None = None
    loss_pause_until: datetime | None = None
    current_trade_date: date | None = None
    i = config.warmup_bars
    trade_id = 0

    while i < len(features) - 2:
        ts = _as_ist(features.index[i])
        if current_trade_date != ts.date():
            current_trade_date = ts.date()
            consecutive_losses = 0
            loss_pause_until = None
        if last_exit_ts is not None and ts < last_exit_ts + timedelta(minutes=config.cooldown_minutes):
            i += 1
            continue
        if loss_pause_until is not None and ts < loss_pause_until:
            i += 1
            continue
        if pnl_today.get(ts.date(), 0.0) <= -(config.stop_points * 2.0):
            i += 1
            continue
        if consecutive_losses >= 3:
            i += 1
            continue
        signal = detect_scalp_signal(
            features.iloc[: i + 1],
            now=ts,
            trades_today=trades_today.get(ts.date(), 0),
            last_exit_ts=last_exit_ts,
            config=config,
        )
        if signal.action not in {"BUY", "SELL"}:
            i += 1
            continue

        trade, exit_idx = _simulate_one_trade(candles, i, signal, config, trade_id + 1)
        if trade is None:
            break
        trade_id += 1
        trades.append(trade)
        trades_today[trade.date] = trades_today.get(trade.date, 0) + 1
        pnl_today[trade.date] = pnl_today.get(trade.date, 0.0) + trade.pnl_points
        consecutive_losses = 0 if trade.is_win else consecutive_losses + 1
        last_exit_ts = trade.exit_ts
        if not trade.is_win and trade.exit_reason == "SL_HIT":
            loss_pause_until = trade.exit_ts + timedelta(minutes=12)
        elif not trade.is_win:
            loss_pause_until = trade.exit_ts + timedelta(minutes=config.cooldown_minutes)
        i = max(i + 1, exit_idx + 1)

    return trades


def _simulate_one_trade(
    candles: pd.DataFrame,
    signal_idx: int,
    signal: ScalpSignal,
    config: ScalpingConfig,
    trade_id: int,
) -> tuple[ScalpTrade | None, int]:
    if signal_idx + 1 >= len(candles):
        return None, signal_idx

    entry_bar = candles.iloc[signal_idx + 1]
    entry_ts = _as_ist(entry_bar.name)
    entry_price = float(entry_bar["open"])
    stop_loss = _price_from_offset(signal.action, entry_price, -config.stop_points)
    take_profit = _price_from_offset(signal.action, entry_price, config.target_points)

    mfe = 0.0
    mae = 0.0
    exit_idx = signal_idx + 1
    exit_ts = entry_ts
    exit_price = float(entry_bar["close"])
    exit_reason = "NO_EXIT"

    for j in range(signal_idx + 2, len(candles)):
        bar = candles.iloc[j]
        ts = _as_ist(bar.name)
        if ts.date() != entry_ts.date():
            prev = candles.iloc[j - 1]
            exit_idx = j - 1
            exit_ts = _as_ist(prev.name)
            exit_price = float(prev["close"])
            exit_reason = "EOD_CLOSE"
            break

        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        if signal.action == "BUY":
            mfe = max(mfe, high - entry_price)
            mae = max(mae, entry_price - low)
            if low <= stop_loss:
                exit_idx, exit_ts, exit_price, exit_reason = j, ts, stop_loss, "SL_HIT"
                break
            if high >= take_profit:
                exit_idx, exit_ts, exit_price, exit_reason = j, ts, take_profit, "TARGET_HIT"
                break
        else:
            mfe = max(mfe, entry_price - low)
            mae = max(mae, high - entry_price)
            if high >= stop_loss:
                exit_idx, exit_ts, exit_price, exit_reason = j, ts, stop_loss, "SL_HIT"
                break
            if low <= take_profit:
                exit_idx, exit_ts, exit_price, exit_reason = j, ts, take_profit, "TARGET_HIT"
                break

        elapsed = int((ts - entry_ts).total_seconds() // 60)
        current_pnl = _move(signal.action, entry_price, close)
        if elapsed >= 4 and mfe < 5.0:
            exit_idx, exit_ts, exit_price, exit_reason = j, ts, close, "TIME_EXIT_NO_FOLLOW"
            break
        if elapsed >= 8 and current_pnl < 8.0:
            exit_idx, exit_ts, exit_price, exit_reason = j, ts, close, "TIME_EXIT_STALL"
            break
        if elapsed >= config.max_hold_minutes:
            exit_idx, exit_ts, exit_price, exit_reason = j, ts, close, "TIME_EXIT"
            break

    pnl = _move(signal.action, entry_price, exit_price)
    holding_minutes = max(1, int((exit_ts - entry_ts).total_seconds() // 60))
    return (
        ScalpTrade(
            trade_id=trade_id,
            date=entry_ts.date(),
            entry_ts=entry_ts,
            action=signal.action,
            setup=signal.setup,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            exit_ts=exit_ts,
            exit_price=round(exit_price, 2),
            exit_reason=exit_reason,
            pnl_points=round(pnl, 2),
            holding_minutes=holding_minutes,
            mfe_points=round(mfe, 2),
            mae_points=round(mae, 2),
            is_win=pnl > 0,
            score=round(signal.score, 1),
        ),
        exit_idx,
    )


def compute_stats(trades: list[ScalpTrade]) -> dict[str, Any]:
    if not trades:
        return {}
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    pnls = [t.pnl_points for t in trades]
    gross_win = sum(t.pnl_points for t in wins)
    gross_loss = abs(sum(t.pnl_points for t in losses))
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    by_setup: dict[str, dict[str, float]] = {}
    by_day: dict[date, int] = {}
    exits: dict[str, int] = {}
    for trade in trades:
        setup = by_setup.setdefault(trade.setup, {"n": 0, "wins": 0, "pnl": 0.0})
        setup["n"] += 1
        setup["wins"] += int(trade.is_win)
        setup["pnl"] += trade.pnl_points
        by_day[trade.date] = by_day.get(trade.date, 0) + 1
        exits[trade.exit_reason] = exits.get(trade.exit_reason, 0) + 1
    return {
        "total_trades": len(trades),
        "trade_days": len(by_day),
        "avg_trades_per_trade_day": round(sum(by_day.values()) / max(len(by_day), 1), 2),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
        "total_pnl_points": round(sum(pnls), 2),
        "avg_pnl_points": round(sum(pnls) / len(trades), 2),
        "avg_win_points": round(gross_win / max(len(wins), 1), 2),
        "avg_loss_points": round(-gross_loss / max(len(losses), 1), 2),
        "profit_factor": round(gross_win / max(gross_loss, 1e-9), 2),
        "max_drawdown_points": round(max_dd, 2),
        "avg_holding_minutes": round(sum(t.holding_minutes for t in trades) / len(trades), 1),
        "by_setup": {
            k: {
                "n": int(v["n"]),
                "win_pct": round(v["wins"] / max(v["n"], 1) * 100, 1),
                "pnl": round(v["pnl"], 2),
            }
            for k, v in by_setup.items()
        },
        "by_exit_reason": exits,
    }


def candidate_configs() -> list[ScalpingConfig]:
    configs: list[ScalpingConfig] = []
    for stop, target, cooldown, max_hold, max_trades, rsi_l, rsi_s in itertools.product(
        [12.0, 14.0, 16.0],
        [18.0, 20.0, 24.0],
        [6, 8, 10],
        [8, 10, 12],
        [4, 5, 6],
        [30.0, 34.0, 38.0],
        [62.0, 66.0, 70.0],
    ):
        if target / stop < 1.2:
            continue
        configs.append(
            ScalpingConfig(
                stop_points=stop,
                target_points=target,
                cooldown_minutes=cooldown,
                max_hold_minutes=max_hold,
                max_trades_per_day=max_trades,
                reversion_rsi_long=rsi_l,
                reversion_rsi_short=rsi_s,
            )
        )
    return configs


def optimize_config(candles: pd.DataFrame) -> tuple[ScalpingConfig, dict[str, Any]]:
    split_idx = int(len(candles) * 0.70)
    train = candles.iloc[:split_idx].copy()
    test = candles.iloc[split_idx:].copy()
    best: tuple[float, ScalpingConfig, dict[str, Any]] | None = None
    for config in candidate_configs():
        train_features = build_scalping_features(train.reset_index(), config).set_index("ts")
        trades = simulate_scalps(train, train_features, config)
        stats = compute_stats(trades)
        if not stats or stats["total_trades"] < 40:
            continue
        score = (
            float(stats["profit_factor"]) * 100
            + float(stats["avg_pnl_points"]) * 8
            - float(stats["max_drawdown_points"]) * 0.12
        )
        if float(stats["avg_trades_per_trade_day"]) > 6.2:
            score -= 100
        if best is None or score > best[0]:
            best = (score, config, stats)
    if best is None:
        cfg = ScalpingConfig()
        features = build_scalping_features(test.reset_index(), cfg).set_index("ts")
        return cfg, compute_stats(simulate_scalps(test, features, cfg))

    selected = best[1]
    test_features = build_scalping_features(test.reset_index(), selected).set_index("ts")
    test_stats = compute_stats(simulate_scalps(test, test_features, selected))
    return selected, {"train": best[2], "test": test_stats}


def _save_logs(
    trades: list[ScalpTrade],
    stats: dict[str, Any],
    config: ScalpingConfig,
    symbol: str,
    days: int,
    output_dir: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_symbol = symbol.replace(" ", "_").replace("/", "_")
    rows = [
        {
            **asdict(trade),
            "date": trade.date.isoformat(),
            "entry_ts": trade.entry_ts.isoformat(),
            "exit_ts": trade.exit_ts.isoformat(),
        }
        for trade in trades
    ]
    csv_path = out / f"scalp_backtest_{clean_symbol}_{stamp}.csv"
    json_path = out / f"scalp_backtest_{clean_symbol}_{stamp}_summary.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "days": days,
                "config": asdict(config),
                "stats": stats,
                "trades": rows,
                "strategy_design": strategy_design(),
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n  Trade log  : {csv_path}")
    print(f"  JSON report: {json_path}")


def strategy_design() -> dict[str, Any]:
    return {
        "objective": "Independent 1-minute Nifty 50 scalper for 5-15 minute micro-moves.",
        "setups": [
            "VWAP/Bollinger mean reversion in low-ADX ranges after band rejection.",
            "Micro momentum burst after 10-bar high/low break with VWAP and EMA-9 alignment.",
        ],
        "risk": "Fixed 10-20 point stop class, target at least 1.2R, hard time exit.",
        "frequency": "Daily cap 4-6 trades with cooldown; active only in liquid morning/afternoon windows.",
        "separation": "No imports from, or writes to, the directional trend strategy path.",
    }


def _print_report(symbol: str, days: int, config: ScalpingConfig, stats: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("  INDEPENDENT SCALPING BACKTEST")
    print(f"  Symbol : {symbol}")
    print(f"  Period : last {days} calendar days")
    print(
        "  Rules  : "
        f"SL {config.stop_points:.0f} | TP {config.target_points:.0f} | "
        f"max hold {config.max_hold_minutes}m | cooldown {config.cooldown_minutes}m | "
        f"cap {config.max_trades_per_day}/day"
    )
    print("=" * 72)
    if not stats:
        print("  No trades generated.")
        return
    print(f"  Total Trades    : {stats['total_trades']}")
    print(f"  Trade Days      : {stats['trade_days']} ({stats['avg_trades_per_trade_day']} trades/trade-day)")
    print(f"  Win Rate        : {stats['win_rate_pct']}%")
    print(f"  Profit Factor   : {stats['profit_factor']}x")
    print(f"  Total P&L       : {stats['total_pnl_points']:+.1f} points")
    print(f"  Avg Trade       : {stats['avg_pnl_points']:+.2f} points")
    print(f"  Avg Win / Loss  : +{stats['avg_win_points']:.2f} / {stats['avg_loss_points']:.2f} points")
    print(f"  Max Drawdown    : {stats['max_drawdown_points']:.1f} points")
    print(f"  Avg Hold        : {stats['avg_holding_minutes']:.1f} minutes")
    print("\n  By Setup:")
    for setup, item in stats["by_setup"].items():
        print(f"    {setup:<16} trades={item['n']:4d} win={item['win_pct']:5.1f}% pnl={item['pnl']:+.1f}")
    print("\n  By Exit:")
    for reason, count in sorted(stats["by_exit_reason"].items(), key=lambda item: -item[1]):
        print(f"    {reason:<12} {count:4d}")
    verdict = "PAPER-READY" if stats["profit_factor"] >= 1.25 and stats["avg_pnl_points"] > 0 else "NEEDS WORK"
    print(f"\n  Verdict: {verdict} - keep disabled for live orders until tomorrow's paper run confirms slippage.")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent scalping strategy backtest")
    parser.add_argument("--symbol", default="Nifty 50")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--output-dir", default="logs/backtests")
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    print(f"\nLoading candles for '{args.symbol}' (last {args.days} days)...")
    candles = _load_candles(args.symbol, args.days)
    print(f"Loaded {len(candles):,} bars ({candles.index[0].date()} -> {candles.index[-1].date()})")

    if args.optimize:
        print("Running conservative train/test parameter search...")
        config, opt_stats = optimize_config(candles)
        print(f"Selected config: {asdict(config)}")
        print(f"Train stats: {opt_stats.get('train')}")
        print(f"Test stats : {opt_stats.get('test')}")
    else:
        config = ScalpingConfig()

    print("Building scalping features...")
    features = build_scalping_features(candles.reset_index(), config).set_index("ts")
    print("Running walk-forward scalping simulation...")
    trades = simulate_scalps(candles, features, config)
    stats = compute_stats(trades)
    _print_report(args.symbol, args.days, config, stats)
    _save_logs(trades, stats, config, args.symbol, args.days, args.output_dir)


if __name__ == "__main__":
    main()
