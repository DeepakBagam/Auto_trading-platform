"""Backtest the graph-marker signal used by live execution.

This tests the same Pine-style overlay markers that the chart renders and that
live execution now consumes. It simulates one non-overlapping underlying trade
per fresh marker, entering on the next 1-minute candle.

Usage:
    python backend/scripts/backtest_graph_signal.py --symbol "Nifty 50" --days 30
"""
try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import and_, select

from backend.db.connection import SessionLocal
from backend.db.models import RawCandle
from backend.execution_engine.live_service import CHART_MARKER_LIMITS, _build_pine_chart_overlay
from backend.utils.config import Settings, get_settings
from backend.utils.constants import IST_ZONE
from backend.utils.symbols import instrument_key_filter

MIN_BARS = 60
DEFAULT_ENTRY_START = time(9, 45)
DEFAULT_ENTRY_END = time(15, 0)


@dataclass(slots=True)
class GraphSignalTrade:
    trade_id: int
    trade_date: date
    signal_ts: datetime
    entry_ts: datetime
    action: str
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


def _as_ist(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(IST_ZONE) if value.tzinfo else value.replace(tzinfo=IST_ZONE)
    out = pd.Timestamp(value).to_pydatetime()
    return out.astimezone(IST_ZONE) if out.tzinfo else out.replace(tzinfo=IST_ZONE)


def _parse_hhmm(value: str | None, fallback: time) -> time:
    if not value:
        return fallback
    try:
        hh, mm = str(value).split(":", 1)
        return time(int(hh), int(mm))
    except Exception:
        return fallback


def _load_candles(symbol: str, days: int) -> pd.DataFrame:
    cutoff = datetime.now(IST_ZONE) - timedelta(days=days + 10)
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


def _marker_map(candles: pd.DataFrame, settings: Settings) -> dict[datetime, str]:
    rows = candles.reset_index().rename(columns={"index": "ts"}).to_dict("records")
    CHART_MARKER_LIMITS["backtest"] = max(len(rows), 1)
    overlay = _build_pine_chart_overlay(rows, interval="1minute", settings=settings, range_key="backtest")
    out: dict[datetime, str] = {}
    for marker in overlay.get("markers") or []:
        action = str(marker.get("text") or "").upper()
        if action not in {"BUY", "SELL"}:
            continue
        out[_as_ist(marker.get("time")).replace(second=0, microsecond=0)] = action
    return out


def _atr_series(candles: pd.DataFrame, length: int = 14) -> pd.Series:
    high = pd.to_numeric(candles["high"], errors="coerce")
    low = pd.to_numeric(candles["low"], errors="coerce")
    close = pd.to_numeric(candles["close"], errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / length, adjust=False).mean()


def _move(action: str, entry: float, price: float) -> float:
    return round((price - entry) * (1 if action == "BUY" else -1), 2)


def _offset_price(action: str, entry: float, points: float) -> float:
    return round(entry + (points * (1 if action == "BUY" else -1)), 2)


def _simulate_one(
    candles: pd.DataFrame,
    signal_idx: int,
    action: str,
    atr: pd.Series,
    trade_id: int,
) -> tuple[GraphSignalTrade | None, int]:
    if signal_idx + 1 >= len(candles):
        return None, signal_idx

    signal_bar = candles.iloc[signal_idx]
    prev_bar = candles.iloc[max(0, signal_idx - 1)]
    entry_bar = candles.iloc[signal_idx + 1]
    signal_ts = _as_ist(signal_bar.name)
    entry_ts = _as_ist(entry_bar.name)
    entry_price = float(entry_bar["open"])
    atr_points = max(1e-9, float(atr.iloc[signal_idx] or 0.0))
    if action == "BUY":
        stop_points = max(atr_points, abs(entry_price - float(prev_bar["low"])))
    else:
        stop_points = max(atr_points, abs(float(prev_bar["high"]) - entry_price))
    target_points = stop_points * 1.5
    stop_loss = _offset_price(action, entry_price, -stop_points)
    take_profit = _offset_price(action, entry_price, target_points)

    mfe = 0.0
    mae = 0.0
    exit_idx = signal_idx + 1
    exit_ts = entry_ts
    exit_price = float(entry_bar["close"])
    exit_reason = "NO_EXIT"

    for idx in range(signal_idx + 2, len(candles)):
        bar = candles.iloc[idx]
        ts = _as_ist(bar.name)
        if ts.date() != entry_ts.date():
            last = candles.iloc[idx - 1]
            exit_idx = idx - 1
            exit_ts = _as_ist(last.name)
            exit_price = float(last["close"])
            exit_reason = "EOD_CLOSE"
            break

        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        if action == "BUY":
            mfe = max(mfe, high - entry_price)
            mae = max(mae, entry_price - low)
            if low <= stop_loss:
                exit_idx, exit_ts, exit_price, exit_reason = idx, ts, stop_loss, "SL_HIT"
                break
            if high >= take_profit:
                exit_idx, exit_ts, exit_price, exit_reason = idx, ts, take_profit, "TP_HIT"
                break
        else:
            mfe = max(mfe, entry_price - low)
            mae = max(mae, high - entry_price)
            if high >= stop_loss:
                exit_idx, exit_ts, exit_price, exit_reason = idx, ts, stop_loss, "SL_HIT"
                break
            if low <= take_profit:
                exit_idx, exit_ts, exit_price, exit_reason = idx, ts, take_profit, "TP_HIT"
                break
        exit_idx, exit_ts, exit_price, exit_reason = idx, ts, close, "EOD_CLOSE"

    pnl = _move(action, entry_price, exit_price)
    holding = max(0, int((exit_ts - entry_ts).total_seconds() // 60))
    return (
        GraphSignalTrade(
            trade_id=trade_id,
            trade_date=entry_ts.date(),
            signal_ts=signal_ts,
            entry_ts=entry_ts,
            action=action,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            exit_ts=exit_ts,
            exit_price=round(exit_price, 2),
            exit_reason=exit_reason,
            pnl_points=round(pnl, 2),
            holding_minutes=holding,
            mfe_points=round(mfe, 2),
            mae_points=round(mae, 2),
            is_win=pnl > 0,
        ),
        exit_idx,
    )


def simulate(candles: pd.DataFrame, settings: Settings, entry_dates: set[date]) -> list[GraphSignalTrade]:
    markers = _marker_map(candles, settings)
    atr = _atr_series(candles)
    entry_start = _parse_hhmm(settings.entry_window_start, DEFAULT_ENTRY_START)
    entry_end = _parse_hhmm(settings.entry_window_end, DEFAULT_ENTRY_END)
    cooldown_minutes = max(1, int(getattr(settings, "signal_cooldown_minutes", 12)))
    max_per_day = max(1, int(getattr(settings, "signal_max_per_day", 2)))

    trades: list[GraphSignalTrade] = []
    trades_today: dict[date, int] = {}
    last_signal_ts: datetime | None = None
    i = MIN_BARS
    while i < len(candles) - 1:
        ts = _as_ist(candles.index[i]).replace(second=0, microsecond=0)
        action = markers.get(ts)
        if action not in {"BUY", "SELL"}:
            i += 1
            continue
        if ts.date() not in entry_dates:
            i += 1
            continue
        now_time = ts.timetz().replace(tzinfo=None)
        if not (entry_start <= now_time <= entry_end):
            i += 1
            continue
        if trades_today.get(ts.date(), 0) >= max_per_day:
            i += 1
            continue
        if last_signal_ts is not None and (ts - last_signal_ts).total_seconds() < cooldown_minutes * 60:
            i += 1
            continue

        trade, exit_idx = _simulate_one(candles, i, action, atr, len(trades) + 1)
        if trade is None:
            break
        trades.append(trade)
        trades_today[trade.trade_date] = trades_today.get(trade.trade_date, 0) + 1
        last_signal_ts = ts
        i = max(i + 1, exit_idx + 1)
    return trades


def _stats(trades: list[GraphSignalTrade]) -> dict[str, Any]:
    total = len(trades)
    pnl = [trade.pnl_points for trade in trades]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value <= 0]
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return {
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round((len(wins) / total * 100.0) if total else 0.0, 2),
        "total_points": round(sum(pnl), 2),
        "avg_points": round((sum(pnl) / total) if total else 0.0, 2),
        "best_trade": round(max(pnl), 2) if pnl else 0.0,
        "worst_trade": round(min(pnl), 2) if pnl else 0.0,
        "max_drawdown_points": round(max_dd, 2),
        "profit_factor": round((sum(wins) / abs(sum(losses))) if losses else 999.0, 2),
    }


def _save(trades: list[GraphSignalTrade], stats: dict[str, Any], symbol: str, days: int, output_dir: str) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(IST_ZONE).strftime("%Y%m%d_%H%M%S")
    clean = "".join(ch if ch.isalnum() else "_" for ch in symbol).strip("_").lower()
    csv_path = out / f"graph_signal_backtest_{clean}_{days}d_{stamp}.csv"
    json_path = out / f"graph_signal_backtest_{clean}_{days}d_{stamp}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(trades[0]).keys()) if trades else ["trade_id"])
        writer.writeheader()
        for trade in trades:
            writer.writerow(asdict(trade))
    json_path.write_text(pd.Series(stats).to_json(indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-marker execution signal backtest")
    parser.add_argument("--symbol", default="Nifty 50")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output-dir", default="logs/backtests")
    args = parser.parse_args()

    settings = get_settings()
    candles = _load_candles(args.symbol, args.days)
    sessions = sorted({idx.date() for idx in candles.index})
    entry_dates = set(sessions[-max(1, int(args.days)):])
    trades = simulate(candles, settings, entry_dates)
    stats = _stats(trades)
    csv_path, json_path = _save(trades, stats, args.symbol, args.days, args.output_dir)

    print("\nGRAPH-MARKER SIGNAL BACKTEST")
    print(f"Symbol: {args.symbol}")
    print(f"Bars: {len(candles):,} ({candles.index[0].date()} -> {candles.index[-1].date()})")
    print(f"Entry sessions: {min(entry_dates)} -> {max(entry_dates)}")
    print(f"Trades: {stats['trades']} | Win rate: {stats['win_rate']}% | Total points: {stats['total_points']}")
    print(f"Avg: {stats['avg_points']} | Best: {stats['best_trade']} | Worst: {stats['worst_trade']}")
    print(f"Profit factor: {stats['profit_factor']} | Max DD points: {stats['max_drawdown_points']}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
