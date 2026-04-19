from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from backtesting.metrics import max_drawdown, sharpe_ratio
from backtesting.strategies import direction_strategy
from db.models import PredictionsDaily, RawCandle
from feature_engine.price_features import build_price_features
from utils.constants import IST_ZONE
from utils.pine_strategy import (
    attach_short_interval_states,
    in_entry_window,
    in_manage_window,
    row_setup_flags,
    strategy_profile_for_symbol,
)
from utils.symbols import instrument_key_filter


def run_backtest(db: Session, symbol: str) -> dict:
    preds = db.execute(
        select(PredictionsDaily).where(PredictionsDaily.symbol == symbol).order_by(PredictionsDaily.target_session_date.asc())
    ).scalars().all()
    if not preds:
        return {"status": "no_predictions"}
    rows = []
    for prediction in preds:
        candle = db.scalar(
            select(RawCandle)
            .where(
                and_(
                    RawCandle.interval == "day",
                    instrument_key_filter(RawCandle.instrument_key, symbol),
                    func.date(RawCandle.ts) == prediction.target_session_date,
                )
            )
            .order_by(RawCandle.ts.asc())
        )
        rows.append(
            {
                "session_date": prediction.target_session_date,
                "direction": prediction.direction,
                "pred_close": prediction.pred_close,
                "actual_close": candle.close if candle else prediction.pred_close,
            }
        )
    frame = pd.DataFrame(rows).sort_values("session_date")
    result = direction_strategy(frame)
    return {
        "status": "ok",
        "sharpe": sharpe_ratio(result["strategy_ret"]),
        "max_drawdown": max_drawdown(result["equity_curve"]),
        "rows": len(result),
    }


def _load_intraday_frame(
    db: Session,
    *,
    symbol: str,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    query = select(RawCandle).where(
        and_(
            instrument_key_filter(RawCandle.instrument_key, symbol),
            RawCandle.interval == "1minute",
        )
    )
    if start_date is not None:
        query = query.where(func.date(RawCandle.ts) >= start_date)
    if end_date is not None:
        query = query.where(func.date(RawCandle.ts) <= end_date)
    rows = db.execute(query.order_by(RawCandle.ts.asc())).scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "ts": [r.ts if r.ts.tzinfo is not None else r.ts.replace(tzinfo=IST_ZONE) for r in rows],
            "open": [float(r.open) for r in rows],
            "high": [float(r.high) for r in rows],
            "low": [float(r.low) for r in rows],
            "close": [float(r.close) for r in rows],
            "volume": [float(r.volume) for r in rows],
        }
    )


def _pine_signals(frame: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    out = build_price_features(frame.copy()).reset_index(drop=True)
    short_intervals = attach_short_interval_states(frame.copy(), symbol=symbol)
    for column in ["mtf_3m_action", "mtf_3m_buy_score", "mtf_3m_sell_score", "mtf_3m_market_regime", "mtf_5m_action", "mtf_5m_buy_score", "mtf_5m_sell_score", "mtf_5m_market_regime"]:
        if column in short_intervals.columns:
            out[column] = short_intervals[column]
    profile = strategy_profile_for_symbol(symbol)
    out["in_entry_window"] = out["ts"].apply(lambda value: in_entry_window(value, symbol=symbol))
    out["in_manage_window"] = out["ts"].apply(lambda value: in_manage_window(value, symbol=symbol))
    out["base_buy_setup"] = False
    out["base_sell_setup"] = False
    out["buy_setup"] = False
    out["sell_setup"] = False
    out["buy_score"] = 0.0
    out["sell_score"] = 0.0
    out["buy_alignment_count"] = 0
    out["sell_alignment_count"] = 0
    out["market_regime"] = "range"
    out["buy_strategy_names"] = ""
    out["sell_strategy_names"] = ""
    out["buy_signal"] = False
    out["sell_signal"] = False
    out["force_exit"] = (~out["in_manage_window"]).astype(bool)

    position = 0
    last_entry_idx = -10_000
    rows = list(out.itertuples(index=False))
    for idx, row in enumerate(rows):
        setups = row_setup_flags(row, symbol=symbol)
        out.at[idx, "base_buy_setup"] = bool(setups.get("base_buy_setup"))
        out.at[idx, "base_sell_setup"] = bool(setups.get("base_sell_setup"))
        buy_setup = bool(setups["buy_setup"])
        sell_setup = bool(setups["sell_setup"])
        out.at[idx, "buy_setup"] = buy_setup
        out.at[idx, "sell_setup"] = sell_setup
        out.at[idx, "buy_score"] = float(setups.get("buy_score") or 0.0)
        out.at[idx, "sell_score"] = float(setups.get("sell_score") or 0.0)
        out.at[idx, "buy_alignment_count"] = int(setups.get("buy_alignment_count") or 0)
        out.at[idx, "sell_alignment_count"] = int(setups.get("sell_alignment_count") or 0)
        out.at[idx, "market_regime"] = str(setups.get("market_regime") or "range")
        out.at[idx, "buy_strategy_names"] = ",".join(setups.get("buy_strategy_names") or [])
        out.at[idx, "sell_strategy_names"] = ",".join(setups.get("sell_strategy_names") or [])

        force_exit = bool(getattr(row, "force_exit"))
        cooldown_ok = idx - last_entry_idx >= int(profile.cooldown_bars)
        buy_signal = bool(getattr(row, "in_entry_window")) and cooldown_ok and buy_setup and position <= 0
        sell_signal = bool(getattr(row, "in_entry_window")) and cooldown_ok and sell_setup and position >= 0
        out.at[idx, "buy_signal"] = buy_signal
        out.at[idx, "sell_signal"] = sell_signal

        if position == 1 and (sell_signal or force_exit):
            position = 0
        elif position == -1 and (buy_signal or force_exit):
            position = 0

        if force_exit:
            continue

        if position == 0 and buy_signal:
            position = 1
            last_entry_idx = idx
        elif position == 0 and sell_signal:
            position = -1
            last_entry_idx = idx
    return out


def run_pine_signal_backtest(
    db: Session,
    *,
    symbol: str,
    start_date: date | None = None,
    end_date: date | None = None,
    initial_capital: float = 100000.0,
) -> dict:
    from utils.symbols import normalize_symbol_key
    
    # Skip India VIX
    if normalize_symbol_key(symbol) == "INDIAVIX":
        return {"status": "skipped", "symbol": symbol, "reason": "India VIX disabled"}
    
    frame = _load_intraday_frame(db, symbol=symbol, start_date=start_date, end_date=end_date)
    if frame.empty:
        return {"status": "no_candles", "symbol": symbol}

    frame = _pine_signals(frame, symbol=symbol)
    capital = float(initial_capital)
    equity_points = [capital]
    returns: list[float] = []
    trades: list[dict] = []
    exit_reasons = {"SL": 0, "TSL": 0, "TARGET": 0, "BREAKEVEN": 0, "PARTIAL_2R": 0, "TREND_WEAK": 0, "EOD": 0}
    
    position = 0
    entry_price = 0.0
    entry_ts: datetime | None = None
    initial_sl = 0.0
    current_sl = 0.0
    target = 0.0
    peak_price = 0.0
    tsl_active = False
    partial_exit_done = False

    for row in frame.itertuples(index=False):
        ts = row.ts if row.ts.tzinfo is not None else row.ts.replace(tzinfo=IST_ZONE)
        close = float(row.close)
        high = float(row.high)
        low = float(row.low)
        buy_signal = bool(row.buy_signal)
        sell_signal = bool(row.sell_signal)
        force_exit = bool(row.force_exit)

        def close_trade(reason: str, exit_price: float, size_pct: float = 1.0) -> None:
            nonlocal capital, position, entry_price, entry_ts, partial_exit_done
            if position == 0 or entry_ts is None:
                return
            pnl_points = (exit_price - entry_price) * position * size_pct
            trade_return = ((exit_price - entry_price) / max(entry_price, 1e-9)) * position * size_pct
            capital *= 1.0 + trade_return
            returns.append(trade_return)
            trades.append({
                "entry_time": entry_ts.isoformat(),
                "exit_time": ts.isoformat(),
                "side": "BUY" if position == 1 else "SELL",
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl_points": round(pnl_points, 2),
                "return_pct": round(trade_return * 100.0, 3),
                "exit_reason": reason,
                "position_size": f"{int(size_pct*100)}%",
            })
            equity_points.append(capital)
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            if size_pct >= 1.0:
                position = 0
                entry_price = 0.0
                entry_ts = None
                partial_exit_done = False

        # Manage open position
        if position != 0:
            risk = abs(entry_price - initial_sl)
            profit_r = (close - entry_price) * position / risk if risk > 0 else 0
            
            # Update peak
            if position == 1:
                peak_price = max(peak_price, high)
            else:
                peak_price = min(peak_price, low)
            
            # Partial exit at 2R
            if not partial_exit_done and profit_r >= 2.0:
                partial_exit_done = True
                close_trade("PARTIAL_2R", close, 0.5)
            
            # Update trailing stop
            if profit_r >= 1.0:
                if position == 1:
                    current_sl = max(current_sl, entry_price)
                else:
                    current_sl = min(current_sl, entry_price)
            
            if profit_r >= 1.5:
                if position == 1:
                    current_sl = max(current_sl, entry_price + 0.5 * risk)
                else:
                    current_sl = min(current_sl, entry_price - 0.5 * risk)
            
            if profit_r >= 2.0:
                tsl_active = True
                if position == 1:
                    trail_sl = round(peak_price * 0.97, 2)
                    current_sl = max(current_sl, trail_sl)
                else:
                    trail_sl = round(peak_price * 1.03, 2)
                    current_sl = min(current_sl, trail_sl)
            
            # Check exits
            exit_triggered = False
            exit_reason = None
            exit_price = close
            
            # Target hit
            if (position == 1 and high >= target) or (position == -1 and low <= target):
                exit_triggered = True
                exit_reason = "TARGET"
                exit_price = target
            # Stop hit
            elif (position == 1 and low <= current_sl) or (position == -1 and high >= current_sl):
                exit_triggered = True
                if abs(current_sl - entry_price) < 0.01:
                    exit_reason = "BREAKEVEN"
                elif tsl_active:
                    exit_reason = "TSL"
                else:
                    exit_reason = "SL"
                exit_price = current_sl
            # Trend weakening (only if profit < 1R)
            elif profit_r < 1.0:
                adx = getattr(row, "adx_14", 25)
                ema_slope = getattr(row, "ema_21_slope_3", 0)
                if adx < 15 and ((position == 1 and ema_slope < 0) or (position == -1 and ema_slope > 0)):
                    exit_triggered = True
                    exit_reason = "TREND_WEAK"
            # Force exit
            elif force_exit:
                exit_triggered = True
                exit_reason = "EOD"
            
            if exit_triggered:
                size = 0.5 if partial_exit_done else 1.0
                close_trade(exit_reason, exit_price, size)

        # Check for new entries
        if position == 0 and not force_exit:
            if buy_signal:
                position = 1
                entry_price = close
                entry_ts = ts
                # Calculate SL
                if entry_price < 50:
                    initial_sl = round(entry_price - 10.0, 2)
                elif entry_price <= 100:
                    initial_sl = round(entry_price * 0.92, 2)
                elif entry_price <= 200:
                    initial_sl = round(entry_price * 0.93, 2)
                else:
                    initial_sl = round(entry_price * 0.94, 2)
                current_sl = initial_sl
                risk = entry_price - initial_sl
                target = round(entry_price + 2.0 * risk, 2)
                peak_price = entry_price
                tsl_active = False
                partial_exit_done = False
            elif sell_signal:
                position = -1
                entry_price = close
                entry_ts = ts
                # Calculate SL
                if entry_price < 50:
                    initial_sl = round(entry_price + 10.0, 2)
                elif entry_price <= 100:
                    initial_sl = round(entry_price * 1.08, 2)
                elif entry_price <= 200:
                    initial_sl = round(entry_price * 1.07, 2)
                else:
                    initial_sl = round(entry_price * 1.06, 2)
                current_sl = initial_sl
                risk = initial_sl - entry_price
                target = round(entry_price - 2.0 * risk, 2)
                peak_price = entry_price
                tsl_active = False
                partial_exit_done = False

    # Close any remaining position
    if position != 0 and entry_ts is not None:
        last_row = frame.iloc[-1]
        close = float(last_row["close"])
        size = 0.5 if partial_exit_done else 1.0
        close_trade("END_OF_DATA", close, size)

    wins = sum(1 for trade in trades if float(trade["return_pct"]) > 0.0)
    sessions = sorted({(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts).astimezone(IST_ZONE).date().isoformat() for ts in frame["ts"]})
    equity_curve = []
    running_equity = float(initial_capital)
    for trade in trades:
        running_equity *= 1.0 + (float(trade["return_pct"]) / 100.0)
        equity_curve.append({"x": trade["exit_time"], "value": round(running_equity, 2)})

    gross_return_pct = ((capital / max(initial_capital, 1e-9)) - 1.0) * 100.0
    return {
        "status": "ok",
        "symbol": symbol,
        "start_date": sessions[0] if sessions else None,
        "end_date": sessions[-1] if sessions else None,
        "sessions": len(sessions),
        "bars": int(len(frame)),
        "signals": int(frame["buy_signal"].sum() + frame["sell_signal"].sum()),
        "trades": len(trades),
        "wins": wins,
        "losses": max(0, len(trades) - wins),
        "win_rate_pct": round((wins / len(trades) * 100.0) if trades else 0.0, 2),
        "gross_return_pct": round(gross_return_pct, 2),
        "final_equity": round(capital, 2),
        "max_drawdown_pct": round(abs(max_drawdown(equity_points)) * 100.0, 2),
        "sharpe": round(float(sharpe_ratio(returns, annualization=252)), 4) if returns else 0.0,
        "exit_reasons": exit_reasons,
        "trades_per_session": round(len(trades) / len(sessions), 2) if sessions else 0.0,
        "equity_curve": equity_curve[-300:],
        "trade_log": trades[-200:],
    }
