"""Real-time monitoring dashboard for signal generation."""
try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import time
from datetime import datetime
from backend.db.connection import SessionLocal
from sqlalchemy import select, func, and_
from backend.db.models import SignalLog, ExecutionPosition, RawCandle
from backend.utils.config import get_settings
from backend.utils.constants import IST_ZONE


def clear_screen():
    """Clear terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def get_latest_candles(db, symbol: str, limit: int = 5):
    """Get latest candles for a symbol."""
    from backend.execution_engine.live_service import resolve_instrument_key
    
    try:
        instrument_key, _ = resolve_instrument_key(db, symbol)
        candles = db.execute(
            select(RawCandle)
            .where(
                and_(
                    RawCandle.instrument_key == instrument_key,
                    RawCandle.interval == "1minute"
                )
            )
            .order_by(RawCandle.ts.desc())
            .limit(limit)
        ).scalars().all()
        return list(reversed(candles))
    except Exception:
        return []


def get_today_stats(db):
    """Get today's trading statistics."""
    today = datetime.now(IST_ZONE).date()
    
    # Count signals
    signal_count = db.scalar(
        select(func.count(SignalLog.id))
        .where(SignalLog.trade_date == today)
    ) or 0
    
    buy_signals = db.scalar(
        select(func.count(SignalLog.id))
        .where(and_(SignalLog.trade_date == today, SignalLog.consensus == "BUY"))
    ) or 0
    
    sell_signals = db.scalar(
        select(func.count(SignalLog.id))
        .where(and_(SignalLog.trade_date == today, SignalLog.consensus == "SELL"))
    ) or 0
    
    # Count positions
    open_positions = db.scalar(
        select(func.count(ExecutionPosition.id))
        .where(ExecutionPosition.status == "OPEN")
    ) or 0
    
    closed_positions = db.scalar(
        select(func.count(ExecutionPosition.id))
        .where(and_(
            ExecutionPosition.trade_date == today,
            ExecutionPosition.status == "CLOSED"
        ))
    ) or 0
    
    # Calculate PnL
    total_pnl = db.scalar(
        select(func.sum(ExecutionPosition.realized_pnl))
        .where(and_(
            ExecutionPosition.trade_date == today,
            ExecutionPosition.status == "CLOSED"
        ))
    ) or 0.0
    
    return {
        'signal_count': signal_count,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'open_positions': open_positions,
        'closed_positions': closed_positions,
        'total_pnl': float(total_pnl)
    }


def get_recent_signals(db, limit: int = 5):
    """Get recent signals."""
    signals = db.execute(
        select(SignalLog)
        .order_by(SignalLog.timestamp.desc())
        .limit(limit)
    ).scalars().all()
    return signals


def get_open_positions(db):
    """Get open positions."""
    positions = db.execute(
        select(ExecutionPosition)
        .where(ExecutionPosition.status == "OPEN")
        .order_by(ExecutionPosition.opened_at.desc())
    ).scalars().all()
    return positions


def display_dashboard():
    """Display real-time monitoring dashboard."""
    db = SessionLocal()
    settings = get_settings()
    
    try:
        clear_screen()
        now = datetime.now(IST_ZONE)
        
        print("=" * 100)
        print(f"{'AI TRADING PLATFORM - LIVE MONITOR':^100}")
        print("=" * 100)
        print(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
        print(f"Mode: {settings.execution_mode.upper()} | Enabled: {settings.execution_enabled}")
        print("=" * 100)
        
        # Today's stats
        stats = get_today_stats(db)
        print("\n📊 TODAY'S STATISTICS")
        print(f"   Signals: {stats['signal_count']} (BUY: {stats['buy_signals']}, SELL: {stats['sell_signals']})")
        print(f"   Positions: {stats['open_positions']} open, {stats['closed_positions']} closed")
        print(f"   PnL: ₹{stats['total_pnl']:,.2f}")
        
        # Latest candles for each symbol
        print("\n📈 LATEST CANDLES (Last 3)")
        for symbol in settings.execution_symbol_list:
            candles = get_latest_candles(db, symbol, limit=3)
            if candles:
                print(f"\n   {symbol}:")
                for candle in candles:
                    ts = candle.ts.strftime('%H:%M')
                    change = candle.close - candle.open
                    change_pct = (change / candle.open * 100) if candle.open else 0
                    arrow = "🟢" if change >= 0 else "🔴"
                    print(f"      {ts} | O:{candle.open:.2f} H:{candle.high:.2f} L:{candle.low:.2f} C:{candle.close:.2f} {arrow} {change_pct:+.2f}%")
            else:
                print(f"\n   {symbol}: No data")
        
        # Recent signals
        print("\n🎯 RECENT SIGNALS (Last 5)")
        signals = get_recent_signals(db, limit=5)
        if signals:
            for sig in signals:
                ts = sig.timestamp.strftime('%H:%M:%S')
                action = sig.consensus
                symbol = sig.symbol
                score = sig.combined_score * 100 if sig.combined_score else 0
                placed = "✅" if sig.trade_placed else "❌"
                skip = f" ({sig.skip_reason})" if sig.skip_reason else ""
                print(f"   {ts} | {symbol:15} | {action:4} | Score: {score:5.1f} | Placed: {placed}{skip}")
        else:
            print("   No signals yet today")
        
        # Open positions
        print("\n💼 OPEN POSITIONS")
        positions = get_open_positions(db)
        if positions:
            for pos in positions:
                entry_time = pos.opened_at.strftime('%H:%M')
                pnl = pos.unrealized_pnl or 0
                pnl_color = "🟢" if pnl >= 0 else "🔴"
                print(f"   {entry_time} | {pos.symbol:15} | {pos.option_type} {pos.strike:.0f} | Entry: {pos.entry_premium:.2f} | Current: {pos.current_premium:.2f} | PnL: {pnl_color} ₹{pnl:,.2f}")
        else:
            print("   No open positions")
        
        print("\n" + "=" * 100)
        print("Press Ctrl+C to exit | Refreshing every 5 seconds...")
        print("=" * 100)
        
    finally:
        db.close()


def main():
    """Run monitoring dashboard."""
    print("\nStarting monitoring dashboard...\n")
    
    try:
        while True:
            display_dashboard()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.\n")


if __name__ == "__main__":
    main()
