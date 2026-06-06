"""Diagnose why no trades are being placed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from sqlalchemy import select, and_, func
from backend.db.connection import SessionLocal
from backend.db.models import SignalLog, ExecutionPosition, ExecutionOrder
from backend.utils.constants import IST_ZONE

def main():
    db = SessionLocal()
    try:
        today = datetime.now(IST_ZONE).date()
        yesterday = today - timedelta(days=1)
        
        print("=" * 80)
        print("TRADE DIAGNOSTIC REPORT")
        print("=" * 80)
        
        # Check signal logs from last 2 days
        signals = db.execute(
            select(SignalLog)
            .where(SignalLog.trade_date >= yesterday)
            .order_by(SignalLog.timestamp.desc())
            .limit(20)
        ).scalars().all()
        
        print(f"\nLast 20 Signals (since {yesterday}):")
        print("-" * 80)
        
        if not signals:
            print("[X] NO SIGNALS FOUND - Signal generation is not working!")
            print("\nPossible causes:")
            print("  1. Execution loop is not running")
            print("  2. Market data is stale")
            print("  3. Entry window restrictions")
            return
        
        buy_signals = [s for s in signals if s.consensus == "BUY"]
        sell_signals = [s for s in signals if s.consensus == "SELL"]
        hold_signals = [s for s in signals if s.consensus not in ("BUY", "SELL")]
        
        print(f"  BUY signals:  {len(buy_signals)}")
        print(f"  SELL signals: {len(sell_signals)}")
        print(f"  HOLD signals: {len(hold_signals)}")
        print(f"  Trades placed: {sum(1 for s in signals if s.trade_placed)}")
        
        # Show recent signals with details
        print("\n📋 Recent Signal Details:")
        print("-" * 80)
        for i, signal in enumerate(signals[:10], 1):
            details = signal.details or {}
            print(f"\n{i}. {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {signal.symbol}")
            print(f"   Consensus: {signal.consensus} | Score: {signal.combined_score or 0:.2f}")
            print(f"   Trade Placed: {'[YES]' if signal.trade_placed else '[NO]'}")
            
            if signal.skip_reason:
                print(f"   Skip Reason: {signal.skip_reason}")
            
            # Show scoring breakdown
            score_buy = details.get('score_buy', 0)
            score_sell = details.get('score_sell', 0)
            print(f"   Scores: BUY={score_buy} | SELL={score_sell}")
            
            # Show key checks
            print("   Checks:")
            print(f"     - EMA cross: {details.get('ema_cross_up', False)} (up) / {details.get('ema_cross_down', False)} (down)")
            print(f"     - RSI: {details.get('rsi_14', 0):.1f} (buy_ok={details.get('rsi_buy_ok', False)}, sell_ok={details.get('rsi_sell_ok', False)})")
            print(f"     - Volume OK: {details.get('volume_ok', False)}")
            print(f"     - Entry window: {details.get('entry_window_open', False)}")
            print(f"     - VIX: {details.get('vix_level', 'N/A')} (too_high={details.get('vix_too_high', False)}, too_low={details.get('vix_too_low', False)})")
        
        # Check positions
        print("\nPositions:")
        print("-" * 80)
        open_pos = db.scalar(select(func.count(ExecutionPosition.id)).where(ExecutionPosition.status == "OPEN"))
        closed_pos = db.scalar(
            select(func.count(ExecutionPosition.id))
            .where(and_(ExecutionPosition.status == "CLOSED", ExecutionPosition.trade_date >= yesterday))
        )
        print(f"  Open positions: {open_pos or 0}")
        print(f"  Closed positions (last 2 days): {closed_pos or 0}")
        
        # Check orders
        orders = db.scalar(
            select(func.count(ExecutionOrder.id))
            .where(ExecutionOrder.trade_date >= yesterday)
        )
        print(f"  Orders (last 2 days): {orders or 0}")
        
        # Analyze why trades aren't happening
        print("\nDIAGNOSIS:")
        print("-" * 80)
        
        if not buy_signals and not sell_signals:
            print("[X] NO BUY/SELL SIGNALS - All signals are HOLD")
            print("\nMost common skip reasons:")
            skip_reasons = {}
            for s in hold_signals[:20]:
                reason = s.skip_reason or "Unknown"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1])[:5]:
                print(f"  • {reason}: {count} times")
        
        elif buy_signals or sell_signals:
            placed = sum(1 for s in signals if s.trade_placed)
            if placed == 0:
                print("[!] SIGNALS GENERATED BUT NO TRADES PLACED")
                print("\nChecking why trades were skipped:")
                for signal in (buy_signals + sell_signals)[:5]:
                    if signal.skip_reason:
                        print(f"  • {signal.timestamp.strftime('%H:%M:%S')}: {signal.skip_reason}")
            else:
                print(f"[OK] {placed} trades were placed successfully")
        
        print("\n" + "=" * 80)
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
