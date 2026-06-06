"""Diagnostic script to check why signals aren't generating."""
try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from datetime import datetime
from backend.db.connection import SessionLocal
from backend.execution_engine.live_service import load_market_context, build_technical_signal
from backend.utils.config import get_settings
from backend.utils.constants import IST_ZONE
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def diagnose_symbol(symbol: str):
    """Diagnose signal generation for a symbol."""
    db = SessionLocal()
    settings = get_settings()
    now = datetime.now(IST_ZONE)
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSING: {symbol} at {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"{'='*80}\n")
    
    try:
        # Load market context
        print("[*] Loading market context...")
        context = load_market_context(db, symbol=symbol, settings=settings, now=now)
        
        print(f"[OK] Loaded {len(context.signal_rows)} candles")
        print(f"     Latest candle: {context.latest_candle_ts}")
        print(f"     Latest price: {context.latest_price:.2f}")
        
        if len(context.signal_rows) < 60:
            print(f"\n[!] WARNING: Only {len(context.signal_rows)} candles (need 60+)")
            print("    Wait for more data to accumulate\n")
            return
        
        # Build technical signal
        print("\n[*] Building technical signal...")
        signal = build_technical_signal(db, context=context, settings=settings, now=now)
        
        print("\n[SIGNAL RESULT]:")
        print(f"   Action: {signal.action}")
        print(f"   Bias: {signal.bias}")
        print(f"   Score: {signal.score:.1f}/100")
        print(f"   Confidence: {signal.confidence:.2%}")
        print(f"   Conviction: {signal.conviction}")
        
        if signal.action in ["BUY", "SELL"]:
            print("\n[TRADE DETAILS]:")
            print(f"   Entry: {signal.entry_price:.2f}")
            print(f"   Stop Loss: {signal.stop_loss:.2f}")
            print(f"   Take Profit: {signal.take_profit:.2f}")
            print(f"   Risk: {signal.entry_price - signal.stop_loss:.2f} points")
            print(f"   Reward: {signal.take_profit - signal.entry_price:.2f} points")
        
        print("\n[REASONS]:")
        for i, reason in enumerate(signal.reasons, 1):
            print(f"   {i}. {reason}")
        
        print("\n[TECHNICAL DETAILS]:")
        details = signal.details
        print(f"   Close: {details.get('close', 0):.2f}")
        print(f"   EMA 9: {details.get('ema_9', 0):.2f}")
        print(f"   EMA 21: {details.get('ema_21', 0):.2f}")
        print(f"   RSI 14: {details.get('rsi_14', 0):.2f}")
        print(f"   Volume Ratio: {details.get('volume_ratio_20', 0):.2f}x")
        print(f"   EMA Cross Up: {details.get('ema_cross_up', False)}")
        print(f"   EMA Cross Down: {details.get('ema_cross_down', False)}")
        print(f"   Score Buy: {details.get('score_buy', 0):.1f}")
        print(f"   Score Sell: {details.get('score_sell', 0):.1f}")
        
        if signal.cooldown_seconds > 0:
            print(f"\n[!] COOLDOWN: {signal.cooldown_seconds}s remaining")
        
        if signal.max_signals_reached:
            print(f"\n[!] MAX SIGNALS REACHED: {details.get('signal_count_today', 0)} signals today")
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"[ERROR]: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def main():
    """Run diagnostics for all symbols."""
    settings = get_settings()
    
    print("\n" + "="*80)
    print("SIGNAL GENERATION DIAGNOSTICS")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Execution Enabled: {settings.execution_enabled}")
    print(f"  Execution Mode: {settings.execution_mode}")
    print(f"  Entry Window: {settings.entry_window_start} - {settings.entry_window_end}")
    print(f"  Force Squareoff: {settings.force_squareoff_time}")
    print(f"  Symbols: {', '.join(settings.execution_symbol_list)}")
    
    for symbol in settings.execution_symbol_list:
        diagnose_symbol(symbol)
    
    print("\n[OK] Diagnostics complete!\n")


if __name__ == "__main__":
    main()
