"""Test signal generation and show detailed breakdown."""
try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

from datetime import datetime
from backend.db.connection import SessionLocal
from backend.execution_engine.live_service import (
    load_market_context, 
    build_technical_signal,
    build_option_selection
)
from backend.utils.config import get_settings
from backend.utils.constants import IST_ZONE


def test_signal_generation(symbol: str):
    """Test signal generation for a symbol with detailed output."""
    db = SessionLocal()
    settings = get_settings()
    now = datetime.now(IST_ZONE)
    
    print(f"\n{'='*100}")
    print(f"TESTING SIGNAL GENERATION: {symbol}")
    print(f"{'='*100}\n")
    
    try:
        # Step 1: Load market context
        print("Step 1: Loading market context...")
        context = load_market_context(db, symbol=symbol, settings=settings, now=now)
        print(f"✅ Loaded {len(context.signal_rows)} candles")
        print(f"   Latest: {context.latest_candle_ts}")
        print(f"   Price: {context.latest_price:.2f}")
        
        if len(context.signal_rows) < 60:
            print(f"\n⚠️  Need 60+ candles, have {len(context.signal_rows)}")
            print("   System needs more data to calculate EMAs")
            return
        
        # Step 2: Build technical signal
        print("\nStep 2: Building technical signal...")
        signal = build_technical_signal(db, context=context, settings=settings, now=now)
        
        print(f"\n{'='*100}")
        print("SIGNAL RESULT")
        print(f"{'='*100}")
        print(f"Action:      {signal.action}")
        print(f"Bias:        {signal.bias}")
        print(f"Score:       {signal.score:.1f}/100")
        print(f"Confidence:  {signal.confidence:.2%}")
        print(f"Conviction:  {signal.conviction}")
        
        # Technical details
        details = signal.details
        print(f"\n{'='*100}")
        print("TECHNICAL INDICATORS")
        print(f"{'='*100}")
        print(f"Close:           {details.get('close', 0):.2f}")
        print(f"Open:            {details.get('open', 0):.2f}")
        print(f"High:            {details.get('high', 0):.2f}")
        print(f"Low:             {details.get('low', 0):.2f}")
        print(f"\nEMA 9:           {details.get('ema_9', 0):.2f}")
        print(f"EMA 21:          {details.get('ema_21', 0):.2f}")
        print(f"Prev EMA 9:      {details.get('prev_ema_9', 0):.2f}")
        print(f"Prev EMA 21:     {details.get('prev_ema_21', 0):.2f}")
        print(f"\nRSI 14:          {details.get('rsi_14', 0):.2f}")
        print(f"ATR 14:          {details.get('atr_14', 0):.2f}")
        print(f"Volume Ratio:    {details.get('volume_ratio_20', 0):.2f}x")
        
        # Signal conditions
        print(f"\n{'='*100}")
        print("SIGNAL CONDITIONS")
        print(f"{'='*100}")
        print(f"EMA Cross Up:    {'✅ YES' if details.get('ema_cross_up') else '❌ NO'}")
        print(f"EMA Cross Down:  {'✅ YES' if details.get('ema_cross_down') else '❌ NO'}")
        print(f"RSI Buy OK:      {'✅ YES' if details.get('rsi_buy_ok') else '❌ NO'}")
        print(f"RSI Sell OK:     {'✅ YES' if details.get('rsi_sell_ok') else '❌ NO'}")
        print(f"Bullish Candle:  {'✅ YES' if details.get('bullish_candle') else '❌ NO'}")
        print(f"Bearish Candle:  {'✅ YES' if details.get('bearish_candle') else '❌ NO'}")
        print(f"Volume OK:       {'✅ YES' if details.get('volume_ok') else '❌ NO'}")
        print(f"Break Prev High: {'✅ YES' if details.get('break_prev_high') else '❌ NO'}")
        print(f"Break Prev Low:  {'✅ YES' if details.get('break_prev_low') else '❌ NO'}")
        
        # Scoring
        print(f"\n{'='*100}")
        print("SCORING BREAKDOWN")
        print(f"{'='*100}")
        print(f"Buy Score:       {details.get('score_buy', 0):.1f}/110")
        print(f"Sell Score:      {details.get('score_sell', 0):.1f}/110")
        print(f"Buy Ready:       {'✅ YES (≥50)' if details.get('buy_ready') else '❌ NO (<50)'}")
        print(f"Sell Ready:      {'✅ YES (≥50)' if details.get('sell_ready') else '❌ NO (<50)'}")
        
        # Trade details
        if signal.action in ["BUY", "SELL"]:
            print(f"\n{'='*100}")
            print("TRADE SETUP")
            print(f"{'='*100}")
            print(f"Entry Price:     {signal.entry_price:.2f}")
            print(f"Stop Loss:       {signal.stop_loss:.2f}")
            print(f"Take Profit:     {signal.take_profit:.2f}")
            risk = signal.entry_price - signal.stop_loss if signal.stop_loss else 0
            reward = signal.take_profit - signal.entry_price if signal.take_profit else 0
            rr_ratio = reward / risk if risk > 0 else 0
            print(f"Risk:            {abs(risk):.2f} points")
            print(f"Reward:          {reward:.2f} points")
            print(f"R:R Ratio:       1:{rr_ratio:.2f}")
            
            # Step 3: Option selection
            print("\nStep 3: Selecting option contract...")
            option_selection = build_option_selection(db, context=context, signal=signal, settings=settings)
            
            print(f"\n{'='*100}")
            print("OPTION SELECTION")
            print(f"{'='*100}")
            print(f"Expiry:          {option_selection.expiry_date}")
            print(f"Strike Step:     {option_selection.strike_step}")
            print(f"Chain Source:    {option_selection.chain_source}")
            
            opt_signal = option_selection.signal
            if opt_signal.get('action') == 'BUY':
                print(f"\nOption Type:     {opt_signal.get('option_type')}")
                print(f"Strike:          {opt_signal.get('strike'):.0f}")
                print(f"Entry Premium:   ₹{opt_signal.get('entry_price'):.2f}")
                print(f"Stop Loss:       ₹{opt_signal.get('stop_loss'):.2f}")
                print(f"Take Profit:     ₹{opt_signal.get('take_profit'):.2f}")
                print(f"Instrument Key:  {opt_signal.get('instrument_key')}")
            else:
                print("\n⚠️  No option selected")
                print(f"   Reasons: {', '.join(opt_signal.get('reasons', []))}")
        
        # Reasons
        print(f"\n{'='*100}")
        print("SIGNAL REASONS")
        print(f"{'='*100}")
        for i, reason in enumerate(signal.reasons, 1):
            print(f"{i}. {reason}")
        
        # Guardrails
        if signal.cooldown_seconds > 0:
            print(f"\n⏳ COOLDOWN: {signal.cooldown_seconds}s remaining")
        
        if signal.max_signals_reached:
            print(f"\n🛑 MAX SIGNALS: {details.get('signal_count_today', 0)} signals today (limit reached)")
        
        print(f"\n{'='*100}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def main():
    """Test signal generation for all symbols."""
    settings = get_settings()
    
    print("\n" + "="*100)
    print(f"{'SIGNAL GENERATION TEST':^100}")
    print("="*100)
    print("\nConfiguration:")
    print(f"  Mode: {settings.execution_mode}")
    print(f"  Enabled: {settings.execution_enabled}")
    print(f"  Entry Window: {settings.entry_window_start} - {settings.entry_window_end}")
    print(f"  Symbols: {', '.join(settings.execution_symbol_list)}")
    
    for symbol in settings.execution_symbol_list:
        test_signal_generation(symbol)
    
    print("\n✅ Testing complete!\n")


if __name__ == "__main__":
    main()
