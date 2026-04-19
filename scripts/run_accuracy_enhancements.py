"""Run all accuracy enhancement modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from sqlalchemy.orm import Session

from db.connection import get_db
from feature_engine.regime_detection import detect_market_regime
from models.online_learning import should_update_online_model, update_online_model
from prediction_engine.adaptive_thresholds import compute_adaptive_thresholds
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def run_regime_detection(db: Session):
    """Detect current market regime."""
    logger.info("=" * 80)
    logger.info("REGIME DETECTION")
    logger.info("=" * 80)
    
    regime = detect_market_regime(db, symbol="Nifty 50")
    
    logger.info("Current Market Regime:")
    logger.info("  Timestamp: %s", regime.timestamp)
    logger.info("  Volatility: %s (VIX: %.2f)", regime.volatility_regime, regime.vix_level)
    logger.info("  Trend: %s (ADX: %.2f)", regime.trend_regime, regime.adx)
    logger.info("  Correlation: %s (Nifty-VIX: %.2f)", regime.correlation_regime, regime.nifty_vix_correlation)
    logger.info("  Regime Score: %.2f", regime.regime_score)
    logger.info("  Details: %s", regime.details)
    
    return regime


def run_adaptive_thresholds(db: Session, symbols: list[str]):
    """Compute adaptive thresholds for all symbols."""
    logger.info("=" * 80)
    logger.info("ADAPTIVE THRESHOLDS")
    logger.info("=" * 80)
    
    settings = get_settings()
    results = {}
    
    for symbol in symbols:
        logger.info("\nSymbol: %s", symbol)
        
        thresholds = compute_adaptive_thresholds(
            db,
            symbol=symbol,
            base_ml_threshold=float(settings.ml_buy_threshold),
            base_combined_threshold=float(settings.combined_score_threshold),
            base_ai_minimum=float(settings.ai_quality_minimum),
            base_expected_move=float(settings.ml_min_expected_move),
        )
        
        logger.info("  ML Threshold: %.3f (base: %.3f)", 
                   thresholds.ml_buy_threshold, settings.ml_buy_threshold)
        logger.info("  Combined Threshold: %.3f (base: %.3f)", 
                   thresholds.combined_score_threshold, settings.combined_score_threshold)
        logger.info("  AI Minimum: %.1f (base: %.1f)", 
                   thresholds.ai_quality_minimum, settings.ai_quality_minimum)
        logger.info("  Expected Move: %.1f (base: %.1f)", 
                   thresholds.min_expected_move, settings.ml_min_expected_move)
        logger.info("  Reason: %s", thresholds.reason)
        
        results[symbol] = thresholds
    
    return results


def run_online_learning_updates(db: Session, symbols: list[str]):
    """Update online learning models for all symbols."""
    logger.info("=" * 80)
    logger.info("ONLINE LEARNING UPDATES")
    logger.info("=" * 80)
    
    results = {}
    
    for symbol in symbols:
        logger.info("\nSymbol: %s", symbol)
        
        if should_update_online_model(db, symbol):
            logger.info("  Updating online model...")
            result = update_online_model(db, symbol)
            logger.info("  Status: %s", result.get("status"))
            logger.info("  Update Count: %s", result.get("update_count", 0))
            logger.info("  Rows Processed: %s", result.get("rows_processed", 0))
            results[symbol] = result
        else:
            logger.info("  Online model is up to date")
            results[symbol] = {"status": "up_to_date"}
    
    return results


def run_performance_summary(db: Session, symbols: list[str]):
    """Display recent performance summary."""
    logger.info("=" * 80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 80)
    
    from prediction_engine.adaptive_thresholds import get_recent_performance
    
    for symbol in symbols:
        logger.info("\nSymbol: %s", symbol)
        
        perf = get_recent_performance(db, symbol=symbol, lookback_days=30)
        
        logger.info("  Total Trades: %d", perf["total_trades"])
        logger.info("  Win Rate: %.2f%%", perf["win_rate"] * 100)
        logger.info("  Avg Profit: %.2f", perf["avg_profit"])
        logger.info("  Avg Loss: %.2f", perf["avg_loss"])
        logger.info("  Profit Factor: %.2f", perf["profit_factor"])
        logger.info("  Max Consecutive Losses: %d", perf["max_consecutive_losses"])


def run_slippage_analysis(db: Session, symbols: list[str]):
    """Analyze realized slippage."""
    logger.info("=" * 80)
    logger.info("SLIPPAGE ANALYSIS")
    logger.info("=" * 80)
    
    from execution_engine.slippage_tracker import get_average_slippage
    
    for symbol in symbols:
        logger.info("\nSymbol: %s", symbol)
        
        slippage = get_average_slippage(db, symbol=symbol, lookback_days=30)
        
        logger.info("  Total Orders: %d", slippage["total_orders"])
        logger.info("  Avg Realized Slippage: %.2f bps", slippage["avg_realized_slippage_bps"])
        logger.info("  Avg Estimated Slippage: %.2f bps", slippage["avg_estimated_slippage_bps"])
        logger.info("  Avg Error: %.2f bps", slippage["avg_error_bps"])
        
        if slippage["total_orders"] > 0:
            logger.info("  Median Realized Slippage: %.2f bps", 
                       slippage.get("median_realized_slippage_bps", 0.0))


def main():
    """Run all accuracy enhancements."""
    logger.info("=" * 80)
    logger.info("ACCURACY ENHANCEMENT SUITE")
    logger.info("Started at: %s", datetime.now())
    logger.info("=" * 80)
    
    settings = get_settings()
    symbols = [
        key.split("|", 1)[1] if "|" in key else key
        for key in settings.instrument_keys
    ]
    
    logger.info("Symbols: %s", symbols)
    
    db = next(get_db())
    
    try:
        # 1. Regime Detection
        regime = run_regime_detection(db)
        
        # 2. Adaptive Thresholds
        thresholds = run_adaptive_thresholds(db, symbols)
        
        # 3. Online Learning Updates
        online_results = run_online_learning_updates(db, symbols)
        
        # 4. Performance Summary
        run_performance_summary(db, symbols)
        
        # 5. Slippage Analysis
        run_slippage_analysis(db, symbols)
        
        logger.info("=" * 80)
        logger.info("ACCURACY ENHANCEMENT SUITE COMPLETED")
        logger.info("Finished at: %s", datetime.now())
        logger.info("=" * 80)
        
    except Exception as exc:
        logger.exception("Accuracy enhancement suite failed: %s", exc)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
