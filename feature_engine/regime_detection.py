"""Market regime detection for adaptive model selection."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from db.models import RawCandle
from utils.constants import IST_ZONE
from utils.logger import get_logger
from utils.symbols import instrument_key_filter

logger = get_logger(__name__)


class VolatilityRegime(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class TrendRegime(str, Enum):
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGE = "range"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class CorrelationRegime(str, Enum):
    HIGH_POSITIVE = "high_positive"
    MODERATE_POSITIVE = "moderate_positive"
    NEUTRAL = "neutral"
    MODERATE_NEGATIVE = "moderate_negative"
    HIGH_NEGATIVE = "high_negative"


@dataclass
class MarketRegime:
    timestamp: datetime
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    correlation_regime: CorrelationRegime
    vix_level: float
    adx: float
    nifty_vix_correlation: float
    regime_score: float
    details: dict


def detect_volatility_regime(vix_level: float, vix_percentile: float) -> VolatilityRegime:
    """
    Detect volatility regime based on VIX level and historical percentile.
    
    Low: VIX < 15 or percentile < 30
    Medium: VIX 15-25 or percentile 30-70
    High: VIX 25-35 or percentile 70-90
    Extreme: VIX > 35 or percentile > 90
    """
    if vix_level > 35 or vix_percentile > 90:
        return VolatilityRegime.EXTREME
    if vix_level > 25 or vix_percentile > 70:
        return VolatilityRegime.HIGH
    if vix_level > 15 or vix_percentile > 30:
        return VolatilityRegime.MEDIUM
    return VolatilityRegime.LOW


def detect_trend_regime(adx: float, ema_slope: float, price_vs_ema: float) -> TrendRegime:
    """
    Detect trend regime based on ADX, EMA slope, and price position.
    
    Strong trend: ADX > 25
    Weak trend: ADX 20-25
    Range: ADX < 20
    """
    if adx > 25:
        if ema_slope > 0.5 and price_vs_ema > 0.02:
            return TrendRegime.STRONG_UPTREND
        if ema_slope < -0.5 and price_vs_ema < -0.02:
            return TrendRegime.STRONG_DOWNTREND
    
    if adx > 20:
        if ema_slope > 0 and price_vs_ema > 0:
            return TrendRegime.WEAK_UPTREND
        if ema_slope < 0 and price_vs_ema < 0:
            return TrendRegime.WEAK_DOWNTREND
    
    return TrendRegime.RANGE


def detect_correlation_regime(correlation: float) -> CorrelationRegime:
    """
    Detect correlation regime between Nifty and VIX.
    
    Typically negative correlation (VIX up when Nifty down).
    """
    if correlation > 0.5:
        return CorrelationRegime.HIGH_POSITIVE
    if correlation > 0.2:
        return CorrelationRegime.MODERATE_POSITIVE
    if correlation > -0.2:
        return CorrelationRegime.NEUTRAL
    if correlation > -0.5:
        return CorrelationRegime.MODERATE_NEGATIVE
    return CorrelationRegime.HIGH_NEGATIVE


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(period).mean()
    
    return adx


def load_recent_candles(
    db: Session, 
    symbol: str, 
    interval: str = "1minute", 
    lookback_minutes: int = 390
) -> pd.DataFrame:
    """Load recent candles for regime detection."""
    cutoff = datetime.now(IST_ZONE) - timedelta(minutes=lookback_minutes)
    
    rows = db.execute(
        select(RawCandle).where(
            and_(
                instrument_key_filter(RawCandle.instrument_key, symbol),
                RawCandle.interval == interval,
                RawCandle.ts >= cutoff,
            )
        ).order_by(RawCandle.ts.asc())
    ).scalars().all()
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame([
        {
            "ts": r.ts,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        }
        for r in rows
    ])


def detect_market_regime(db: Session, symbol: str = "Nifty 50") -> MarketRegime:
    """
    Detect current market regime for adaptive model selection.
    
    Returns regime classification and confidence score.
    """
    # Load recent data
    nifty_df = load_recent_candles(db, "Nifty 50", interval="1minute", lookback_minutes=390)
    vix_df = load_recent_candles(db, "India VIX", interval="1minute", lookback_minutes=390)
    
    if nifty_df.empty or vix_df.empty:
        return MarketRegime(
            timestamp=datetime.now(IST_ZONE),
            volatility_regime=VolatilityRegime.MEDIUM,
            trend_regime=TrendRegime.RANGE,
            correlation_regime=CorrelationRegime.NEUTRAL,
            vix_level=20.0,
            adx=15.0,
            nifty_vix_correlation=0.0,
            regime_score=0.5,
            details={"error": "insufficient_data"},
        )
    
    # VIX analysis
    current_vix = float(vix_df["close"].iloc[-1])
    vix_history = vix_df["close"].tail(100)
    vix_percentile = float((vix_history < current_vix).sum() / len(vix_history) * 100)
    
    volatility_regime = detect_volatility_regime(current_vix, vix_percentile)
    
    # Trend analysis
    nifty_df["ema_21"] = nifty_df["close"].ewm(span=21).mean()
    nifty_df["ema_50"] = nifty_df["close"].ewm(span=50).mean()
    nifty_df["adx"] = compute_adx(nifty_df, period=14)
    
    current_adx = float(nifty_df["adx"].iloc[-1]) if not nifty_df["adx"].isna().iloc[-1] else 15.0
    ema_slope = float(
        (nifty_df["ema_21"].iloc[-1] - nifty_df["ema_21"].iloc[-10]) 
        / nifty_df["ema_21"].iloc[-10] * 100
    )
    price_vs_ema = float(
        (nifty_df["close"].iloc[-1] - nifty_df["ema_21"].iloc[-1]) 
        / nifty_df["ema_21"].iloc[-1]
    )
    
    trend_regime = detect_trend_regime(current_adx, ema_slope, price_vs_ema)
    
    # Correlation analysis
    nifty_returns = nifty_df["close"].pct_change().tail(100)
    vix_returns = vix_df["close"].pct_change().tail(100)
    
    if len(nifty_returns) == len(vix_returns):
        correlation = float(nifty_returns.corr(vix_returns))
    else:
        correlation = -0.5  # Default negative correlation
    
    correlation_regime = detect_correlation_regime(correlation)
    
    # Regime score (0-1, higher = more confident)
    regime_score = 0.5
    
    # High confidence in extreme volatility
    if volatility_regime == VolatilityRegime.EXTREME:
        regime_score += 0.3
    elif volatility_regime == VolatilityRegime.LOW:
        regime_score += 0.2
    
    # High confidence in strong trends
    if trend_regime in [TrendRegime.STRONG_UPTREND, TrendRegime.STRONG_DOWNTREND]:
        regime_score += 0.2
    
    regime_score = min(1.0, regime_score)
    
    return MarketRegime(
        timestamp=datetime.now(IST_ZONE),
        volatility_regime=volatility_regime,
        trend_regime=trend_regime,
        correlation_regime=correlation_regime,
        vix_level=current_vix,
        adx=current_adx,
        nifty_vix_correlation=correlation,
        regime_score=regime_score,
        details={
            "vix_percentile": vix_percentile,
            "ema_slope": ema_slope,
            "price_vs_ema": price_vs_ema,
            "symbol": symbol,
        },
    )


def should_use_regime_model(regime: MarketRegime) -> bool:
    """Determine if regime-specific model should be used."""
    # Use regime model if confidence is high
    if regime.regime_score > 0.7:
        return True
    
    # Always use regime model in extreme volatility
    if regime.volatility_regime == VolatilityRegime.EXTREME:
        return True
    
    # Use regime model in strong trends
    if regime.trend_regime in [TrendRegime.STRONG_UPTREND, TrendRegime.STRONG_DOWNTREND]:
        return True
    
    return False
