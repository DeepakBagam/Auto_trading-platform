"""Slippage estimation and execution quality tracking."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from db.models import ExecutionOrder, OrderBookSnapshot
from utils.constants import IST_ZONE
from utils.logger import get_logger
from utils.symbols import instrument_key_filter

logger = get_logger(__name__)


@dataclass
class SlippageEstimate:
    estimated_slippage_bps: float
    spread_component_bps: float
    volume_impact_bps: float
    volatility_premium_bps: float
    time_premium_bps: float
    confidence: float
    details: dict


def get_current_spread(db: Session, instrument_key: str) -> float:
    """Get current bid-ask spread in basis points."""
    snapshot = db.scalar(
        select(OrderBookSnapshot)
        .where(OrderBookSnapshot.instrument_key == instrument_key)
        .order_by(OrderBookSnapshot.ts.desc())
        .limit(1)
    )
    
    if snapshot and snapshot.spread_bps is not None:
        return float(snapshot.spread_bps)
    
    # Fallback: estimate from recent candles
    cutoff = datetime.now(IST_ZONE) - timedelta(minutes=5)
    from db.models import RawCandle
    
    candles = db.execute(
        select(RawCandle)
        .where(
            and_(
                RawCandle.instrument_key == instrument_key,
                RawCandle.interval == "1minute",
                RawCandle.ts >= cutoff,
            )
        )
        .order_by(RawCandle.ts.desc())
        .limit(5)
    ).scalars().all()
    
    if not candles:
        return 10.0  # Default 10 bps
    
    # Estimate spread as average (high-low) / close
    spreads = [
        (c.high - c.low) / c.close * 10000.0
        for c in candles
        if c.close > 0
    ]
    
    return sum(spreads) / len(spreads) if spreads else 10.0


def get_avg_volume(db: Session, instrument_key: str, minutes: int = 5) -> float:
    """Get average volume over recent period."""
    from db.models import RawCandle
    
    cutoff = datetime.now(IST_ZONE) - timedelta(minutes=minutes)
    
    avg_vol = db.scalar(
        select(func.avg(RawCandle.volume))
        .where(
            and_(
                RawCandle.instrument_key == instrument_key,
                RawCandle.interval == "1minute",
                RawCandle.ts >= cutoff,
            )
        )
    )
    
    return float(avg_vol) if avg_vol else 1000.0


def get_vix_level(db: Session) -> float:
    """Get current VIX level."""
    from db.models import RawCandle
    
    vix_candle = db.scalar(
        select(RawCandle)
        .where(
            and_(
                instrument_key_filter(RawCandle.instrument_key, "India VIX"),
                RawCandle.interval == "1minute",
            )
        )
        .order_by(RawCandle.ts.desc())
        .limit(1)
    )
    
    return float(vix_candle.close) if vix_candle else 20.0


def get_vix_context(db: Session) -> tuple[float, float, float]:
    """Return (current_vix, vix_ma20, vix_ratio).

    vix_ratio = current / ma20.  >1.40 = spike; <0.70 & level <12 = too calm.
    Falls back to (20.0, 20.0, 1.0) when no VIX data is available.
    """
    from db.models import RawCandle

    candles = (
        db.execute(
            select(RawCandle)
            .where(
                and_(
                    instrument_key_filter(RawCandle.instrument_key, "India VIX"),
                    RawCandle.interval == "1minute",
                )
            )
            .order_by(RawCandle.ts.desc())
            .limit(200)
        )
        .scalars()
        .all()
    )
    if not candles:
        return 20.0, 20.0, 1.0

    closes = [float(c.close) for c in candles]
    current_vix = closes[0]
    window = closes[: min(20, len(closes))]
    ma20 = sum(window) / len(window)
    ratio = current_vix / ma20 if ma20 > 0 else 1.0
    return current_vix, ma20, ratio


def get_time_of_day_premium(now: datetime) -> float:
    """
    Get time-of-day slippage premium.
    
    Higher slippage during:
    - Market open (9:15-9:30): 3-5 bps
    - Market close (15:15-15:30): 3-5 bps
    - Lunch hour (12:00-13:00): 1-2 bps (low liquidity)
    """
    current_time = now.time()
    
    # Market open - high slippage
    if time(9, 15) <= current_time < time(9, 30):
        return 5.0
    
    # Market close - high slippage
    if time(15, 15) <= current_time <= time(15, 30):
        return 4.0
    
    # Early morning volatility
    if time(9, 30) <= current_time < time(10, 0):
        return 2.0
    
    # Lunch hour - low liquidity
    if time(12, 0) <= current_time < time(13, 0):
        return 1.5
    
    # Normal hours
    return 0.5


def estimate_slippage(
    db: Session,
    symbol: str,
    instrument_key: str,
    quantity: int,
    order_type: str = "MARKET",
    side: str = "BUY",
    now: datetime | None = None,
) -> SlippageEstimate:
    """
    Estimate slippage for an order.
    
    Slippage components:
    1. Spread component: Half the bid-ask spread (crossing the spread)
    2. Volume impact: Order size relative to average volume
    3. Volatility premium: Higher VIX = higher slippage
    4. Time premium: Time of day effects
    
    Formula:
    slippage_bps = (spread/2) + (volume_impact * 10) + (vix_level * 0.5) + time_premium
    """
    now = now or datetime.now(IST_ZONE)
    
    # Get spread
    spread_bps = get_current_spread(db, instrument_key)
    spread_component = spread_bps / 2.0  # Cross half the spread
    
    # Get volume impact
    avg_volume = get_avg_volume(db, instrument_key, minutes=5)
    volume_ratio = quantity / max(avg_volume, 1.0)
    
    # Volume impact increases non-linearly
    # Small orders (<5% of volume): minimal impact
    # Medium orders (5-20%): moderate impact
    # Large orders (>20%): high impact
    if volume_ratio < 0.05:
        volume_impact_bps = volume_ratio * 5.0
    elif volume_ratio < 0.20:
        volume_impact_bps = 0.25 + (volume_ratio - 0.05) * 15.0
    else:
        volume_impact_bps = 2.5 + (volume_ratio - 0.20) * 25.0
    
    # Get volatility premium
    vix_level = get_vix_level(db)
    volatility_premium_bps = vix_level * 0.5
    
    # Get time premium
    time_premium_bps = get_time_of_day_premium(now)
    
    # LIMIT orders have lower slippage than MARKET orders
    order_type_multiplier = 0.5 if order_type == "LIMIT" else 1.0
    
    # Total slippage
    total_slippage_bps = (
        spread_component * order_type_multiplier
        + volume_impact_bps
        + volatility_premium_bps
        + time_premium_bps
    )
    
    # Confidence based on data availability
    confidence = 0.5
    if spread_bps < 50.0:  # Have recent spread data
        confidence += 0.2
    if avg_volume > 100:  # Have volume data
        confidence += 0.2
    if vix_level > 0:  # Have VIX data
        confidence += 0.1
    
    return SlippageEstimate(
        estimated_slippage_bps=round(total_slippage_bps, 2),
        spread_component_bps=round(spread_component, 2),
        volume_impact_bps=round(volume_impact_bps, 2),
        volatility_premium_bps=round(volatility_premium_bps, 2),
        time_premium_bps=round(time_premium_bps, 2),
        confidence=round(confidence, 2),
        details={
            "spread_bps": spread_bps,
            "avg_volume": avg_volume,
            "volume_ratio": volume_ratio,
            "vix_level": vix_level,
            "order_type": order_type,
            "order_type_multiplier": order_type_multiplier,
        },
    )


def track_realized_slippage(
    db: Session,
    order: ExecutionOrder,
    expected_price: float,
    actual_price: float,
) -> dict:
    """
    Track realized slippage vs estimated.
    
    Store in order metadata for analysis.
    """
    if expected_price <= 0 or actual_price <= 0:
        return {"error": "invalid_prices"}
    
    realized_slippage_bps = abs(actual_price - expected_price) / expected_price * 10000.0
    
    # Get estimated slippage from order metadata
    metadata = order.response_json or {}
    estimated_slippage_bps = metadata.get("estimated_slippage_bps", 0.0)
    
    slippage_error = realized_slippage_bps - estimated_slippage_bps
    
    # Update order metadata
    if order.response_json is None:
        order.response_json = {}
    
    order.response_json["slippage_tracking"] = {
        "expected_price": expected_price,
        "actual_price": actual_price,
        "realized_slippage_bps": round(realized_slippage_bps, 2),
        "estimated_slippage_bps": round(estimated_slippage_bps, 2),
        "slippage_error_bps": round(slippage_error, 2),
        "slippage_error_pct": round(slippage_error / max(estimated_slippage_bps, 1.0) * 100, 2),
    }
    
    db.commit()
    
    return order.response_json["slippage_tracking"]


def get_average_slippage(
    db: Session,
    symbol: str | None = None,
    lookback_days: int = 30,
) -> dict:
    """Get average realized slippage over recent period."""
    from utils.symbols import symbol_value_filter
    
    cutoff = datetime.now(IST_ZONE) - timedelta(days=lookback_days)
    
    query = select(ExecutionOrder).where(
        and_(
            ExecutionOrder.created_at >= cutoff,
            ExecutionOrder.order_kind == "ENTRY",
        )
    )
    
    if symbol:
        query = query.where(symbol_value_filter(ExecutionOrder.symbol, symbol))
    
    orders = db.execute(query).scalars().all()
    
    if not orders:
        return {
            "total_orders": 0,
            "avg_realized_slippage_bps": 0.0,
            "avg_estimated_slippage_bps": 0.0,
            "avg_error_bps": 0.0,
        }
    
    realized_slippages = []
    estimated_slippages = []
    errors = []
    
    for order in orders:
        tracking = (order.response_json or {}).get("slippage_tracking", {})
        if tracking:
            realized_slippages.append(tracking.get("realized_slippage_bps", 0.0))
            estimated_slippages.append(tracking.get("estimated_slippage_bps", 0.0))
            errors.append(tracking.get("slippage_error_bps", 0.0))
    
    if not realized_slippages:
        return {
            "total_orders": len(orders),
            "avg_realized_slippage_bps": 0.0,
            "avg_estimated_slippage_bps": 0.0,
            "avg_error_bps": 0.0,
        }
    
    return {
        "total_orders": len(orders),
        "avg_realized_slippage_bps": round(sum(realized_slippages) / len(realized_slippages), 2),
        "avg_estimated_slippage_bps": round(sum(estimated_slippages) / len(estimated_slippages), 2),
        "avg_error_bps": round(sum(errors) / len(errors), 2),
        "median_realized_slippage_bps": round(sorted(realized_slippages)[len(realized_slippages) // 2], 2),
    }
