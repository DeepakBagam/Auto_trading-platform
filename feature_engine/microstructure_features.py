"""Microstructure features for high-frequency trading signals."""
from __future__ import annotations

import pandas as pd
import numpy as np
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from db.models import OrderBookSnapshot, RawCandle
from utils.logger import get_logger

logger = get_logger(__name__)


def build_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build microstructure features from OHLCV data.
    
    Features:
    - Bid-ask spread proxy (high-low range)
    - Volume profile (POC, VAH, VAL)
    - Delta volume (buy vs sell pressure)
    - Tick direction (uptick/downtick ratio)
    - Amihud illiquidity measure
    """
    df = df.copy()
    
    # Spread proxy using high-low range
    df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]
    df["spread_proxy_ma5"] = df["spread_proxy"].rolling(5).mean()
    
    # Volume profile - Price at which most volume traded
    df["volume_weighted_price"] = (df["high"] + df["low"] + df["close"]) / 3.0 * df["volume"]
    df["vwap"] = df["volume_weighted_price"].rolling(20).sum() / df["volume"].rolling(20).sum()
    df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]
    
    # Delta volume (buy pressure - sell pressure proxy)
    # Uptick volume = volume when close > open
    # Downtick volume = volume when close < open
    df["uptick_volume"] = df["volume"] * (df["close"] > df["open"]).astype(float)
    df["downtick_volume"] = df["volume"] * (df["close"] < df["open"]).astype(float)
    df["delta_volume"] = df["uptick_volume"] - df["downtick_volume"]
    df["delta_volume_ma5"] = df["delta_volume"].rolling(5).mean()
    df["delta_volume_ratio"] = df["delta_volume"] / (df["volume"] + 1e-9)
    
    # Cumulative delta volume
    df["cumulative_delta"] = df["delta_volume"].cumsum()
    df["cumulative_delta_normalized"] = df["cumulative_delta"] / (df["volume"].cumsum() + 1e-9)
    
    # Tick direction (uptick/downtick ratio)
    df["price_change"] = df["close"].diff()
    df["uptick"] = (df["price_change"] > 0).astype(float)
    df["downtick"] = (df["price_change"] < 0).astype(float)
    df["uptick_ratio_10"] = df["uptick"].rolling(10).mean()
    df["downtick_ratio_10"] = df["downtick"].rolling(10).mean()
    df["tick_imbalance"] = df["uptick_ratio_10"] - df["downtick_ratio_10"]
    
    # Amihud illiquidity measure: |return| / volume
    df["returns"] = df["close"].pct_change()
    df["amihud_illiquidity"] = df["returns"].abs() / (df["volume"] + 1e-9)
    df["amihud_illiquidity_ma10"] = df["amihud_illiquidity"].rolling(10).mean()
    
    # Volume at price levels (simplified)
    df["volume_at_high"] = df["volume"] * (df["close"] >= df["high"] * 0.98).astype(float)
    df["volume_at_low"] = df["volume"] * (df["close"] <= df["low"] * 1.02).astype(float)
    df["volume_concentration"] = (df["volume_at_high"] + df["volume_at_low"]) / (df["volume"] + 1e-9)
    
    # Order flow toxicity (Kyle's lambda proxy)
    df["price_impact"] = df["returns"].abs() / (df["volume"].pct_change().abs() + 1e-9)
    df["price_impact_ma5"] = df["price_impact"].rolling(5).mean()
    
    # Volume-synchronized probability of informed trading (VPIN proxy)
    df["volume_bucket"] = df["volume"].rolling(50).sum() / 50.0
    df["buy_volume_bucket"] = df["uptick_volume"].rolling(50).sum()
    df["sell_volume_bucket"] = df["downtick_volume"].rolling(50).sum()
    df["vpin_proxy"] = (
        (df["buy_volume_bucket"] - df["sell_volume_bucket"]).abs() 
        / (df["volume_bucket"] + 1e-9)
    )
    
    return df


def load_order_book_features(
    db: Session, 
    instrument_key: str, 
    lookback_minutes: int = 60
) -> pd.DataFrame:
    """Load order book snapshots and compute aggregated features."""
    from datetime import datetime, timedelta
    from utils.constants import IST_ZONE
    
    cutoff = datetime.now(IST_ZONE) - timedelta(minutes=lookback_minutes)
    
    rows = db.execute(
        select(OrderBookSnapshot).where(
            and_(
                OrderBookSnapshot.instrument_key == instrument_key,
                OrderBookSnapshot.ts >= cutoff,
            )
        ).order_by(OrderBookSnapshot.ts.asc())
    ).scalars().all()
    
    if not rows:
        return pd.DataFrame()
    
    data = [
        {
            "ts": r.ts,
            "spread_bps": r.spread_bps,
            "depth_imbalance": r.depth_imbalance,
            "liquidity_score": r.liquidity_score,
            "mid_price": r.mid_price,
        }
        for r in rows
    ]
    
    df = pd.DataFrame(data)
    
    # Aggregate features
    features = {
        "spread_bps_mean": df["spread_bps"].mean(),
        "spread_bps_std": df["spread_bps"].std(),
        "spread_bps_min": df["spread_bps"].min(),
        "spread_bps_max": df["spread_bps"].max(),
        "depth_imbalance_mean": df["depth_imbalance"].mean(),
        "depth_imbalance_std": df["depth_imbalance"].std(),
        "liquidity_score_mean": df["liquidity_score"].mean(),
        "liquidity_score_min": df["liquidity_score"].min(),
    }
    
    return pd.DataFrame([features])


def compute_volume_profile(df: pd.DataFrame, bins: int = 20) -> dict:
    """
    Compute volume profile: POC, VAH, VAL.
    
    POC = Point of Control (price with highest volume)
    VAH = Value Area High (top 70% volume)
    VAL = Value Area Low (bottom 70% volume)
    """
    if df.empty or "close" not in df.columns or "volume" not in df.columns:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0}
    
    price_min = df["low"].min()
    price_max = df["high"].max()
    
    if price_min >= price_max:
        return {"poc": df["close"].iloc[-1], "vah": price_max, "val": price_min}
    
    price_bins = np.linspace(price_min, price_max, bins)
    volume_at_price = np.zeros(bins - 1)
    
    for _, row in df.iterrows():
        low, high, volume = row["low"], row["high"], row["volume"]
        for i in range(len(price_bins) - 1):
            if low <= price_bins[i + 1] and high >= price_bins[i]:
                volume_at_price[i] += volume
    
    poc_idx = np.argmax(volume_at_price)
    poc = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2.0
    
    total_volume = volume_at_price.sum()
    target_volume = total_volume * 0.70
    
    sorted_indices = np.argsort(volume_at_price)[::-1]
    cumulative_volume = 0.0
    value_area_indices = []
    
    for idx in sorted_indices:
        cumulative_volume += volume_at_price[idx]
        value_area_indices.append(idx)
        if cumulative_volume >= target_volume:
            break
    
    vah = price_bins[max(value_area_indices) + 1]
    val = price_bins[min(value_area_indices)]
    
    return {"poc": float(poc), "vah": float(vah), "val": float(val)}
