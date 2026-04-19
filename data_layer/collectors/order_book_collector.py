                                                                                                                     """Order book and microstructure data collector for Upstox."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from db.models import OrderBookSnapshot
from sqlalchemy.orm import Session
from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)


class OrderBookCollector:
    def __init__(self, base_url: str, access_token: str):
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

    def collect_order_book(self, instrument_key: str, depth: int = 5) -> dict[str, Any] | None:
        """Collect order book snapshot with bid/ask levels."""
        try:
            url = f"{self.base_url}/v2/market-quote/quotes"
            params = {"instrument_key": instrument_key}
            
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
            
            if data.get("status") != "success":
                return None
            
            quote_data = data.get("data", {}).get(instrument_key, {})
            depth_data = quote_data.get("depth", {})
            
            bids = depth_data.get("buy", [])[:depth]
            asks = depth_data.get("sell", [])[:depth]
            
            if not bids or not asks:
                return None
            
            best_bid = float(bids[0].get("price", 0))
            best_ask = float(asks[0].get("price", 0))
            mid_price = (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0
            
            spread_bps = ((best_ask - best_bid) / mid_price * 10000.0) if mid_price > 0 else 0.0
            
            bid_volume = sum(float(b.get("quantity", 0)) for b in bids)
            ask_volume = sum(float(a.get("quantity", 0)) for a in asks)
            total_volume = bid_volume + ask_volume
            
            depth_imbalance = (
                (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0.0
            )
            
            liquidity_score = min(100.0, (total_volume / 1000.0) * (1.0 / max(spread_bps, 1.0)))
            
            return {
                "instrument_key": instrument_key,
                "timestamp": datetime.now(IST_ZONE),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": mid_price,
                "spread_bps": spread_bps,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "depth_imbalance": depth_imbalance,
                "liquidity_score": liquidity_score,
                "bids": bids,
                "asks": asks,
            }
        except Exception as exc:
            logger.error("Order book collection failed for %s: %s", instrument_key, exc)
            return None

    def store_order_book_snapshot(self, db: Session, snapshot: dict[str, Any]) -> None:
        """Store order book snapshot in database."""
        try:
            db.add(
                OrderBookSnapshot(
                    instrument_key=snapshot["instrument_key"],
                    ts=snapshot["timestamp"],
                    best_bid=snapshot["best_bid"],
                    best_ask=snapshot["best_ask"],
                    mid_price=snapshot["mid_price"],
                    spread_bps=snapshot["spread_bps"],
                    bid_volume=snapshot["bid_volume"],
                    ask_volume=snapshot["ask_volume"],
                    depth_imbalance=snapshot["depth_imbalance"],
                    liquidity_score=snapshot["liquidity_score"],
                    depth_data={"bids": snapshot["bids"], "asks": snapshot["asks"]},
                )
            )
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.error("Failed to store order book snapshot: %s", exc)

    def collect_tick_data(self, instrument_key: str) -> dict[str, Any] | None:
        """Collect tick-by-tick data for volume analysis."""
        try:
            url = f"{self.base_url}/v2/market-quote/ltp"
            params = {"instrument_key": instrument_key}
            
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
            
            if data.get("status") != "success":
                return None
            
            quote = data.get("data", {}).get(instrument_key, {})
            
            return {
                "instrument_key": instrument_key,
                "timestamp": datetime.now(IST_ZONE),
                "ltp": float(quote.get("last_price", 0)),
                "volume": float(quote.get("volume", 0)),
                "oi": float(quote.get("oi", 0)),
            }
        except Exception as exc:
            logger.error("Tick data collection failed for %s: %s", instrument_key, exc)
            return None
