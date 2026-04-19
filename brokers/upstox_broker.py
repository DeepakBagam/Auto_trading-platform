"""Upstox broker implementation."""
from datetime import datetime
from typing import Any

import httpx
from upstox_client import ApiClient, Configuration

from brokers.base import (
    BaseBroker,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    Position,
    Quote,
)
from observability.metrics import execution_latency_seconds
from utils.logger import get_logger

logger = get_logger(__name__)


class UpstoxBroker(BaseBroker):
    """Upstox broker implementation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.access_token = config.get("access_token")
        self.api_key = config.get("api_key")
        self.api_secret = config.get("api_secret")
        self.base_url = config.get("base_url", "https://api.upstox.com")
        self._client: ApiClient | None = None

    @property
    def broker_name(self) -> str:
        return "upstox"

    def authenticate(self) -> bool:
        """Authenticate with Upstox API."""
        if not self.access_token:
            logger.error("Upstox access token not provided")
            return False

        try:
            configuration = Configuration()
            configuration.access_token = self.access_token
            self._client = ApiClient(configuration)
            self._authenticated = True
            logger.info("Upstox authentication successful")
            return True
        except Exception as e:
            logger.exception("Upstox authentication failed")
            self._authenticated = False
            return False

    def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place an order with Upstox."""
        import time

        start_time = time.time()

        try:
            # Map to Upstox format
            payload = {
                "quantity": order.quantity,
                "product": self._map_product_type(order.product_type),
                "validity": order.validity,
                "price": order.price or 0,
                "tag": order.tag or "automated_trading",
                "instrument_token": self._get_instrument_token(order.symbol, order.exchange),
                "order_type": order.order_type.value,
                "transaction_type": order.side.value,
                "disclosed_quantity": order.disclosed_quantity,
                "trigger_price": order.trigger_price or 0,
                "is_amo": False,
            }

            response = self._make_request("POST", "/v2/order/place", json=payload)

            duration = time.time() - start_time
            execution_latency_seconds.labels(broker="upstox", order_type=order.order_type.value).observe(duration)

            if response.get("status") == "success":
                order_id = response.get("data", {}).get("order_id")
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.PENDING,
                    message="Order placed successfully",
                    broker_response=response,
                )
            else:
                return OrderResponse(
                    order_id="",
                    status=OrderStatus.REJECTED,
                    message=response.get("message", "Order placement failed"),
                    broker_response=response,
                )

        except Exception as e:
            logger.exception("Failed to place order with Upstox")
            return OrderResponse(
                order_id="",
                status=OrderStatus.REJECTED,
                message=str(e),
                broker_response={},
            )

    def modify_order(self, order_id: str, quantity: int | None = None, price: float | None = None) -> OrderResponse:
        """Modify an existing order."""
        try:
            payload = {"order_id": order_id}
            if quantity is not None:
                payload["quantity"] = quantity
            if price is not None:
                payload["price"] = price

            response = self._make_request("PUT", "/v2/order/modify", json=payload)

            if response.get("status") == "success":
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.OPEN,
                    message="Order modified successfully",
                    broker_response=response,
                )
            else:
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    message=response.get("message", "Order modification failed"),
                    broker_response=response,
                )

        except Exception as e:
            logger.exception("Failed to modify order")
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message=str(e),
                broker_response={},
            )

    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an order."""
        try:
            response = self._make_request("DELETE", f"/v2/order/cancel?order_id={order_id}")

            if response.get("status") == "success":
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.CANCELLED,
                    message="Order cancelled successfully",
                    broker_response=response,
                )
            else:
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    message=response.get("message", "Order cancellation failed"),
                    broker_response=response,
                )

        except Exception as e:
            logger.exception("Failed to cancel order")
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message=str(e),
                broker_response={},
            )

    def get_order_status(self, order_id: str) -> OrderResponse:
        """Get order status."""
        try:
            response = self._make_request("GET", f"/v2/order/details?order_id={order_id}")

            if response.get("status") == "success":
                data = response.get("data", {})
                status_map = {
                    "open": OrderStatus.OPEN,
                    "complete": OrderStatus.COMPLETE,
                    "cancelled": OrderStatus.CANCELLED,
                    "rejected": OrderStatus.REJECTED,
                }
                status = status_map.get(data.get("status", "").lower(), OrderStatus.PENDING)

                return OrderResponse(
                    order_id=order_id,
                    status=status,
                    message=data.get("status_message", ""),
                    broker_response=response,
                )
            else:
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    message=response.get("message", "Failed to get order status"),
                    broker_response=response,
                )

        except Exception as e:
            logger.exception("Failed to get order status")
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message=str(e),
                broker_response={},
            )

    def get_positions(self) -> list[Position]:
        """Get all positions."""
        try:
            response = self._make_request("GET", "/v2/portfolio/short-term-positions")

            if response.get("status") == "success":
                positions = []
                for pos_data in response.get("data", []):
                    positions.append(
                        Position(
                            symbol=pos_data.get("tradingsymbol", ""),
                            exchange=pos_data.get("exchange", ""),
                            product_type=pos_data.get("product", ""),
                            quantity=pos_data.get("quantity", 0),
                            average_price=pos_data.get("average_price", 0.0),
                            last_price=pos_data.get("last_price", 0.0),
                            pnl=pos_data.get("pnl", 0.0),
                            day_pnl=pos_data.get("day_pnl", 0.0),
                        )
                    )
                return positions
            else:
                logger.error("Failed to get positions: %s", response.get("message"))
                return []

        except Exception as e:
            logger.exception("Failed to get positions")
            return []

    def get_quote(self, symbol: str, exchange: str) -> Quote:
        """Get market quote."""
        try:
            instrument_key = f"{exchange}|{symbol}"
            response = self._make_request("GET", f"/v2/market-quote/quotes?instrument_key={instrument_key}")

            if response.get("status") == "success":
                data = response.get("data", {}).get(instrument_key, {})
                ohlc = data.get("ohlc", {})
                depth = data.get("depth", {})

                return Quote(
                    symbol=symbol,
                    exchange=exchange,
                    ltp=data.get("last_price", 0.0),
                    open=ohlc.get("open", 0.0),
                    high=ohlc.get("high", 0.0),
                    low=ohlc.get("low", 0.0),
                    close=ohlc.get("close", 0.0),
                    volume=data.get("volume", 0),
                    bid=depth.get("buy", [{}])[0].get("price", 0.0) if depth.get("buy") else 0.0,
                    ask=depth.get("sell", [{}])[0].get("price", 0.0) if depth.get("sell") else 0.0,
                    timestamp=datetime.now(),
                )
            else:
                raise ValueError(f"Failed to get quote: {response.get('message')}")

        except Exception as e:
            logger.exception("Failed to get quote")
            raise

    def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> list[dict[str, Any]]:
        """Get historical candle data."""
        try:
            instrument_key = f"{exchange}|{symbol}"
            params = {
                "instrument_key": instrument_key,
                "interval": interval,
                "from_date": from_date.isoformat(),
                "to_date": to_date.isoformat(),
            }

            response = self._make_request("GET", "/v2/historical-candle/intraday", params=params)

            if response.get("status") == "success":
                return response.get("data", {}).get("candles", [])
            else:
                logger.error("Failed to get historical data: %s", response.get("message"))
                return []

        except Exception as e:
            logger.exception("Failed to get historical data")
            return []

    def _make_request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        """Make HTTP request to Upstox API."""
        if not self._authenticated:
            raise RuntimeError("Broker not authenticated")

        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json()

    def _map_product_type(self, product_type) -> str:
        """Map product type to Upstox format."""
        mapping = {
            "INTRADAY": "I",
            "DELIVERY": "D",
            "MARGIN": "M",
        }
        return mapping.get(str(product_type.value), "I")

    def _get_instrument_token(self, symbol: str, exchange: str) -> str:
        """Get instrument token for symbol."""
        return f"{exchange}|{symbol}"
