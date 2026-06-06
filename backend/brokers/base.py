"""
Abstract broker interface for multi-broker support.
Enables switching between Upstox, Zerodha, Angel One, etc.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "SL"
    STOP_LOSS_MARKET = "SL-M"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class ProductType(Enum):
    """Product type enumeration."""

    INTRADAY = "INTRADAY"
    DELIVERY = "DELIVERY"
    MARGIN = "MARGIN"


@dataclass
class OrderRequest:
    """Order request data structure."""

    symbol: str
    exchange: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    product_type: ProductType
    price: float | None = None
    trigger_price: float | None = None
    validity: str = "DAY"
    disclosed_quantity: int = 0
    tag: str | None = None


@dataclass
class OrderResponse:
    """Order response data structure."""

    order_id: str
    status: OrderStatus
    message: str
    broker_response: dict[str, Any]


@dataclass
class Position:
    """Position data structure."""

    symbol: str
    exchange: str
    product_type: ProductType
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    day_pnl: float


@dataclass
class Quote:
    """Market quote data structure."""

    symbol: str
    exchange: str
    ltp: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: float
    ask: float
    timestamp: datetime


class BaseBroker(ABC):
    """Abstract base class for broker implementations."""

    def __init__(self, config: dict[str, Any]):
        """Initialize broker with configuration."""
        self.config = config
        self._authenticated = False

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with broker API."""
        pass

    @abstractmethod
    def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place an order."""
        pass

    @abstractmethod
    def modify_order(self, order_id: str, quantity: int | None = None, price: float | None = None) -> OrderResponse:
        """Modify an existing order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an order."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResponse:
        """Get order status."""
        pass

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all positions."""
        pass

    @abstractmethod
    def get_quote(self, symbol: str, exchange: str) -> Quote:
        """Get market quote for a symbol."""
        pass

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> list[dict[str, Any]]:
        """Get historical candle data."""
        pass

    @property
    def is_authenticated(self) -> bool:
        """Check if broker is authenticated."""
        return self._authenticated

    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Get broker name."""
        pass
