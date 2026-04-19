"""
Broker factory for multi-broker support.
Dynamically creates broker instances based on configuration.
"""
from typing import Any

from brokers.base import BaseBroker
from brokers.upstox_broker import UpstoxBroker
from utils.logger import get_logger

logger = get_logger(__name__)


class BrokerFactory:
    """Factory for creating broker instances."""

    _brokers: dict[str, type[BaseBroker]] = {
        "upstox": UpstoxBroker,
        # Add more brokers here:
        # "zerodha": ZerodhaBroker,
        # "angelone": AngelOneBroker,
        # "fyers": FyersBroker,
    }

    @classmethod
    def create_broker(cls, broker_name: str, config: dict[str, Any]) -> BaseBroker:
        """
        Create a broker instance.

        Args:
            broker_name: Name of the broker (upstox, zerodha, etc.)
            config: Broker configuration dictionary

        Returns:
            BaseBroker instance

        Raises:
            ValueError: If broker is not supported
        """
        broker_name = broker_name.lower().strip()

        if broker_name not in cls._brokers:
            raise ValueError(
                f"Unsupported broker: {broker_name}. "
                f"Supported brokers: {', '.join(cls._brokers.keys())}"
            )

        broker_class = cls._brokers[broker_name]
        broker = broker_class(config)

        logger.info(f"Created broker instance: {broker_name}")
        return broker

    @classmethod
    def register_broker(cls, name: str, broker_class: type[BaseBroker]) -> None:
        """
        Register a new broker implementation.

        Args:
            name: Broker name
            broker_class: Broker class implementing BaseBroker
        """
        cls._brokers[name.lower()] = broker_class
        logger.info(f"Registered broker: {name}")

    @classmethod
    def list_brokers(cls) -> list[str]:
        """List all registered brokers."""
        return list(cls._brokers.keys())
