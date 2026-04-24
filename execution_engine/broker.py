from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import count
from typing import Any

import requests

from utils.constants import IST_ZONE
from utils.logger import get_logger

logger = get_logger(__name__)

_REDACTED = "***REDACTED***"


def _mask_token(token: str) -> str:
    if not token or len(token) < 8:
        return _REDACTED
    return token[:4] + "..." + token[-4:]


@dataclass(slots=True)
class BrokerOrderRequest:
    instrument_key: str  # Upstox key: "{exchange_segment}|{token}", e.g. "NSE_FO|57807"
    option_type: str
    strike: float
    expiry_date: str
    side: str
    qty: int
    order_type: str = "MARKET"
    price: float | None = None
    trigger_price: float | None = None
    product: str = "MIS"
    tag: str | None = None


@dataclass(slots=True)
class BrokerOrderResponse:
    success: bool
    order_id: str | None
    status: str
    message: str
    payload: dict[str, Any]


class BaseBroker:
    broker_name = "base"

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResponse:
        raise NotImplementedError

    def modify_order(
        self, order_id: str, *, trigger_price: float | None = None, price: float | None = None
    ) -> BrokerOrderResponse:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> BrokerOrderResponse:
        raise NotImplementedError

    def cancel_all_pending(self) -> BrokerOrderResponse:
        raise NotImplementedError


class PaperBroker(BaseBroker):
    broker_name = "paper"

    def __init__(self) -> None:
        self._counter = count(1)
        self._orders: dict[str, dict[str, Any]] = {}

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResponse:
        order_id = f"PAPER-{next(self._counter):08d}"
        status = "TRIGGER_PENDING" if request.trigger_price is not None else "FILLED"
        self._orders[order_id] = {
            "request": request,
            "status": status,
            "created_at": datetime.now(IST_ZONE).isoformat(),
        }
        return BrokerOrderResponse(
            success=True,
            order_id=order_id,
            status=status,
            message="paper_order_accepted",
            payload={"broker": self.broker_name},
        )

    def modify_order(
        self, order_id: str, *, trigger_price: float | None = None, price: float | None = None
    ) -> BrokerOrderResponse:
        row = self._orders.get(order_id)
        if row is None:
            return BrokerOrderResponse(
                success=False,
                order_id=order_id,
                status="REJECTED",
                message="order_not_found",
                payload={},
            )
        req: BrokerOrderRequest = row["request"]
        row["request"] = BrokerOrderRequest(
            instrument_key=req.instrument_key,
            option_type=req.option_type,
            strike=req.strike,
            expiry_date=req.expiry_date,
            side=req.side,
            qty=req.qty,
            order_type=req.order_type,
            price=price if price is not None else req.price,
            trigger_price=trigger_price if trigger_price is not None else req.trigger_price,
            product=req.product,
            tag=req.tag,
        )
        return BrokerOrderResponse(
            success=True,
            order_id=order_id,
            status="MODIFIED",
            message="paper_order_modified",
            payload={},
        )

    def cancel_order(self, order_id: str) -> BrokerOrderResponse:
        row = self._orders.get(order_id)
        if row is None:
            return BrokerOrderResponse(
                success=False,
                order_id=order_id,
                status="REJECTED",
                message="order_not_found",
                payload={},
            )
        row["status"] = "CANCELLED"
        return BrokerOrderResponse(
            success=True,
            order_id=order_id,
            status="CANCELLED",
            message="paper_order_cancelled",
            payload={},
        )

    def cancel_all_pending(self) -> BrokerOrderResponse:
        cancelled = 0
        for order_id, row in self._orders.items():
            if row.get("status") in {"OPEN", "TRIGGER_PENDING"}:
                row["status"] = "CANCELLED"
                cancelled += 1
                logger.info("Paper broker cancelled pending order=%s", order_id)
        return BrokerOrderResponse(
            success=True,
            order_id=None,
            status="OK",
            message=f"cancelled={cancelled}",
            payload={"cancelled": cancelled},
        )


class UpstoxBroker(BaseBroker):
    broker_name = "upstox"

    # Circuit breaker: open after this many consecutive failures
    _CB_FAILURE_THRESHOLD = 5
    _CB_RESET_SECONDS = 60

    # Token refresh: re-read env/file at most once per minute to avoid syscall overhead
    _TOKEN_CHECK_INTERVAL_SECONDS = 60

    def __init__(self, *, base_url: str, access_token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._consecutive_failures = 0
        self._circuit_open_until: datetime | None = None
        self._last_token_check: datetime | None = None
        self._token_set_at: datetime = datetime.now(IST_ZONE)

    def _is_circuit_open(self) -> bool:
        if self._circuit_open_until is None:
            return False
        if datetime.now(IST_ZONE) >= self._circuit_open_until:
            self._circuit_open_until = None
            self._consecutive_failures = 0
            logger.info("UpstoxBroker circuit breaker reset — retrying")
            return False
        return True

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._CB_FAILURE_THRESHOLD:
            self._circuit_open_until = datetime.now(IST_ZONE) + timedelta(seconds=self._CB_RESET_SECONDS)
            logger.error(
                "UpstoxBroker circuit breaker OPEN after %d consecutive failures — pausing for %ds",
                self._consecutive_failures,
                self._CB_RESET_SECONDS,
            )

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open_until = None

    def _read_token_from_sources(self) -> str:
        """Read the latest token from env var or token file (whichever is populated).

        Priority: UPSTOX_TOKEN_FILE path > UPSTOX_ACCESS_TOKEN env var.
        Token file contains just the raw token string (no 'Bearer' prefix).
        """
        token_file = os.environ.get("UPSTOX_TOKEN_FILE", "").strip()
        if token_file:
            try:
                from pathlib import Path
                content = Path(token_file).read_text(encoding="utf-8").strip()
                if content:
                    return content
            except OSError:
                pass
        return os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip()

    def _refresh_token_if_available(self) -> None:
        """Proactively re-read token from env/file; rate-limited to once per minute."""
        now = datetime.now(IST_ZONE)
        if (
            self._last_token_check is not None
            and (now - self._last_token_check).total_seconds() < self._TOKEN_CHECK_INTERVAL_SECONDS
        ):
            return
        self._last_token_check = now

        token = self._read_token_from_sources()
        if not token:
            return
        new_auth = f"Bearer {token}"
        if self.headers.get("Authorization") != new_auth:
            age_hours = (now - self._token_set_at).total_seconds() / 3600
            self.headers["Authorization"] = new_auth
            self._token_set_at = now
            logger.info(
                "UpstoxBroker access token rotated (previous age=%.1fh, new=%s)",
                age_hours,
                _mask_token(token),
            )

        # Warn if the current token has been unchanged for > 22 h (likely stale)
        token_age_h = (now - self._token_set_at).total_seconds() / 3600
        if token_age_h > 22:
            logger.warning(
                "UpstoxBroker token unchanged for %.1f hours — it may have expired. "
                "Update UPSTOX_ACCESS_TOKEN or UPSTOX_TOKEN_FILE.",
                token_age_h,
            )

    def _post(self, path: str, payload: dict[str, Any]) -> BrokerOrderResponse:
        # Proactively rotate token before every call (rate-limited internally to once/min)
        self._refresh_token_if_available()

        if self._is_circuit_open():
            return BrokerOrderResponse(
                success=False,
                order_id=None,
                status="CIRCUIT_OPEN",
                message=f"Circuit breaker open until {self._circuit_open_until}",
                payload={"path": path},
            )

        url = f"{self.base_url}{path}"
        res = None
        data: dict[str, Any] | Any = {}
        for attempt in range(2):
            try:
                res = self.session.post(url, headers=self.headers, json=payload, timeout=15)
                data = res.json() if res.content else {}
                if res.status_code == 401 and attempt == 0:
                    # Force immediate re-read by resetting the throttle clock
                    self._last_token_check = None
                    self._refresh_token_if_available()
                    continue
                break
            except Exception as exc:
                self._record_failure()
                if attempt == 1:
                    return BrokerOrderResponse(
                        success=False,
                        order_id=None,
                        status="ERROR",
                        message=str(exc),
                        payload={"path": path},
                    )

        if not res.ok:
            self._record_failure()
            logger.error(
                "UpstoxBroker order rejected path=%s status=%s response=%s",
                path, res.status_code, str(data)[:200],
            )
            return BrokerOrderResponse(
                success=False,
                order_id=None,
                status=str(res.status_code),
                message=str(data),
                payload={"path": path, "response": data},
            )

        self._record_success()
        order_id = (
            (data.get("data") or {}).get("order_id")
            or (data.get("data") or {}).get("id")
            or data.get("order_id")
        )
        return BrokerOrderResponse(
            success=True,
            order_id=str(order_id) if order_id else None,
            status="ACCEPTED",
            message="ok",
            payload=data if isinstance(data, dict) else {"raw": data},
        )

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResponse:
        if "|" not in request.instrument_key:
            logger.error(
                "UpstoxBroker.place_order: instrument_key %r is not a valid Upstox token (expected 'exchange|token'). "
                "Order aborted.",
                request.instrument_key,
            )
            return BrokerOrderResponse(
                success=False,
                order_id=None,
                status="INVALID_INSTRUMENT",
                message=f"instrument_key {request.instrument_key!r} is not a valid Upstox key",
                payload={},
            )
        payload = {
            "quantity": int(request.qty),
            "product": request.product,
            "validity": "DAY",
            "price": float(request.price or 0.0),
            "tag": request.tag or "ai_options_exec",
            "instrument_token": request.instrument_key,
            "order_type": request.order_type,
            "transaction_type": request.side.upper(),
            "trigger_price": float(request.trigger_price) if request.trigger_price is not None else 0.0,
            "disclosed_quantity": 0,
            "is_amo": False,
        }
        return self._post("/v2/order/place", payload)

    def modify_order(
        self, order_id: str, *, trigger_price: float | None = None, price: float | None = None
    ) -> BrokerOrderResponse:
        payload = {
            "order_id": order_id,
            "price": float(price or 0.0),
            "trigger_price": float(trigger_price or 0.0),
        }
        return self._post("/v2/order/modify", payload)

    def cancel_order(self, order_id: str) -> BrokerOrderResponse:
        return self._post("/v2/order/cancel", {"order_id": order_id})

    def cancel_all_pending(self) -> BrokerOrderResponse:
        # Broker-side bulk cancel endpoint varies by broker account setup.
        return BrokerOrderResponse(
            success=False,
            order_id=None,
            status="UNSUPPORTED",
            message="bulk_cancel_not_supported_by_adapter",
            payload={},
        )
