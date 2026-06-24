from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

import requests
import upstox_client
from upstox_client.rest import ApiException

from backend.utils.constants import IST_ZONE
from backend.utils.config import read_runtime_upstox_access_token
from backend.utils.logger import get_logger

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
        self,
        order_id: str,
        *,
        trigger_price: float | None = None,
        price: float | None = None,
        quantity: int | None = None,
        order_type: str | None = None,
    ) -> BrokerOrderResponse:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> BrokerOrderResponse:
        raise NotImplementedError

    def cancel_all_pending(self) -> BrokerOrderResponse:
        raise NotImplementedError

    def get_order_status(self, order_id: str) -> BrokerOrderResponse:
        raise NotImplementedError

    def get_order_book(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_order_history(self, order_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def get_trade_pnl_report(
        self,
        *,
        segment: str,
        financial_year: str,
        from_date: str,
        to_date: str,
        page_number: int = 1,
        page_size: int = 100,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def get_portfolio(self) -> dict[str, Any]:
        raise NotImplementedError


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
        """Read the latest token from env var, token file, or .env.

        Priority: UPSTOX_TOKEN_FILE path > UPSTOX_ACCESS_TOKEN env var > .env.
        Token file contains just the raw token string (no 'Bearer' prefix).
        """
        return read_runtime_upstox_access_token()

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

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> BrokerOrderResponse:
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
                request_kwargs: dict[str, Any] = {"headers": self.headers, "timeout": 15}
                normalized_method = method.upper()
                if normalized_method in {"GET", "DELETE"}:
                    request_kwargs["params"] = payload or {}
                else:
                    request_kwargs["json"] = payload or {}
                if normalized_method == "GET":
                    res = self.session.get(url, **request_kwargs)
                elif normalized_method == "POST":
                    res = self.session.post(url, **request_kwargs)
                elif normalized_method == "PUT":
                    res = self.session.put(url, **request_kwargs)
                elif normalized_method == "DELETE":
                    res = self.session.delete(url, **request_kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                data = res.json() if res.content else {}
                if res.status_code == 401 and attempt == 0:
                    # Force immediate re-read by resetting the throttle clock
                    self._last_token_check = None
                    self._refresh_token_if_available()
                    continue
                break
            except Exception as exc:
                self._record_failure()
                # A mutating request may have reached the broker even when the
                # response was lost. Never resubmit it automatically.
                if method.upper() != "GET" or attempt == 1:
                    return BrokerOrderResponse(
                        success=False,
                        order_id=None,
                        status="AMBIGUOUS" if method.upper() != "GET" else "ERROR",
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
        response_data = data.get("data") if isinstance(data, dict) else None
        response_item = response_data if isinstance(response_data, dict) else {}
        order_id = response_item.get("order_id") or response_item.get("id")
        if not order_id and isinstance(data, dict):
            order_id = data.get("order_id")
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
        return self._request("POST", "/v2/order/place", payload)

    def modify_order(
        self,
        order_id: str,
        *,
        trigger_price: float | None = None,
        price: float | None = None,
        quantity: int | None = None,
        order_type: str | None = None,
    ) -> BrokerOrderResponse:
        payload = {
            "order_id": order_id,
            "price": float(price or 0.0),
            "trigger_price": float(trigger_price or 0.0),
        }
        return self._request("PUT", "/v2/order/modify", payload)

    def cancel_order(self, order_id: str) -> BrokerOrderResponse:
        return self._request("DELETE", "/v2/order/cancel", {"order_id": order_id})

    def cancel_all_pending(self) -> BrokerOrderResponse:
        # Broker-side bulk cancel endpoint varies by broker account setup.
        return BrokerOrderResponse(
            success=False,
            order_id=None,
            status="UNSUPPORTED",
            message="bulk_cancel_not_supported_by_adapter",
            payload={},
        )

    def get_order_status(self, order_id: str) -> BrokerOrderResponse:
        response = self._request("GET", "/v2/order/details", {"order_id": order_id})
        response.order_id = order_id
        return response

    def _read_endpoint(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._request("GET", path, params)
        if response.success:
            return response.payload
        return {
            "status": "error",
            "errors": [
                {
                    "status": response.status,
                    "message": response.message,
                }
            ],
            "data": [],
        }

    def get_order_book(self) -> dict[str, Any]:
        return self._read_endpoint("/v2/order/retrieve-all")

    def get_order_history(self, order_id: str) -> dict[str, Any]:
        return self._read_endpoint("/v2/order/history", {"order_id": order_id})

    def get_trade_pnl_report(
        self,
        *,
        segment: str,
        financial_year: str,
        from_date: str,
        to_date: str,
        page_number: int = 1,
        page_size: int = 100,
    ) -> dict[str, Any]:
        return self._read_endpoint(
            "/v2/trade/profit-loss/data",
            {
                "segment": segment,
                "financial_year": financial_year,
                "from_date": from_date,
                "to_date": to_date,
                "page_number": page_number,
                "page_size": page_size,
            },
        )

    def get_portfolio(self) -> dict[str, Any]:
        self._refresh_token_if_available()
        positions_url = f"{self.base_url}/v2/portfolio/short-term-positions"
        funds_url = f"{self.base_url}/v2/user/get-funds-and-margin"

        def fetch_once() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
            positions_payload: dict[str, Any] = {}
            funds_payload: dict[str, Any] = {}
            errors: list[dict[str, Any]] = []
            try:
                positions_res = self.session.get(positions_url, headers=self.headers, timeout=15)
                if positions_res.ok and positions_res.content:
                    positions_payload = positions_res.json()
                elif not positions_res.ok:
                    errors.append({
                        "source": "positions",
                        "status_code": positions_res.status_code,
                        "body": positions_res.text[:500],
                    })
            except Exception as exc:
                positions_payload = {"error": str(exc)}
                errors.append({"source": "positions", "error": str(exc)})
            try:
                funds_res = self.session.get(
                    funds_url,
                    headers=self.headers,
                    params={"segment": "SEC"},
                    timeout=15,
                )
                if funds_res.ok and funds_res.content:
                    funds_payload = funds_res.json()
                elif not funds_res.ok:
                    errors.append({
                        "source": "funds",
                        "status_code": funds_res.status_code,
                        "body": funds_res.text[:500],
                    })
            except Exception as exc:
                funds_payload = {"error": str(exc)}
                errors.append({"source": "funds", "error": str(exc)})
            return positions_payload, funds_payload, errors

        positions_payload, funds_payload, errors = fetch_once()
        if any(error.get("status_code") == 401 for error in errors):
            self._last_token_check = None
            self._refresh_token_if_available()
            positions_payload, funds_payload, errors = fetch_once()
        funds = (funds_payload.get("data") or {}) if isinstance(funds_payload, dict) else {}
        return {
            "broker": self.broker_name,
            "positions": (positions_payload.get("data") or []) if isinstance(positions_payload, dict) else [],
            "funds": funds,
            "raw_positions": positions_payload,
            "raw_funds": funds_payload,
            "errors": errors,
            "status": "ok" if funds or not errors else "warn",
        }


class UpstoxSandboxBroker(BaseBroker):
    """Order-only adapter pinned to the Upstox sandbox host."""

    broker_name = "upstox_sandbox"
    SANDBOX_ORDER_HOST = "https://api-sandbox.upstox.com"

    def __init__(self, *, access_token: str) -> None:
        configuration = upstox_client.Configuration(sandbox=True)
        # Some older SDK builds accept sandbox=True without applying the
        # sandbox hosts. Keep both endpoints hard-coded and non-configurable.
        configuration.sandbox = True
        configuration.host = self.SANDBOX_ORDER_HOST
        configuration.order_host = self.SANDBOX_ORDER_HOST
        configuration.access_token = str(access_token or "").strip()
        configured_hosts = {
            str(getattr(configuration, "host", "") or "").rstrip("/"),
            str(getattr(configuration, "order_host", "") or "").rstrip("/"),
        }
        if configured_hosts != {self.SANDBOX_ORDER_HOST} or any(
            urlsplit(host).scheme != "https"
            or urlsplit(host).hostname != "api-sandbox.upstox.com"
            or urlsplit(host).port is not None
            or urlsplit(host).path not in {"", "/"}
            or urlsplit(host).query
            or urlsplit(host).fragment
            for host in configured_hosts
        ):
            raise RuntimeError("Unsafe Upstox sandbox order host")
        self.configuration = configuration
        self.api_client = upstox_client.ApiClient(configuration)
        self.order_api = upstox_client.OrderApiV3(self.api_client)
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {configuration.access_token}",
            "Accept": "application/json",
        }

    @staticmethod
    def _payload(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if value is None:
            return {}
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            return payload if isinstance(payload, dict) else {"data": payload}
        return {"data": str(value)}

    @classmethod
    def _success(
        cls,
        value: Any,
        *,
        fallback_order_id: str | None = None,
        status: str = "ACCEPTED",
        message: str = "sandbox_order_accepted",
    ) -> BrokerOrderResponse:
        payload = cls._payload(value)
        data = payload.get("data") or {}
        order_ids = data.get("order_ids") if isinstance(data, dict) else None
        order_id = (
            data.get("order_id") if isinstance(data, dict) else None
        ) or (order_ids[0] if isinstance(order_ids, list) and order_ids else None)
        return BrokerOrderResponse(
            success=True,
            order_id=str(order_id or fallback_order_id) if (order_id or fallback_order_id) else None,
            status=status,
            message=message,
            payload=payload,
        )

    @staticmethod
    def _failure(exc: Exception, *, order_id: str | None = None) -> BrokerOrderResponse:
        status_code = getattr(exc, "status", None)
        body = getattr(exc, "body", None)
        ambiguous = not isinstance(exc, ApiException) or status_code is None
        return BrokerOrderResponse(
            success=False,
            order_id=order_id,
            status="AMBIGUOUS" if ambiguous else str(status_code),
            message=str(body or exc),
            payload={"sandbox": True, "status_code": status_code, "body": body},
        )

    @staticmethod
    def _product(product: str) -> str:
        return "I" if str(product or "").upper() in {"MIS", "I"} else str(product or "D").upper()

    def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResponse:
        if "|" not in request.instrument_key:
            return BrokerOrderResponse(
                False,
                None,
                "INVALID_INSTRUMENT",
                f"instrument_key {request.instrument_key!r} is not a valid Upstox key",
                {},
            )
        body = upstox_client.PlaceOrderV3Request(
            quantity=int(request.qty),
            product=self._product(request.product),
            validity="DAY",
            price=float(request.price or 0.0),
            tag=request.tag or "ai_sandbox_exec",
            slice=False,
            instrument_token=request.instrument_key,
            order_type=str(request.order_type or "LIMIT").upper(),
            transaction_type=str(request.side).upper(),
            disclosed_quantity=0,
            trigger_price=float(request.trigger_price or 0.0),
            is_amo=False,
        )
        try:
            return self._success(self.order_api.place_order(body))
        except Exception as exc:
            return self._failure(exc)

    def modify_order(
        self,
        order_id: str,
        *,
        trigger_price: float | None = None,
        price: float | None = None,
        quantity: int | None = None,
        order_type: str | None = None,
    ) -> BrokerOrderResponse:
        if int(quantity or 0) <= 0:
            return BrokerOrderResponse(
                False,
                order_id,
                "INVALID_QUANTITY",
                "Sandbox modify requires quantity",
                {},
            )
        body = upstox_client.ModifyOrderRequest(
            quantity=int(quantity),
            validity="DAY",
            price=float(price or 0.0),
            order_id=order_id,
            order_type=str(order_type or "SL").upper(),
            disclosed_quantity=0,
            trigger_price=float(trigger_price or 0.0),
        )
        try:
            return self._success(
                self.order_api.modify_order(body),
                fallback_order_id=order_id,
                status="MODIFIED",
                message="sandbox_order_modified",
            )
        except Exception as exc:
            return self._failure(exc, order_id=order_id)

    def cancel_order(self, order_id: str) -> BrokerOrderResponse:
        try:
            return self._success(
                self.order_api.cancel_order(order_id),
                fallback_order_id=order_id,
                status="CANCELLED",
                message="sandbox_order_cancelled",
            )
        except Exception as exc:
            return self._failure(exc, order_id=order_id)

    def cancel_all_pending(self) -> BrokerOrderResponse:
        return BrokerOrderResponse(
            False,
            None,
            "UNSUPPORTED",
            "sandbox_bulk_cancel_not_supported",
            {"sandbox": True},
        )

    def get_order_status(self, order_id: str) -> BrokerOrderResponse:
        payload = self._read_endpoint("/v2/order/details", {"order_id": order_id})
        success = payload.get("status") == "success"
        return BrokerOrderResponse(
            success,
            order_id,
            "ACCEPTED" if success else "ERROR",
            "ok" if success else str(payload.get("errors") or "sandbox_order_status_failed"),
            payload,
        )

    def _read_endpoint(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.SANDBOX_ORDER_HOST}{path}"
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                params=params or {},
                timeout=15,
            )
            payload = response.json() if response.content else {}
        except Exception as exc:
            return {
                "status": "error",
                "errors": [{"status": "ERROR", "message": str(exc)}],
                "data": [],
            }
        if response.ok and isinstance(payload, dict):
            return payload
        return {
            "status": "error",
            "errors": [
                {
                    "status": str(response.status_code),
                    "message": payload if payload else response.text[:500],
                }
            ],
            "data": [],
        }

    def get_order_book(self) -> dict[str, Any]:
        return self._read_endpoint("/v2/order/retrieve-all")

    def get_order_history(self, order_id: str) -> dict[str, Any]:
        return self._read_endpoint("/v2/order/history", {"order_id": order_id})

    def get_trade_pnl_report(
        self,
        *,
        segment: str,
        financial_year: str,
        from_date: str,
        to_date: str,
        page_number: int = 1,
        page_size: int = 100,
    ) -> dict[str, Any]:
        return {
            "status": "error",
            "errors": [
                {
                    "status": "UNSUPPORTED",
                    "message": "Upstox Sandbox does not expose the trade profit/loss report API.",
                }
            ],
            "data": [],
        }

    def get_portfolio(self) -> dict[str, Any]:
        return {
            "broker": self.broker_name,
            "positions": [],
            "funds": {},
            "errors": [],
            "status": "sandbox",
            "host": self.configuration.order_host,
        }
