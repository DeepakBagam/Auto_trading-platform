from __future__ import annotations

from copy import copy
from datetime import date, datetime, time
import hashlib
import math
import time as time_module
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.db.models import DailySummary, ExecutionOrder, ExecutionPosition, ExecutionSignalAudit
from backend.execution_engine.broker import (
    BaseBroker,
    BrokerOrderRequest,
    UpstoxBroker,
    UpstoxSandboxBroker,
)
from backend.execution_engine.live_service import (
    DIRECTIONAL_SIGNALS_ENABLED,
    build_option_selection,
    compute_sandbox_portfolio_metrics,
    build_technical_signal,
    latest_option_premium,
    load_market_context,
    log_signal_decision,
)
from backend.execution_engine.risk_manager import compute_quantity, update_risk_plan
from backend.execution_engine.slippage_tracker import estimate_slippage
from backend.execution_engine.strike_selector import compute_position_lots, lot_size_for_symbol
from backend.utils.calendar_utils import is_trading_day
from backend.utils.app_state import create_audit_log, get_runtime_trading_mode
from backend.utils.config import (
    Settings,
    get_settings,
    read_runtime_upstox_access_token,
    read_runtime_upstox_sandbox_access_token,
)
from backend.utils.constants import IST_ZONE
from backend.utils.logger import get_logger
from backend.utils.notifications import send_order_notification
from backend.utils.symbols import is_option_execution_symbol, normalize_symbol_key, symbol_value_filter

logger = get_logger(__name__)


def _now_ist() -> datetime:
    return datetime.now(IST_ZONE)


def _parse_time(value: str, fallback: time) -> time:
    try:
        hour, minute = str(value).split(":", 1)
        return time(int(hour), int(minute))
    except Exception:
        return fallback


class IntradayOptionsExecutionEngine:
    def __init__(self, settings: Settings | None = None, broker: BaseBroker | None = None) -> None:
        self.settings = settings or get_settings()
        self.broker = broker or self._build_broker()
        self._broker_credential_marker = self._active_broker_credential_marker()
        self._last_entry_candle: dict[str, str] = {}

    def _active_broker_credential_marker(self, mode: str | None = None) -> tuple[str, str]:
        normalized_mode = str(mode or self.settings.execution_mode).lower()
        token = (
            read_runtime_upstox_access_token(self.settings)
            if normalized_mode == "live"
            else read_runtime_upstox_sandbox_access_token(self.settings)
        )
        digest = hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()
        return normalized_mode, digest

    def _build_broker(self) -> BaseBroker:
        if str(self.settings.execution_mode).lower() == "live":
            return UpstoxBroker(
                base_url=self.settings.upstox_base_url,
                access_token=read_runtime_upstox_access_token(self.settings),
            )
        return UpstoxSandboxBroker(
            access_token=read_runtime_upstox_sandbox_access_token(self.settings),
        )

    def _sync_runtime_mode(self, db: Session) -> str:
        mode = get_runtime_trading_mode(db, settings=self.settings)
        expected_broker = "upstox" if mode == "live" else "upstox_sandbox"
        credential_marker = self._active_broker_credential_marker(mode)
        if (
            str(self.settings.execution_mode).lower() != mode
            or str(self.broker.broker_name).lower() != expected_broker
            or self._broker_credential_marker != credential_marker
        ):
            self.settings.execution_mode = mode
            self.broker = self._build_broker()
            self._broker_credential_marker = credential_marker
        return mode

    def _live_broker_ready(self) -> tuple[bool, str]:
        if self._is_sandbox_mode():
            if not read_runtime_upstox_sandbox_access_token(self.settings):
                return False, "UPSTOX_SANDBOX_ACCESS_TOKEN is missing"
            return True, "sandbox_ready"
        if not self._is_live_mode():
            return False, "unsupported_execution_mode"
        if not read_runtime_upstox_access_token(self.settings):
            return False, "UPSTOX_ACCESS_TOKEN is missing"
        try:
            portfolio = self.broker.get_portfolio()
        except Exception as exc:
            logger.exception("Live broker readiness check failed")
            return False, str(exc)
        errors = portfolio.get("errors") or []
        if errors:
            return False, str(errors[:2])
        if portfolio.get("status") not in {None, "ok"}:
            return False, str(portfolio.get("status"))
        return True, "ok"

    def _entry_start(self) -> time:
        return _parse_time(self.settings.entry_window_start, time(9, 20))

    def _entry_end(self) -> time:
        return _parse_time(self.settings.entry_window_end, time(13, 30))

    def _force_squareoff_time(self) -> time:
        return _parse_time(self.settings.force_squareoff_time, time(15, 15))

    def _is_entry_window(self, now: datetime) -> bool:
        current = now.timetz().replace(tzinfo=None)
        return self._entry_start() <= current <= self._entry_end()

    def _is_force_squareoff(self, now: datetime) -> bool:
        current = now.timetz().replace(tzinfo=None)
        return current >= self._force_squareoff_time()

    def _claim_signal_candle(self, db: Session, *, signal) -> bool:
        candle_ts = signal.timestamp
        claim = ExecutionSignalAudit(
            trade_date=candle_ts.date(),
            symbol=signal.symbol,
            interval=signal.interval,
            candle_ts=candle_ts,
            signal_action=signal.action,
            strategy_name=str(signal.details.get("strategy_name") or "auto"),
            executed=False,
        )
        db.add(claim)
        try:
            db.commit()
            return True
        except IntegrityError:
            db.rollback()
            return False

    def _positions_with_status(
        self,
        db: Session,
        statuses: set[str],
        *,
        symbol: str | None = None,
    ) -> list[ExecutionPosition]:
        query = select(ExecutionPosition).where(ExecutionPosition.status.in_(sorted(statuses)))
        if symbol:
            query = query.where(symbol_value_filter(ExecutionPosition.symbol, symbol))
        mode = str(self.settings.execution_mode).lower()
        return [
            position
            for position in db.execute(query.order_by(ExecutionPosition.opened_at.asc())).scalars().all()
            if self._position_execution_mode(position) == mode
        ]

    def _open_positions(self, db: Session, symbol: str | None = None) -> list[ExecutionPosition]:
        return self._positions_with_status(
            db,
            {"OPEN", "ENTRY_PENDING", "EXIT_SUBMITTING", "EXIT_PENDING", "MANUAL_REVIEW"},
            symbol=symbol,
        )

    def _managed_positions(self, db: Session) -> list[ExecutionPosition]:
        return self._positions_with_status(db, {"OPEN"})

    def _position_execution_mode(self, position: ExecutionPosition) -> str:
        return str((position.metadata_json or {}).get("execution_mode") or "archived_paper").lower()

    def _entry_positions_for_day(
        self,
        db: Session,
        *,
        trade_date: date,
        symbol: str | None = None,
    ) -> list[ExecutionPosition]:
        query = select(ExecutionPosition).where(ExecutionPosition.trade_date == trade_date)
        if symbol:
            query = query.where(symbol_value_filter(ExecutionPosition.symbol, symbol))
        mode = str(self.settings.execution_mode).lower()
        return [
            position
            for position in db.execute(query.order_by(ExecutionPosition.opened_at.asc())).scalars().all()
            if self._position_execution_mode(position) == mode
            and str(position.status).upper() not in {"ENTRY_PENDING", "ENTRY_FAILED"}
        ]

    def _mode_positions_for_day(
        self,
        db: Session,
        *,
        trade_date: date,
        statuses: set[str] | None = None,
    ) -> list[ExecutionPosition]:
        query = select(ExecutionPosition).where(ExecutionPosition.trade_date == trade_date)
        if statuses:
            query = query.where(ExecutionPosition.status.in_(sorted(statuses)))
        mode = str(self.settings.execution_mode).lower()
        return [
            position
            for position in db.execute(query.order_by(ExecutionPosition.opened_at.asc())).scalars().all()
            if self._position_execution_mode(position) == mode
        ]

    def _successful_trade_guard(
        self,
        db: Session,
        *,
        now: datetime,
        symbol: str,
    ) -> tuple[int, int]:
        positions = self._entry_positions_for_day(db, trade_date=now.date(), symbol=symbol)
        trade_count = len(positions)
        cooldown_seconds = 0
        opened_values = [position.opened_at for position in positions if position.opened_at is not None]
        if opened_values:
            latest_opened = max(opened_values)
            if latest_opened.tzinfo is None:
                latest_opened = latest_opened.replace(tzinfo=IST_ZONE)
            else:
                latest_opened = latest_opened.astimezone(IST_ZONE)
            cooldown_minutes = max(0, int(getattr(self.settings, "signal_cooldown_minutes", 12)))
            elapsed = int((now - latest_opened).total_seconds())
            cooldown_seconds = max(0, (cooldown_minutes * 60) - elapsed)
        return trade_count, cooldown_seconds

    def _available_trading_balance(self, db: Session) -> tuple[float, str, dict[str, Any]]:
        mode = str(self.settings.execution_mode).lower()
        if mode == "sandbox":
            sandbox = compute_sandbox_portfolio_metrics(db, settings=self.settings)
            balance = max(0.0, float(sandbox["available_balance"]))
            return balance, "sandbox_available_balance", {"sandbox_portfolio": sandbox}

        portfolio = self.broker.get_portfolio()
        funds = portfolio.get("funds") or {}
        containers = [funds]
        if isinstance(funds, dict):
            containers.extend(
                value
                for key, value in funds.items()
                if key in {"equity", "securities", "commodity"} and isinstance(value, dict)
            )
        for container in containers:
            if not isinstance(container, dict):
                continue
            for key in ("available_margin", "available_funds", "available_cash"):
                if key not in container or container.get(key) is None:
                    continue
                try:
                    balance = max(0.0, float(container[key]))
                except (TypeError, ValueError):
                    continue
                return balance, f"broker_{key}", {"broker": portfolio.get("broker"), "funds": funds}

        return 0.0, "broker_margin_unavailable", {
            "broker": portfolio.get("broker"),
            "funds": funds,
            "reason": "broker_available_margin_unavailable",
            "errors": portfolio.get("errors") or [],
        }

    @staticmethod
    def _sizing_metadata(sizing, *, balance: float, balance_source: str) -> dict[str, Any]:
        return {
            "available_balance": round(float(balance), 2),
            "balance_source": balance_source,
            "entry_premium": round(float(sizing.entry_premium), 2),
            "lots": int(sizing.lots),
            "quantity": int(sizing.qty),
            "capital_required": round(float(sizing.capital_allocated), 2),
            "risk_budget": round(float(sizing.risk_budget), 2),
            "risk_per_lot": round(float(sizing.risk_per_lot), 2),
            "estimated_risk": round(float(sizing.estimated_risk), 2),
            "affordable_lots": int(sizing.affordable_lots),
            "risk_limited_lots": int(sizing.risk_limited_lots),
            "vix_multiplier": round(float(sizing.vix_multiplier), 4),
            "sizing_reason": str(sizing.reason),
        }

    @staticmethod
    def _extract_fill_price(payload: Any) -> float | None:
        if not isinstance(payload, dict):
            return None
        for key in ("average_price", "average_fill_price", "filled_price", "fill_price"):
            value = payload.get(key)
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if price > 0.0:
                return price
        for key in ("data", "order", "trade"):
            nested = IntradayOptionsExecutionEngine._extract_fill_price(payload.get(key))
            if nested is not None:
                return nested
        return None

    @staticmethod
    def _extract_order_status(payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        for key in ("order_status", "status"):
            value = payload.get(key)
            if value is not None and not isinstance(value, (dict, list)):
                normalized = str(value).strip().upper()
                if normalized and normalized not in {"SUCCESS", "OK"}:
                    return normalized
        for key in ("data", "order", "trade"):
            normalized = IntradayOptionsExecutionEngine._extract_order_status(payload.get(key))
            if normalized:
                return normalized
        return ""

    @staticmethod
    def _extract_int(payload: Any, *keys: str) -> int | None:
        if not isinstance(payload, dict):
            return None
        for key in keys:
            value = payload.get(key)
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                return parsed
        for key in ("data", "order", "trade"):
            parsed = IntradayOptionsExecutionEngine._extract_int(payload.get(key), *keys)
            if parsed is not None:
                return parsed
        return None

    @classmethod
    def _extract_filled_quantity(cls, payload: Any) -> int:
        return int(cls._extract_int(payload, "filled_quantity", "filled_qty", "traded_quantity") or 0)

    @classmethod
    def _extract_pending_quantity(cls, payload: Any) -> int | None:
        return cls._extract_int(payload, "pending_quantity", "pending_qty")

    def _entry_order_row(self, db: Session, position: ExecutionPosition) -> ExecutionOrder | None:
        return db.scalar(
            select(ExecutionOrder)
            .where(
                ExecutionOrder.position_id == position.id,
                ExecutionOrder.order_kind == "ENTRY",
            )
            .order_by(ExecutionOrder.created_at.desc())
            .limit(1)
        )

    def _finalize_live_entry(
        self,
        db: Session,
        *,
        position: ExecutionPosition,
        now: datetime,
        filled_quantity: int,
        fill_price: float,
        broker_status: str,
    ) -> ExecutionOrder | None:
        metadata = dict(position.metadata_json or {})
        filled_quantity = max(0, int(filled_quantity))
        position.quantity = filled_quantity
        position.entry_price = float(fill_price)
        position.entry_premium = float(fill_price)
        position.current_price = float(fill_price)
        position.current_premium = float(fill_price)
        position.peak_premium = float(fill_price)
        position.pnl_points = 0.0
        position.pnl_value = 0.0
        position.realized_pnl = 0.0
        position.unrealized_pnl = 0.0
        position.status = "OPEN"
        position.opened_at = now
        metadata["entry_filled_quantity"] = filled_quantity
        metadata["entry_fill_price"] = float(fill_price)
        metadata["entry_reconciled_at"] = now.isoformat()
        metadata["entry_order_status"] = broker_status
        metadata["capital_invested"] = round(float(fill_price) * filled_quantity, 2)
        metadata["premium_history"] = [
            {
                "timestamp": now.isoformat(),
                "premium": round(float(fill_price), 2),
                "current_sl": round(float(position.current_sl or position.stop_loss or 0.0), 2),
                "tsl_active": False,
                "unrealized_pnl": 0.0,
            }
        ]
        metadata.pop("entry_reconciliation_last_error", None)
        position.metadata_json = metadata
        return self._place_live_protective_stop(db, position=position)

    def _reconcile_pending_entries(self, db: Session, now: datetime) -> dict[str, int]:
        pending = self._positions_with_status(db, {"ENTRY_PENDING"})
        opened = 0
        failed = 0
        still_pending = 0
        terminal_statuses = {"COMPLETE", "COMPLETED", "FILLED", "CANCELLED", "CANCELED", "REJECTED", "FAILED"}
        filled_statuses = {"COMPLETE", "COMPLETED", "FILLED"}
        for position in pending:
            metadata = dict(position.metadata_json or {})
            order_id = str(position.entry_order_id or metadata.get("entry_order_id") or "").strip()
            if not order_id:
                metadata["entry_reconciliation_last_error"] = "missing_entry_order_id"
                position.metadata_json = metadata
                still_pending += 1
                continue
            response = self.broker.get_order_status(order_id)
            payload = getattr(response, "payload", {}) or {}
            broker_status = self._extract_order_status(payload) or str(response.status or "").upper()
            requested_quantity = int(metadata.get("entry_requested_quantity") or position.quantity or 0)
            filled_quantity = self._extract_filled_quantity(payload)
            pending_quantity = self._extract_pending_quantity(payload)
            fill_price = self._extract_fill_price(payload)
            metadata["entry_reconciliation_checked_at"] = now.isoformat()
            metadata["entry_order_status"] = broker_status
            metadata["entry_filled_quantity"] = filled_quantity
            metadata["entry_pending_quantity"] = pending_quantity
            order_row = self._entry_order_row(db, position)
            if order_row is not None:
                order_row.status = broker_status or str(response.status)
                order_row.response_json = payload

            is_terminal = broker_status in terminal_statuses
            if filled_quantity > 0 and fill_price is not None and is_terminal:
                self._finalize_live_entry(
                    db,
                    position=position,
                    now=now,
                    filled_quantity=min(filled_quantity, requested_quantity),
                    fill_price=fill_price,
                    broker_status=broker_status,
                )
                if order_row is not None:
                    order_row.quantity = min(filled_quantity, requested_quantity)
                    order_row.price = fill_price
                    order_row.entry_premium = fill_price
                if not bool((position.metadata_json or {}).get("broker_sl_active")):
                    self._close_position(
                        db,
                        position=position,
                        now=now,
                        reason="PROTECTIVE_SL_FAILED",
                        exit_premium=fill_price,
                    )
                opened += 1
                continue

            if filled_quantity > 0 and not is_terminal:
                if not metadata.get("entry_cancel_requested_at"):
                    cancel_response = self.broker.cancel_order(order_id)
                    metadata["entry_cancel_requested_at"] = now.isoformat()
                    metadata["entry_cancel_status"] = cancel_response.status
                    metadata["entry_cancel_message"] = cancel_response.message
                position.metadata_json = metadata
                still_pending += 1
                continue

            if broker_status in {"CANCELLED", "CANCELED", "REJECTED", "FAILED"}:
                position.status = "ENTRY_FAILED"
                position.quantity = 0
                metadata["entry_reconciliation_needed"] = False
                metadata["entry_failure_status"] = broker_status
                position.metadata_json = metadata
                failed += 1
                continue

            if broker_status in filled_statuses and (fill_price is None or filled_quantity <= 0):
                metadata["entry_reconciliation_last_error"] = "confirmed_entry_fill_details_unavailable"
            position.metadata_json = metadata
            still_pending += 1
        if pending:
            db.commit()
        return {
            "reconciled_entries": opened,
            "pending_entry_reconciliations": still_pending,
            "failed_entries": failed,
        }

    def _apply_exit_fill_progress(
        self,
        *,
        position: ExecutionPosition,
        now: datetime,
        broker_status: str,
        payload: dict[str, Any],
        order_row: ExecutionOrder | None,
    ) -> tuple[int, int]:
        metadata = dict(position.metadata_json or {})
        requested_quantity = int(metadata.get("exit_requested_quantity") or position.quantity or 0)
        filled_quantity = self._extract_filled_quantity(payload)
        if broker_status in {"COMPLETE", "COMPLETED", "FILLED"} and filled_quantity <= 0:
            filled_quantity = requested_quantity
        filled_quantity = min(max(0, filled_quantity), requested_quantity)
        fill_price = self._extract_fill_price(payload)
        accounted_quantity = int(metadata.get("exit_accounted_quantity") or 0)
        base_realized = float(metadata.get("exit_base_realized_pnl") or 0.0)
        if filled_quantity > accounted_quantity and fill_price is not None:
            cumulative_realized = round(
                base_realized
                + (float(fill_price) - float(position.entry_premium or position.entry_price)) * filled_quantity,
                2,
            )
            position.realized_pnl = cumulative_realized
            position.pnl_value = cumulative_realized
            metadata["exit_accounted_quantity"] = filled_quantity
            metadata["exit_average_fill_price"] = fill_price
            metadata["exit_realized_pnl"] = cumulative_realized
        remaining_quantity = max(0, requested_quantity - filled_quantity)
        if filled_quantity > 0 and remaining_quantity > 0:
            position.quantity = remaining_quantity
            position.unrealized_pnl = round(
                (float(position.current_premium or position.entry_premium or position.entry_price)
                 - float(position.entry_premium or position.entry_price))
                * remaining_quantity,
                2,
            )
        metadata["exit_filled_quantity"] = filled_quantity
        metadata["exit_remaining_quantity"] = remaining_quantity
        metadata["exit_order_status"] = broker_status
        position.metadata_json = metadata
        if order_row is not None:
            order_row.status = broker_status
            order_row.response_json = payload
            if fill_price is not None:
                order_row.price = fill_price
                order_row.exit_premium = fill_price
                order_row.realized_pnl = position.realized_pnl
        return filled_quantity, remaining_quantity

    def _complete_exit(
        self,
        db: Session,
        *,
        position: ExecutionPosition,
        now: datetime,
        fill_price: float,
    ) -> None:
        metadata = dict(position.metadata_json or {})
        original_entry_quantity = int(metadata.get("entry_filled_quantity") or position.quantity or 0)
        position.status = "CLOSED"
        position.quantity = original_entry_quantity
        position.closed_at = now
        position.current_price = fill_price
        position.current_premium = fill_price
        position.exit_premium = fill_price
        position.pnl_points = round(fill_price - float(position.entry_premium or position.entry_price), 2)
        position.unrealized_pnl = 0.0
        position.exit_reason = str(metadata.get("pending_exit_reason") or position.exit_reason or "EXIT")
        metadata["reconciliation_needed"] = False
        metadata["reconciled_at"] = now.isoformat()
        metadata["reconciled_fill_price"] = fill_price
        metadata["broker_sl_active"] = False
        metadata.pop("reconciliation_last_error", None)
        position.metadata_json = metadata
        self._refresh_daily_summary(db, position.trade_date)

    def _reopen_exit_residual(
        self,
        db: Session,
        *,
        position: ExecutionPosition,
        now: datetime,
        remaining_quantity: int,
        broker_status: str,
    ) -> None:
        metadata = dict(position.metadata_json or {})
        position.quantity = int(remaining_quantity)
        position.status = "OPEN"
        position.unrealized_pnl = round(
            (float(position.current_premium or position.entry_premium or position.entry_price)
             - float(position.entry_premium or position.entry_price))
            * int(remaining_quantity),
            2,
        )
        metadata["reconciliation_needed"] = False
        metadata["exit_order_failed"] = True
        metadata["exit_order_error"] = f"broker_status:{broker_status.lower()}"
        metadata["exit_reopened_at"] = now.isoformat()
        metadata["broker_sl_active"] = False
        position.metadata_json = metadata
        self._place_live_protective_stop(db, position=position)

    def _reconcile_pending_exits(self, db: Session, now: datetime) -> dict[str, int]:
        pending = self._positions_with_status(db, {"EXIT_PENDING", "EXIT_SUBMITTING"})
        reconciled = 0
        still_pending = 0
        reopened = 0
        for position in pending:
            metadata = dict(position.metadata_json or {})
            order_id = str(metadata.get("exit_order_id") or "").strip()
            if not order_id:
                metadata["reconciliation_last_error"] = "missing_exit_order_id"
                position.metadata_json = metadata
                still_pending += 1
                continue
            response = self.broker.get_order_status(order_id)
            payload = getattr(response, "payload", {}) or {}
            broker_status = self._extract_order_status(payload) or str(response.status or "").upper()
            metadata["reconciliation_checked_at"] = now.isoformat()
            metadata["reconciliation_broker_status"] = broker_status
            position.metadata_json = metadata
            order_row = db.scalar(
                select(ExecutionOrder)
                .where(
                    ExecutionOrder.position_id == position.id,
                    ExecutionOrder.broker_order_id == order_id,
                )
                .order_by(ExecutionOrder.created_at.desc())
                .limit(1)
            )

            if metadata.get("exit_reconciliation_source") == "protective_stop_cancel":
                if broker_status in {"CANCELLED", "CANCELED", "REJECTED", "FAILED"}:
                    self._submit_market_exit(
                        db,
                        position=position,
                        now=now,
                        reason=str(metadata.get("pending_exit_reason") or "EXIT"),
                        exit_premium=metadata.get("exit_reference_quote"),
                    )
                    if str(position.status).upper() == "CLOSED":
                        reconciled += 1
                    elif str(position.status).upper() == "OPEN":
                        reopened += 1
                    else:
                        still_pending += 1
                    continue
                if broker_status not in {"COMPLETE", "COMPLETED", "FILLED"}:
                    still_pending += 1
                    continue

            filled_quantity, remaining_quantity = self._apply_exit_fill_progress(
                position=position,
                now=now,
                broker_status=broker_status,
                payload=payload,
                order_row=order_row,
            )
            fill_price = self._extract_fill_price(payload)
            terminal = broker_status in {"COMPLETE", "COMPLETED", "FILLED", "REJECTED", "CANCELLED", "CANCELED", "FAILED"}
            if remaining_quantity == 0 and filled_quantity > 0 and fill_price is not None:
                self._complete_exit(db, position=position, now=now, fill_price=fill_price)
                reconciled += 1
                continue
            if terminal:
                if remaining_quantity > 0:
                    self._reopen_exit_residual(
                        db,
                        position=position,
                        now=now,
                        remaining_quantity=remaining_quantity,
                        broker_status=broker_status,
                    )
                    reopened += 1
                else:
                    metadata = dict(position.metadata_json or {})
                    metadata["reconciliation_last_error"] = "broker_fill_price_unavailable"
                    position.metadata_json = metadata
                    still_pending += 1
                continue
            still_pending += 1
        if pending:
            db.commit()
        return {
            "reconciled_exits": reconciled,
            "pending_exit_reconciliations": still_pending,
            "reopened_exits": reopened,
        }

    def _append_position_history(self, position: ExecutionPosition, *, now: datetime, premium: float) -> None:
        metadata = dict(position.metadata_json or {})
        history = list(metadata.get("premium_history") or [])
        history.append(
            {
                "timestamp": now.isoformat(),
                "premium": round(float(premium), 2),
                "current_sl": round(float(position.current_sl or position.stop_loss or 0.0), 2),
                "tsl_active": bool(position.tsl_active),
                "unrealized_pnl": round(float(position.unrealized_pnl or 0.0), 2),
            }
        )
        metadata["premium_history"] = history[-500:]
        position.metadata_json = metadata

    def _log_order(
        self,
        db: Session,
        *,
        position_id: int | None,
        trade_date: date,
        symbol: str,
        order_kind: str,
        side: str,
        quantity: int,
        response,
        strike_price: float | None = None,
        option_type: str | None = None,
        expiry_date: date | None = None,
        entry_premium: float | None = None,
        initial_sl: float | None = None,
        current_sl: float | None = None,
        target_premium: float | None = None,
        peak_premium: float | None = None,
        tsl_active: bool = False,
        exit_premium: float | None = None,
        exit_reason: str | None = None,
        realized_pnl: float | None = None,
        unrealized_pnl: float | None = None,
        consensus_reason: str | None = None,
    ) -> ExecutionOrder:
        row = ExecutionOrder(
            position_id=position_id,
            trade_date=trade_date,
            symbol=symbol,
            strike_price=strike_price,
            option_type=option_type,
            expiry_date=expiry_date,
            order_kind=order_kind,
            side=side,
            quantity=int(quantity),
            price=entry_premium if order_kind == "ENTRY" else exit_premium,
            trigger_price=current_sl,
            entry_premium=entry_premium,
            initial_sl=initial_sl,
            current_sl=current_sl,
            target_premium=target_premium,
            peak_premium=peak_premium,
            tsl_active=bool(tsl_active),
            exit_premium=exit_premium,
            exit_reason=exit_reason,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            ml_confidence=None,
            ai_score=None,
            pine_signal=None,
            consensus_reason=consensus_reason,
            status=str(getattr(response, "status", "NEW")),
            broker_name=str(self.broker.broker_name),
            broker_order_id=getattr(response, "order_id", None),
            response_json=getattr(response, "payload", {}) or {},
            created_at=_now_ist(),
        )
        db.add(row)
        db.flush()
        return row

    def _notify_order(self, order: ExecutionOrder, position: ExecutionPosition | None = None) -> None:
        metadata = position.metadata_json if position is not None else {}
        payload = {
            "order_id": order.id,
            "trade_date": order.trade_date,
            "symbol": order.symbol,
            "order_kind": order.order_kind,
            "side": order.side,
            "quantity": order.quantity,
            "strike_price": order.strike_price,
            "option_type": order.option_type,
            "expiry_date": order.expiry_date,
            "price": order.price,
            "trigger_price": order.trigger_price,
            "entry_premium": order.entry_premium,
            "initial_sl": order.initial_sl,
            "current_sl": order.current_sl,
            "target_premium": order.target_premium,
            "tsl_active": order.tsl_active,
            "exit_premium": order.exit_premium,
            "exit_reason": order.exit_reason,
            "realized_pnl": order.realized_pnl,
            "unrealized_pnl": order.unrealized_pnl,
            "consensus_reason": order.consensus_reason,
            "status": order.status,
            "broker_name": order.broker_name,
            "broker_order_id": order.broker_order_id,
            "created_at": order.created_at,
            "position_status": getattr(position, "status", None),
            "position_opened_at": getattr(position, "opened_at", None),
            "position_closed_at": getattr(position, "closed_at", None),
            "capital_invested": (metadata or {}).get("capital_invested"),
            "balance_before_trade": (
                (metadata or {}).get("sandbox_balance_before_trade")
                if self._is_sandbox_mode()
                else (metadata or {}).get("paper_balance_before_trade")
            ),
            "balance_after_trade": (
                (metadata or {}).get("sandbox_balance_after_trade")
                if self._is_sandbox_mode()
                else (metadata or {}).get("paper_balance_after_trade")
            ),
            "latest_quote_source": (metadata or {}).get("latest_quote_source"),
            "latest_quote_ts": (metadata or {}).get("latest_quote_ts"),
        }
        send_order_notification(payload, settings=self.settings)

    def _refresh_daily_summary(self, db: Session, trade_date: date) -> None:
        positions = self._mode_positions_for_day(db, trade_date=trade_date, statuses={"CLOSED"})
        winning = [row for row in positions if float(row.realized_pnl or row.pnl_value or 0.0) > 0]
        losing = [row for row in positions if float(row.realized_pnl or row.pnl_value or 0.0) <= 0]
        pnls = [float(row.realized_pnl or row.pnl_value or 0.0) for row in positions]
        row = db.get(DailySummary, trade_date)
        if row is None:
            row = DailySummary(date=trade_date)
            db.add(row)
        row.total_trades = len(positions)
        row.winning_trades = len(winning)
        row.losing_trades = len(losing)
        row.total_pnl = round(sum(pnls), 2)
        row.max_profit_trade = round(max(pnls), 2) if pnls else 0.0
        row.max_loss_trade = round(min(pnls), 2) if pnls else 0.0
        row.win_rate = round((len(winning) / len(positions) * 100.0) if positions else 0.0, 2)
        row.is_green = bool(row.total_pnl > 0)

    def _place_order_with_retry(
        self,
        db: Session,
        *,
        request: BrokerOrderRequest,
        action: str,
        resource_id: str,
    ):
        attempts = (
            1
            if self._is_live_mode() or self._is_sandbox_mode()
            else max(1, int(getattr(self.settings, "order_retry_attempts", 2)))
        )
        backoff_ms = max(0, int(getattr(self.settings, "order_retry_backoff_ms", 300)))
        retryable_statuses = {"ERROR", "CIRCUIT_OPEN", "408", "429", "500", "502", "503", "504"}
        last_response = None
        for attempt in range(1, attempts + 1):
            response = self.broker.place_order(request)
            last_response = response
            create_audit_log(
                db,
                action=action,
                resource="order",
                resource_id=resource_id,
                status="SUCCESS" if response.success else "ERROR",
                message=response.message,
                details={
                    "attempt": attempt,
                    "status": response.status,
                    "broker": self.broker.broker_name,
                    "instrument_key": request.instrument_key,
                    "side": request.side,
                    "qty": request.qty,
                    "tag": request.tag,
                },
            )
            if response.success:
                return response
            if str(response.status) not in retryable_statuses or attempt == attempts:
                return response
            time_module.sleep(backoff_ms / 1000.0)
        return last_response

    def _is_live_mode(self) -> bool:
        return str(self.settings.execution_mode).lower() == "live"

    def _is_sandbox_mode(self) -> bool:
        return str(self.settings.execution_mode).lower() == "sandbox"

    def _uses_broker_protection(self) -> bool:
        return self._is_live_mode() or self._is_sandbox_mode()

    def _sandbox_protected_price(
        self,
        reference: float,
        side: str,
        *,
        tick_size: float | None = None,
    ) -> float:
        tick = max(
            0.01,
            float(tick_size or getattr(self.settings, "sandbox_price_tick", 0.05)),
        )
        protection = max(0.0, float(getattr(self.settings, "sandbox_limit_protection_pct", 0.01)))
        raw = float(reference) * (1.0 + protection if str(side).upper() == "BUY" else 1.0 - protection)
        ticks = (
            math.ceil(raw / tick)
            if str(side).upper() == "BUY"
            else math.floor(raw / tick)
        )
        return round(max(tick, ticks * tick), 2)

    @staticmethod
    def _round_to_tick(reference: float, tick_size: float) -> float:
        tick = max(0.01, float(tick_size or 0.05))
        return round(max(tick, round(float(reference) / tick) * tick), 2)

    def _place_live_protective_stop(
        self,
        db: Session,
        *,
        position: ExecutionPosition,
    ) -> ExecutionOrder | None:
        if not self._uses_broker_protection():
            return None
        instrument_key = str((position.metadata_json or {}).get("instrument_key") or "")
        if not instrument_key:
            logger.error("Cannot place live protective SL for position=%s without instrument_key", position.id)
            return None
        sandbox = self._is_sandbox_mode()
        tick_size = float((position.metadata_json or {}).get("contract_tick_size") or 0.05)
        trigger_price = (
            self._round_to_tick(float(position.current_sl or position.stop_loss or 0.0), tick_size)
            if sandbox
            else round(float(position.current_sl or position.stop_loss or 0.0), 2)
        )
        request = BrokerOrderRequest(
            instrument_key=instrument_key,
            option_type=str(position.option_type),
            strike=float(position.strike),
            expiry_date=position.expiry_date.isoformat(),
            side="SELL",
            qty=int(position.quantity),
            order_type="SL" if sandbox else "SL-M",
            price=(
                self._sandbox_protected_price(trigger_price, "SELL", tick_size=tick_size)
                if sandbox
                else None
            ),
            trigger_price=trigger_price,
            tag=f"protective_sl_{position.id}",
        )
        response = self._place_order_with_retry(
            db,
            request=request,
            action="protective_sl_place",
            resource_id=str(position.id),
        )
        order_row = self._log_order(
            db,
            position_id=position.id,
            trade_date=position.trade_date,
            symbol=position.symbol,
            order_kind="SL",
            side="SELL",
            quantity=position.quantity,
            response=response,
            strike_price=position.strike,
            option_type=position.option_type,
            expiry_date=position.expiry_date,
            entry_premium=position.entry_premium,
            initial_sl=position.initial_sl,
            current_sl=trigger_price,
            target_premium=position.target_premium,
            peak_premium=position.peak_premium,
            tsl_active=bool(position.tsl_active),
            unrealized_pnl=position.unrealized_pnl,
            consensus_reason="Broker-side protective stop loss",
        )
        metadata = dict(position.metadata_json or {})
        metadata["broker_sl_order_id"] = getattr(response, "order_id", None)
        metadata["broker_sl_order_status"] = getattr(response, "status", None)
        metadata["broker_sl_trigger_price"] = trigger_price
        metadata["broker_sl_updated_at"] = _now_ist().isoformat()
        metadata["broker_sl_active"] = bool(response.success and getattr(response, "order_id", None))
        if self._is_sandbox_mode() and str(response.status).upper() == "AMBIGUOUS":
            metadata["manual_review_required"] = True
            metadata["manual_review_operation"] = "protective_stop_place"
            metadata["manual_review_message"] = response.message
            position.status = "MANUAL_REVIEW"
        position.metadata_json = metadata
        if not response.success and str(response.status).upper() != "AMBIGUOUS":
            create_audit_log(
                db,
                action="protective_sl_place_failed",
                resource="position",
                resource_id=str(position.id),
                status="ERROR",
                message=response.message,
                details={"status": response.status, "trigger_price": trigger_price},
            )
        return order_row

    def _modify_live_protective_stop(
        self,
        db: Session,
        *,
        position: ExecutionPosition,
        trigger_price: float,
    ) -> None:
        if not self._uses_broker_protection():
            return
        metadata = dict(position.metadata_json or {})
        order_id = str(metadata.get("broker_sl_order_id") or "")
        if not order_id:
            self._place_live_protective_stop(db, position=position)
            return
        previous_trigger = float(metadata.get("broker_sl_trigger_price") or 0.0)
        tick_size = float(metadata.get("contract_tick_size") or 0.05)
        trigger_price = (
            self._round_to_tick(float(trigger_price), tick_size)
            if self._is_sandbox_mode()
            else round(float(trigger_price), 2)
        )
        if trigger_price <= previous_trigger:
            return
        if self._is_sandbox_mode():
            response = self.broker.modify_order(
                order_id,
                trigger_price=trigger_price,
                price=self._sandbox_protected_price(
                    trigger_price,
                    "SELL",
                    tick_size=tick_size,
                ),
                quantity=int(position.quantity),
                order_type="SL",
            )
        else:
            response = self.broker.modify_order(order_id, trigger_price=trigger_price)
        create_audit_log(
            db,
            action="protective_sl_modify",
            resource="order",
            resource_id=order_id,
            status="SUCCESS" if response.success else "ERROR",
            message=response.message,
            details={
                "position_id": position.id,
                "old_trigger_price": previous_trigger,
                "new_trigger_price": trigger_price,
                "status": response.status,
            },
        )
        self._log_order(
            db,
            position_id=position.id,
            trade_date=position.trade_date,
            symbol=position.symbol,
            order_kind="MODIFY",
            side="SELL",
            quantity=position.quantity,
            response=response,
            strike_price=position.strike,
            option_type=position.option_type,
            expiry_date=position.expiry_date,
            entry_premium=position.entry_premium,
            initial_sl=position.initial_sl,
            current_sl=trigger_price,
            target_premium=position.target_premium,
            peak_premium=position.peak_premium,
            tsl_active=bool(position.tsl_active),
            unrealized_pnl=position.unrealized_pnl,
            consensus_reason="Broker-side protective stop modified",
        )
        metadata["broker_sl_order_status"] = response.status
        metadata["broker_sl_updated_at"] = _now_ist().isoformat()
        if response.success:
            metadata["broker_sl_trigger_price"] = trigger_price
            metadata["broker_sl_active"] = True
        elif self._is_sandbox_mode() and str(response.status).upper() == "AMBIGUOUS":
            metadata["manual_review_required"] = True
            metadata["manual_review_operation"] = "protective_stop_modify"
            metadata["manual_review_message"] = response.message
            position.status = "MANUAL_REVIEW"
        position.metadata_json = metadata

    def _cancel_live_protective_stop(
        self,
        db: Session,
        *,
        position: ExecutionPosition,
        reason: str,
        now: datetime,
        exit_premium: float | None,
    ) -> bool:
        if not self._uses_broker_protection():
            return True
        metadata = dict(position.metadata_json or {})
        order_id = str(metadata.get("broker_sl_order_id") or "")
        if not order_id or not bool(metadata.get("broker_sl_active")):
            return True
        response = self.broker.cancel_order(order_id)
        create_audit_log(
            db,
            action="protective_sl_cancel",
            resource="order",
            resource_id=order_id,
            status="SUCCESS" if response.success else "ERROR",
            message=response.message,
            details={"position_id": position.id, "reason": reason, "status": response.status},
        )
        if self._is_sandbox_mode():
            if response.success:
                metadata["broker_sl_active"] = False
                metadata["broker_sl_order_status"] = response.status
                metadata["broker_sl_cancelled_at"] = now.isoformat()
                position.metadata_json = metadata
                return True
            if str(response.status).upper() == "AMBIGUOUS":
                metadata["manual_review_required"] = True
                metadata["manual_review_operation"] = "protective_stop_cancel"
                metadata["manual_review_message"] = response.message
                position.status = "MANUAL_REVIEW"
                position.metadata_json = metadata
            return False

        status_response = self.broker.get_order_status(order_id)
        status_payload = getattr(status_response, "payload", {}) or {}
        broker_status = self._extract_order_status(status_payload) or str(status_response.status or "").upper()
        metadata["broker_sl_cancel_status"] = response.status
        metadata["broker_sl_cancel_checked_status"] = broker_status
        metadata["broker_sl_cancel_checked_at"] = now.isoformat()
        if broker_status in {"CANCELLED", "CANCELED", "REJECTED", "FAILED"}:
            metadata["broker_sl_active"] = False
            metadata["broker_sl_order_status"] = broker_status
            metadata["broker_sl_cancelled_at"] = now.isoformat()
            position.metadata_json = metadata
            return True

        metadata["reconciliation_needed"] = True
        metadata["exit_reconciliation_source"] = "protective_stop_cancel"
        metadata["exit_order_id"] = order_id
        metadata["exit_reference_quote"] = round(float(exit_premium), 2) if exit_premium is not None else None
        metadata["pending_exit_reason"] = reason
        metadata["exit_order_status"] = broker_status
        position.status = "EXIT_PENDING"
        position.exit_reason = reason
        position.metadata_json = metadata
        return False

    def _claim_exit(self, db: Session, position: ExecutionPosition, now: datetime, reason: str) -> bool:
        if self._position_execution_mode(position) != str(self.settings.execution_mode).lower():
            return False
        claimed = db.execute(
            update(ExecutionPosition)
            .where(
                ExecutionPosition.id == position.id,
                ExecutionPosition.status == "OPEN",
            )
            .values(status="EXIT_SUBMITTING", exit_reason=reason)
        )
        if int(claimed.rowcount or 0) != 1:
            db.rollback()
            return False
        db.commit()
        db.refresh(position)
        metadata = dict(position.metadata_json or {})
        metadata["exit_claimed_at"] = now.isoformat()
        metadata["pending_exit_reason"] = reason
        position.metadata_json = metadata
        db.commit()
        return True

    def _submit_market_exit(
        self,
        db: Session,
        *,
        position: ExecutionPosition,
        now: datetime,
        reason: str,
        exit_premium: float | None,
    ) -> ExecutionOrder:
        metadata = dict(position.metadata_json or {})
        instrument_key = str(metadata.get("instrument_key") or "")
        requested_quantity = int(position.quantity)
        metadata["exit_requested_quantity"] = requested_quantity
        metadata.setdefault("entry_filled_quantity", requested_quantity)
        metadata["exit_accounted_quantity"] = 0
        metadata["exit_base_realized_pnl"] = float(position.realized_pnl or 0.0)
        metadata["exit_reference_quote"] = round(float(exit_premium), 2) if exit_premium is not None else None
        metadata["exit_order_requested_at"] = now.isoformat()
        metadata["pending_exit_reason"] = reason
        metadata["exit_reconciliation_source"] = "market_exit"
        position.metadata_json = metadata
        request = BrokerOrderRequest(
            instrument_key=instrument_key,
            option_type=str(position.option_type),
            strike=float(position.strike),
            expiry_date=position.expiry_date.isoformat(),
            side="SELL",
            qty=requested_quantity,
            order_type="MARKET",
            tag=f"exit_{position.id}_{reason.lower()}",
        )
        response = self._place_order_with_retry(
            db,
            request=request,
            action="position_exit",
            resource_id=str(position.id),
        )
        metadata = dict(position.metadata_json or {})
        metadata["exit_order_status"] = str(response.status)
        metadata["exit_order_id"] = response.order_id
        position.metadata_json = metadata
        order_row = self._log_order(
            db,
            position_id=position.id,
            trade_date=position.trade_date,
            symbol=position.symbol,
            order_kind="EXIT",
            side="SELL",
            quantity=requested_quantity,
            response=response,
            strike_price=position.strike,
            option_type=position.option_type,
            expiry_date=position.expiry_date,
            entry_premium=position.entry_premium,
            initial_sl=position.initial_sl,
            current_sl=position.current_sl,
            target_premium=position.target_premium,
            peak_premium=position.peak_premium,
            tsl_active=bool(position.tsl_active),
            exit_premium=None,
            exit_reason=reason,
            realized_pnl=position.realized_pnl,
            unrealized_pnl=position.unrealized_pnl,
            consensus_reason=position.consensus_reason,
        )
        if not response.success:
            metadata["exit_order_failed"] = True
            metadata["exit_order_error"] = response.message
            if str(response.status).upper() == "AMBIGUOUS":
                position.status = "EXIT_PENDING"
                metadata["reconciliation_needed"] = True
                metadata["reconciliation_last_error"] = "ambiguous_exit_submission"
            else:
                position.status = "OPEN"
                metadata["reconciliation_needed"] = False
                metadata["broker_sl_active"] = False
                position.metadata_json = metadata
                self._place_live_protective_stop(db, position=position)
                metadata = dict(position.metadata_json or {})
            position.metadata_json = metadata
            return order_row

        position.status = "EXIT_PENDING"
        position.exit_reason = reason
        metadata["reconciliation_needed"] = True
        position.metadata_json = metadata
        payload = getattr(response, "payload", {}) or {}
        broker_status = self._extract_order_status(payload) or str(response.status or "").upper()
        filled_quantity, remaining_quantity = self._apply_exit_fill_progress(
            position=position,
            now=now,
            broker_status=broker_status,
            payload=payload,
            order_row=order_row,
        )
        fill_price = self._extract_fill_price(payload)
        if remaining_quantity == 0 and filled_quantity > 0 and fill_price is not None:
            self._complete_exit(db, position=position, now=now, fill_price=fill_price)
        return order_row

    def _max_daily_loss_amount(self) -> float:
        capital = float(self.settings.execution_capital)
        return capital * float(getattr(self.settings, "execution_max_daily_loss_pct", 0.05))

    def _daily_total_pnl(self, db: Session, trade_date: date) -> float:
        realized = self._daily_realized_pnl(db, trade_date)
        open_positions = self._positions_with_status(db, {"OPEN", "EXIT_PENDING", "EXIT_SUBMITTING"})
        unrealized = sum(float(p.unrealized_pnl or 0.0) for p in open_positions)
        return realized + unrealized

    def _close_position(
        self,
        db: Session,
        *,
        position: ExecutionPosition,
        now: datetime,
        reason: str,
        exit_premium: float | None,
    ) -> ExecutionOrder | None:
        is_live = self._is_live_mode()
        metadata = dict(position.metadata_json or {})
        if not is_live and exit_premium is None:
            metadata["exit_deferred"] = True
            metadata["exit_deferred_reason"] = "real_option_quote_unavailable"
            metadata["exit_deferred_at"] = now.isoformat()
            metadata["pending_exit_reason"] = reason
            position.metadata_json = metadata
            create_audit_log(
                db,
                action="position_exit_deferred",
                resource="position",
                resource_id=str(position.id),
                status="WARN",
                message="Sandbox exit deferred because no real option quote is available",
                details={"symbol": position.symbol, "reason": reason},
            )
            return None

        if is_live:
            if str(position.status).upper() == "OPEN" and not self._claim_exit(db, position, now, reason):
                return None
            if str(position.status).upper() != "EXIT_SUBMITTING":
                return None
            if not self._cancel_live_protective_stop(
                db,
                position=position,
                reason=reason,
                now=now,
                exit_premium=exit_premium,
            ):
                return None
            return self._submit_market_exit(
                db,
                position=position,
                now=now,
                reason=reason,
                exit_premium=exit_premium,
            )

        if self._is_sandbox_mode() and not self._cancel_live_protective_stop(
            db,
            position=position,
            reason=reason,
            now=now,
            exit_premium=exit_premium,
        ):
            return None

        instrument_key = (position.metadata_json or {}).get("instrument_key") or ""
        balance_before_close = compute_sandbox_portfolio_metrics(
            db,
            settings=self.settings,
        )["available_balance"]

        # Log slippage estimate for exits (never block — exits are always executed)
        try:
            if instrument_key and "|" in instrument_key:
                slip = estimate_slippage(
                    db,
                    symbol=position.symbol,
                    instrument_key=instrument_key,
                    quantity=int(position.quantity),
                    order_type="MARKET",
                    side="SELL",
                    now=now,
                )
                logger.info(
                    "Exit slippage estimate symbol=%s reason=%s bps=%.1f confidence=%.2f",
                    position.symbol, reason, slip.estimated_slippage_bps, slip.confidence,
                )
        except Exception:
            pass

        request = BrokerOrderRequest(
            instrument_key=str(instrument_key),
            option_type=str(position.option_type),
            strike=float(position.strike),
            expiry_date=position.expiry_date.isoformat(),
            side="SELL",
            qty=int(position.quantity),
            order_type="LIMIT" if self._is_sandbox_mode() else "MARKET",
            price=(
                self._sandbox_protected_price(
                    float(exit_premium),
                    "SELL",
                    tick_size=(position.metadata_json or {}).get("contract_tick_size"),
                )
                if self._is_sandbox_mode() and exit_premium is not None
                else None
            ),
            tag=f"exit_{reason.lower()}",
        )
        response = self._place_order_with_retry(
            db,
            request=request,
            action="position_exit",
            resource_id=str(position.id),
        )

        actual_exit_premium = exit_premium
        metadata["exit_reference_quote"] = round(float(exit_premium), 2) if exit_premium is not None else None
        metadata["exit_order_status"] = str(response.status)
        metadata["exit_order_id"] = response.order_id
        metadata["exit_order_requested_at"] = now.isoformat()
        metadata["pending_exit_reason"] = reason

        if not response.success:
            metadata["exit_order_failed"] = True
            metadata["exit_order_error"] = response.message
            position.metadata_json = metadata
            if self._is_sandbox_mode() and str(response.status).upper() == "AMBIGUOUS":
                position.status = "MANUAL_REVIEW"
                metadata["manual_review_required"] = True
                metadata["manual_review_operation"] = "exit"
                metadata["manual_review_message"] = response.message
                position.metadata_json = metadata
            create_audit_log(
                db,
                action="position_exit_failed",
                resource="position",
                resource_id=str(position.id),
                status="ERROR",
                message=response.message,
                details={"symbol": position.symbol, "reason": reason, "status": response.status},
            )
            return self._log_order(
                db,
                position_id=position.id,
                trade_date=position.trade_date,
                symbol=position.symbol,
                order_kind="EXIT",
                side="SELL",
                quantity=position.quantity,
                response=response,
                strike_price=position.strike,
                option_type=position.option_type,
                expiry_date=position.expiry_date,
                entry_premium=position.entry_premium,
                initial_sl=position.initial_sl,
                current_sl=position.current_sl,
                target_premium=position.target_premium,
                peak_premium=position.peak_premium,
                tsl_active=bool(position.tsl_active),
                exit_premium=None,
                exit_reason=reason,
                realized_pnl=None,
                unrealized_pnl=position.unrealized_pnl,
                consensus_reason=position.consensus_reason,
            )

        if actual_exit_premium is None:
            raise RuntimeError("Exit premium is required to close a reconciled position")

        metadata["reconciliation_needed"] = False
        metadata["exit_deferred"] = False
        position.metadata_json = metadata
        position.status = "CLOSED"
        position.closed_at = now
        position.current_price = float(actual_exit_premium)
        position.current_premium = float(actual_exit_premium)
        position.exit_premium = float(actual_exit_premium)
        realized = round(
            (float(actual_exit_premium) - float(position.entry_premium or position.entry_price)) * int(position.quantity),
            2,
        )
        position.pnl_points = round(
            float(actual_exit_premium) - float(position.entry_premium or position.entry_price),
            2,
        )
        position.pnl_value = float(realized)
        position.realized_pnl = float(realized)
        position.unrealized_pnl = 0.0
        position.exit_reason = reason
        position.current_sl = position.current_sl or position.stop_loss
        capital_invested = float(position.entry_premium or position.entry_price or 0.0) * int(position.quantity or 0)
        balance_after_close = round(
            float(balance_before_close or 0.0) + capital_invested + float(position.realized_pnl or 0.0),
            2,
        )
        metadata["capital_invested"] = round(capital_invested, 2)
        metadata["sandbox_balance_before_trade"] = round(float(balance_before_close or 0.0), 2)
        metadata["sandbox_balance_after_trade"] = balance_after_close
        position.metadata_json = metadata
        logger.info(
            "Sandbox exit symbol=%s strike=%s option=%s price=%.2f balance_before=%.2f balance_after=%.2f realized_pnl=%.2f",
            position.symbol,
            position.strike,
            position.option_type,
            float(actual_exit_premium),
            float(balance_before_close or 0.0),
            balance_after_close,
            float(position.realized_pnl or 0.0),
        )
        create_audit_log(
            db,
            action="position_closed",
            resource="position",
            resource_id=str(position.id),
            status="SUCCESS",
            message=reason,
            details={
                "symbol": position.symbol,
                "strike": position.strike,
                "option_type": position.option_type,
                "exit_premium": position.exit_premium,
                "realized_pnl": position.realized_pnl,
            },
        )

        order_row = self._log_order(
            db,
            position_id=position.id,
            trade_date=position.trade_date,
            symbol=position.symbol,
            order_kind="EXIT",
            side="SELL",
            quantity=position.quantity,
            response=response,
            strike_price=position.strike,
            option_type=position.option_type,
            expiry_date=position.expiry_date,
            entry_premium=position.entry_premium,
            initial_sl=position.initial_sl,
            current_sl=position.current_sl,
            target_premium=position.target_premium,
            peak_premium=position.peak_premium,
            tsl_active=bool(position.tsl_active),
            exit_premium=position.exit_premium,
            exit_reason=reason,
            realized_pnl=position.realized_pnl,
            unrealized_pnl=position.unrealized_pnl,
            consensus_reason=position.consensus_reason,
        )
        self._refresh_daily_summary(db, position.trade_date)
        return order_row

    def _manage_open_positions(self, db: Session, now: datetime) -> dict[str, int]:
        updated = 0
        closed = 0
        notifications: list[tuple[ExecutionOrder, ExecutionPosition]] = []
        for position in self._managed_positions(db):
            premium = latest_option_premium(
                db,
                symbol=position.symbol,
                expiry_date=position.expiry_date,
                strike=float(position.strike),
                option_type=str(position.option_type),
                instrument_key=str((position.metadata_json or {}).get("instrument_key") or "") or None,
                settings=self.settings,
            )
            if premium is None:
                continue

            position.current_price = float(premium)
            position.current_premium = float(premium)
            position.peak_premium = max(
                float(position.peak_premium or position.entry_premium or position.entry_price),
                float(premium),
            )
            position.unrealized_pnl = round(
                (float(premium) - float(position.entry_premium or position.entry_price)) * int(position.quantity),
                2,
            )
            position.pnl_value = float(position.unrealized_pnl)
            position.pnl_points = round(float(premium) - float(position.entry_premium or position.entry_price), 2)

            # Standard risk management
            risk_update = update_risk_plan(
                entry_price=float(position.entry_premium or position.entry_price),
                current_price=float(premium),
                initial_sl=float(position.initial_sl or position.stop_loss),
                current_sl=float(position.current_sl or position.stop_loss),
                peak_price=float(position.peak_premium or position.entry_premium or position.entry_price),
                tsl_active=bool(position.tsl_active),
                target_price=float(position.target_premium or position.take_profit or 0.0),
                tsl_activation_percent=float(self.settings.tsl_activation_percent),
                tsl_trail_percent=float(self.settings.tsl_trail_percent),
                tsl_immediate=bool(getattr(self.settings, "tsl_immediate", True)),
            )
            previous_sl = float(position.current_sl or position.stop_loss or 0.0)
            position.current_sl = float(risk_update.current_sl)
            position.trailing_stop = float(risk_update.trailing_sl or position.trailing_stop or 0.0)
            position.tsl_active = bool(risk_update.tsl_active)
            position.peak_premium = float(risk_update.peak_price)
            if risk_update.current_sl > previous_sl:
                self._modify_live_protective_stop(
                    db,
                    position=position,
                    trigger_price=float(risk_update.current_sl),
                )
            
            exit_reason = None
            if self._is_force_squareoff(now):
                exit_reason = "FORCE_SQUAREOFF"
            elif risk_update.exit_triggered:
                exit_reason = str(risk_update.exit_reason)
            
            self._append_position_history(position, now=now, premium=float(premium))

            if exit_reason:
                order_row = self._close_position(
                    db,
                    position=position,
                    now=now,
                    reason=exit_reason,
                    exit_premium=float(premium),
                )
                if order_row is not None:
                    notifications.append((order_row, position))
                if str(position.status).upper() in {"CLOSED", "EXIT_PENDING"}:
                    closed += 1
            updated += 1
        if updated or closed:
            db.commit()
        for order_row, position in notifications:
            self._notify_order(order_row, position)
        return {"updated_positions": updated, "closed_positions": closed}

    def _daily_realized_pnl(self, db: Session, trade_date: date) -> float:
        positions = self._mode_positions_for_day(db, trade_date=trade_date)
        return sum(float(p.realized_pnl or 0.0) for p in positions)

    def _evaluate_symbol(self, db: Session, now: datetime, symbol: str) -> str:
        self._sync_runtime_mode(db)
        if not DIRECTIONAL_SIGNALS_ENABLED:
            return "skip:signals_disabled"
        if not is_option_execution_symbol(symbol):
            return "skip:unsupported_option_underlying"

        # Guard: max simultaneous trades across all symbols
        all_open = self._open_positions(db)
        max_trades = int(getattr(self.settings, "execution_max_simultaneous_trades", 1))
        if len(all_open) >= max_trades:
            return "skip:max_simultaneous_trades_reached"

        # Guard: daily loss limit
        max_daily_loss = self._max_daily_loss_amount()
        daily_pnl = self._daily_realized_pnl(db, now.date())
        if daily_pnl < -max_daily_loss:
            return "skip:daily_loss_limit_breached"

        # Count every accepted entry for the active mode, whether still open or closed.
        daily_trade_count = len(self._entry_positions_for_day(db, trade_date=now.date()))
        max_daily = int(getattr(self.settings, "execution_max_daily_trades", 5))
        if daily_trade_count >= max_daily:
            return "skip:max_daily_trades_reached"

        symbol_trade_count, cooldown_seconds = self._successful_trade_guard(db, now=now, symbol=symbol)
        max_symbol_trades = max(1, int(getattr(self.settings, "signal_max_per_day", 2)))
        if symbol_trade_count >= max_symbol_trades:
            return "skip:max_symbol_trades_reached"
        if cooldown_seconds > 0:
            return f"skip:successful_trade_cooldown:{cooldown_seconds}s"

        context = load_market_context(db, symbol=symbol, settings=self.settings, now=now)
        signal_settings = copy(self.settings)
        signal_settings.signal_max_per_day = 1_000_000
        signal_settings.signal_cooldown_minutes = 1
        signal = build_technical_signal(db, context=context, settings=signal_settings, now=now)
        signal.cooldown_seconds = 0
        signal.max_signals_reached = False
        signal.details = {
            **(signal.details or {}),
            "successful_trades_today": symbol_trade_count,
            "successful_trade_limit": max_symbol_trades,
            "successful_trade_cooldown_seconds": cooldown_seconds,
        }

        # Expire stale entry-candle cache entries (older than cooldown window)
        candle_key = str(signal.details.get("signal_candle_ts") or signal.timestamp.isoformat())
        cooldown_minutes = int(getattr(self.settings, "signal_cooldown_minutes", 12))
        cutoff_key = (now - __import__("datetime").timedelta(minutes=cooldown_minutes)).isoformat()
        self._last_entry_candle = {
            k: v for k, v in self._last_entry_candle.items() if v >= cutoff_key
        }
        if self._last_entry_candle.get(symbol) == candle_key:
            return "skip:duplicate_candle"
        self._last_entry_candle[symbol] = candle_key
        if not self._claim_signal_candle(db, signal=signal):
            return "skip:duplicate_candle"

        log_row = log_signal_decision(db, signal=signal)

        if (
            str(self.settings.execution_mode).lower() == "live"
            and normalize_symbol_key(symbol) in set(getattr(self.settings, "live_execution_blocked_symbol_list", []))
        ):
            log_row.skip_reason = "Live execution blocked for this symbol."
            db.commit()
            return "skip:live_execution_blocked"

        if self._open_positions(db, symbol=symbol):
            log_row.skip_reason = "Open position already active for symbol."
            db.commit()
            return "skip:open_position_active"

        if signal.action not in {"BUY", "SELL"}:
            db.commit()
            return f"skip:{log_row.skip_reason or 'non_trade_signal'}"

        option_selection = build_option_selection(db, context=context, signal=signal, settings=self.settings)
        option_signal = option_selection.signal
        selection_details = {
            "chain_source": option_selection.chain_source,
            "chain_generated_at": (
                option_selection.chain_generated_at.isoformat()
                if option_selection.chain_generated_at is not None
                else None
            ),
            "quote_status": option_signal.get("quote_status"),
            "quote_source": option_signal.get("quote_source"),
            "quote_ts": option_signal.get("quote_ts"),
            "quote_age_seconds": option_signal.get("quote_age_seconds"),
            "requested_atm": option_signal.get("requested_atm"),
            "candidate_diagnostics": option_signal.get("candidate_diagnostics") or [],
            "reasons": option_signal.get("reasons") or [],
        }
        log_row.details = {**(log_row.details or {}), "option_selection": selection_details}
        if option_signal.get("action") != "BUY":
            reasons = [str(reason) for reason in (option_signal.get("reasons") or []) if reason]
            log_row.skip_reason = reasons[0] if reasons else "No liquid option contract passed the live filter."
            db.commit()
            return "skip:no_liquid_strike"

        if (
            str(self.settings.execution_mode).lower() in {"sandbox", "live"}
            and option_selection.chain_source == "synthetic"
        ):
            log_row.skip_reason = "Synthetic option chain is not allowed for broker execution."
            db.commit()
            return "skip:synthetic_chain_broker_blocked"

        entry_price = float(option_signal["entry_price"])
        regime = str(signal.details.get("regime", "TRENDING"))
        base_lots = max(1, int(getattr(self.settings, "execution_lot_size", 1) or 1))
        max_lots = max(1, int(getattr(self.settings, "execution_max_lots", 2)))
        base_lots = min(base_lots, max_lots)
        scaled_lots = compute_position_lots(
            confidence=signal.confidence,
            regime=regime,
            base_lots=base_lots,
            max_lots=max_lots,
        )
        available_balance, balance_source, balance_meta = self._available_trading_balance(db)
        sizing = compute_quantity(
            capital=float(available_balance),
            capital_per_trade_pct=float(self.settings.execution_per_trade_risk_pct),
            entry_price=entry_price,
            lot_size=(
                int(option_signal["lot_size"])
                if self._is_sandbox_mode() and option_signal.get("lot_size")
                else lot_size_for_symbol(symbol)
            ),
            stop_loss_price=float(option_signal["stop_loss"]),
            max_lots=max_lots,
            fixed_lots=scaled_lots,
            vix_level=signal.details.get("vix_level"),
        )
        sizing_meta = {
            **self._sizing_metadata(sizing, balance=available_balance, balance_source=balance_source),
            **balance_meta,
            "strategy_requested_lots": int(scaled_lots),
            "configured_max_lots": int(max_lots),
            "stop_loss_price": round(float(option_signal["stop_loss"]), 2),
        }
        log_row.details = {**(log_row.details or {}), "sizing": sizing_meta}
        if sizing.qty <= 0:
            log_row.skip_reason = (
                f"Position sizing rejected: {sizing.reason}; "
                f"available={available_balance:.2f}, investment_budget={sizing.risk_budget:.2f}, "
                f"risk_per_lot={sizing.risk_per_lot:.2f}"
            )
            db.commit()
            return f"skip:sizing:{sizing.reason}"

        capital_invested = round(float(entry_price) * int(sizing.qty), 2)
        if capital_invested > available_balance:
            log_row.skip_reason = (
                f"Insufficient available balance after sizing: source={balance_source}, "
                f"available={available_balance:.2f}, required={capital_invested:.2f}"
            )
            db.commit()
            return "skip:insufficient_available_balance"
        instrument_key = str(option_signal.get("instrument_key") or "")
        if not instrument_key or "|" not in instrument_key:
            logger.warning(
                "No valid Upstox instrument_key for %s (got %r) — cannot place order",
                symbol, instrument_key,
            )
            log_row.skip_reason = f"missing_instrument_key:{instrument_key or 'empty'}"
            db.commit()
            return "skip:no_instrument_key"
        slippage_meta: dict = {}
        try:
            slip = estimate_slippage(
                db,
                symbol=symbol,
                instrument_key=instrument_key,
                quantity=int(sizing.qty),
                order_type="MARKET",
                side="BUY",
                now=now,
            )
            slippage_meta = {
                "estimated_slippage_bps": slip.estimated_slippage_bps,
                "slippage_confidence": slip.confidence,
                "slippage_details": slip.details,
            }
            # Block if slippage estimate is extreme (>200 bps = 2%)
            if slip.estimated_slippage_bps > 200 and slip.confidence >= 0.7:
                log_row.skip_reason = f"Slippage too high: {slip.estimated_slippage_bps:.1f} bps"
                db.commit()
                return "skip:slippage_too_high"
        except Exception:
            logger.warning("Slippage estimation failed for %s, proceeding anyway", symbol)

        sandbox_entry_price = (
            self._sandbox_protected_price(
                entry_price,
                "BUY",
                tick_size=option_signal.get("tick_size"),
            )
            if self._is_sandbox_mode()
            else None
        )
        request = BrokerOrderRequest(
            instrument_key=instrument_key,
            option_type=str(option_signal["option_type"]),
            strike=float(option_signal["strike"]),
            expiry_date=option_selection.expiry_date.isoformat(),
            side="BUY",
            qty=int(sizing.qty),
            order_type="LIMIT" if self._is_sandbox_mode() else "MARKET",
            price=sandbox_entry_price,
            tag="fast_live_entry",
        )
        response = self._place_order_with_retry(
            db,
            request=request,
            action="position_entry",
            resource_id=f"{symbol}:{option_signal['strike']}:{option_signal['option_type']}",
        )
        if not response.success:
            log_row.skip_reason = f"Broker rejected order: {response.message}"
            if self._is_sandbox_mode() and str(response.status).upper() == "AMBIGUOUS":
                review_position = ExecutionPosition(
                    trade_date=now.date(),
                    symbol=symbol,
                    interval="1minute",
                    strategy_name="fast_live_breakout",
                    option_type=str(option_signal["option_type"]),
                    side="BUY",
                    expiry_date=option_selection.expiry_date,
                    strike=float(option_signal["strike"]),
                    quantity=int(sizing.qty),
                    status="MANUAL_REVIEW",
                    entry_price=entry_price,
                    entry_premium=entry_price,
                    stop_loss=float(option_signal["stop_loss"]),
                    initial_sl=float(option_signal["stop_loss"]),
                    current_sl=float(option_signal["stop_loss"]),
                    trailing_stop=float(option_signal["stop_loss"]),
                    peak_premium=entry_price,
                    take_profit=float(option_signal["take_profit"]),
                    target_premium=float(option_signal["take_profit"]),
                    current_price=entry_price,
                    current_premium=entry_price,
                    entry_order_id=None,
                    opened_at=now,
                    metadata_json={
                        "execution_mode": "sandbox",
                        "instrument_key": instrument_key,
                        "manual_review_required": True,
                        "manual_review_operation": "entry",
                        "manual_review_message": response.message,
                        "entry_reference_price": entry_price,
                        "entry_requested_quantity": int(sizing.qty),
                        "sandbox_limit_price": sandbox_entry_price,
                        "contract_lot_size": option_signal.get("lot_size"),
                        "contract_tick_size": option_signal.get("tick_size"),
                        "contract_freeze_quantity": option_signal.get("freeze_quantity"),
                        "signal_log_id": log_row.id,
                    },
                )
                db.add(review_position)
                db.flush()
                self._log_order(
                    db,
                    position_id=review_position.id,
                    trade_date=review_position.trade_date,
                    symbol=review_position.symbol,
                    order_kind="ENTRY",
                    side="BUY",
                    quantity=review_position.quantity,
                    response=response,
                    strike_price=review_position.strike,
                    option_type=review_position.option_type,
                    expiry_date=review_position.expiry_date,
                    entry_premium=review_position.entry_premium,
                    initial_sl=review_position.initial_sl,
                    current_sl=review_position.current_sl,
                    target_premium=review_position.target_premium,
                    peak_premium=review_position.peak_premium,
                    tsl_active=False,
                    consensus_reason="Sandbox entry requires manual review",
                )
            create_audit_log(
                db,
                action="position_entry_failed",
                resource="signal",
                resource_id=str(log_row.id),
                status="ERROR",
                message=response.message,
                details={"symbol": symbol, "strike": option_signal["strike"], "status": response.status},
            )
            db.commit()
            return "skip:broker_reject"

        balance_before_trade = round(float(available_balance), 2)
        balance_after_trade = round(balance_before_trade - capital_invested, 2)
        logger.info(
            "Entry selected symbol=%s strike=%s option=%s instrument=%s entry_price=%.2f qty=%d balance_before=%.2f balance_after=%.2f",
            symbol,
            option_signal["strike"],
            option_signal["option_type"],
            instrument_key,
            entry_price,
            int(sizing.qty),
            balance_before_trade,
            balance_after_trade,
        )

        position = ExecutionPosition(
            trade_date=now.date(),
            symbol=symbol,
            interval="1minute",
            strategy_name="fast_live_breakout",
            option_type=str(option_signal["option_type"]),
            side="BUY",
            expiry_date=option_selection.expiry_date,
            strike=float(option_signal["strike"]),
            quantity=int(sizing.qty),
            status="ENTRY_PENDING" if self._is_live_mode() else "OPEN",
            entry_price=entry_price,
            entry_premium=entry_price,
            stop_loss=float(option_signal["stop_loss"]),
            initial_sl=float(option_signal["stop_loss"]),
            current_sl=float(option_signal["stop_loss"]),
            trailing_stop=float(option_signal["stop_loss"]),
            peak_premium=entry_price,
            tsl_active=False,
            take_profit=float(option_signal["take_profit"]),
            target_premium=float(option_signal["take_profit"]),
            current_price=entry_price,
            current_premium=entry_price,
            pnl_points=0.0,
            pnl_value=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            ml_confidence=float(signal.confidence),
            ai_score=None,
            pine_signal=None,
            consensus_reason=f"Score {signal.score:.1f} | {' | '.join(signal.reasons[:2])}",
            entry_order_id=getattr(response, "order_id", None),
            opened_at=now,
            metadata_json={
                "instrument_key": option_signal.get("instrument_key"),
                "signal_action": signal.action,
                "signal_bias": signal.bias,
                "signal_score": signal.score,
                "signal_reasons": signal.reasons,
                "chain_source": option_selection.chain_source,
                "latest_quote_status": option_signal.get("quote_status"),
                "latest_quote_source": option_signal.get("quote_source"),
                "latest_quote_ts": option_signal.get("quote_ts"),
                "latest_quote_age_seconds": option_signal.get("quote_age_seconds"),
                "execution_mode": str(self.settings.execution_mode).lower(),
                "entry_order_id": getattr(response, "order_id", None),
                "entry_order_status": str(response.status),
                "entry_requested_quantity": int(sizing.qty),
                "entry_reference_price": entry_price,
                "contract_lot_size": option_signal.get("lot_size"),
                "contract_tick_size": option_signal.get("tick_size"),
                "contract_freeze_quantity": option_signal.get("freeze_quantity"),
                "entry_reconciliation_needed": self._is_live_mode(),
                "capital_invested": capital_invested,
                **(
                    {
                        "sandbox_balance_before_trade": balance_before_trade,
                        "sandbox_balance_after_trade": balance_after_trade,
                    }
                    if self._is_sandbox_mode()
                    else {
                        "paper_balance_before_trade": balance_before_trade,
                        "paper_balance_after_trade": balance_after_trade,
                    }
                ),
                "sizing": sizing_meta,
                **slippage_meta,
                "premium_history": [
                    {
                        "timestamp": now.isoformat(),
                        "premium": round(entry_price, 2),
                        "current_sl": round(float(option_signal["stop_loss"]), 2),
                        "tsl_active": False,
                        "unrealized_pnl": 0.0,
                    }
                ],
                "signal_log_id": log_row.id,
            },
        )
        db.add(position)
        db.flush()

        entry_order = self._log_order(
            db,
            position_id=position.id,
            trade_date=position.trade_date,
            symbol=position.symbol,
            order_kind="ENTRY",
            side="BUY",
            quantity=position.quantity,
            response=response,
            strike_price=position.strike,
            option_type=position.option_type,
            expiry_date=position.expiry_date,
            entry_premium=position.entry_premium,
            initial_sl=position.initial_sl,
            current_sl=position.current_sl,
            target_premium=position.target_premium,
            peak_premium=position.peak_premium,
            tsl_active=bool(position.tsl_active),
            exit_premium=None,
            exit_reason=None,
            realized_pnl=position.realized_pnl,
            unrealized_pnl=position.unrealized_pnl,
            consensus_reason=f"Score {signal.score:.1f} | {' | '.join(option_signal.get('reasons') or [])}",
        )
        protective_order = None
        if self._is_live_mode():
            payload = getattr(response, "payload", {}) or {}
            broker_status = self._extract_order_status(payload) or str(response.status or "").upper()
            filled_quantity = self._extract_filled_quantity(payload)
            fill_price = self._extract_fill_price(payload)
            if (
                broker_status in {"COMPLETE", "COMPLETED", "FILLED"}
                and filled_quantity > 0
                and fill_price is not None
            ):
                protective_order = self._finalize_live_entry(
                    db,
                    position=position,
                    now=now,
                    filled_quantity=min(filled_quantity, int(sizing.qty)),
                    fill_price=fill_price,
                    broker_status=broker_status,
                )
                entry_order.status = broker_status
                entry_order.quantity = position.quantity
                entry_order.price = fill_price
                entry_order.entry_premium = fill_price
                if not bool((position.metadata_json or {}).get("broker_sl_active")):
                    exit_order = self._close_position(
                        db,
                        position=position,
                        now=now,
                        reason="PROTECTIVE_SL_FAILED",
                        exit_premium=float(fill_price),
                    )
                    log_row.trade_placed = False
                    log_row.skip_reason = "protective_sl_failed"
                    db.commit()
                    self._notify_order(entry_order, position)
                    if exit_order is not None:
                        self._notify_order(exit_order, position)
                    return "skip:protective_sl_failed"
            else:
                log_row.trade_placed = False
                log_row.skip_reason = (
                    "entry_submission_ambiguous"
                    if str(response.status).upper() == "AMBIGUOUS"
                    else "entry_pending_broker_confirmation"
                )
                db.commit()
                self._notify_order(entry_order, position)
                return "entry_pending"

        if self._is_sandbox_mode():
            protective_order = self._place_live_protective_stop(db, position=position)
        if str(position.status).upper() == "MANUAL_REVIEW":
            log_row.trade_placed = False
            log_row.skip_reason = "sandbox_manual_review_required"
            db.commit()
            self._notify_order(entry_order, position)
            if protective_order is not None:
                self._notify_order(protective_order, position)
            return "manual_review"
        if self._uses_broker_protection() and not bool((position.metadata_json or {}).get("broker_sl_active")):
            exit_order = self._close_position(
                db,
                position=position,
                now=now,
                reason="PROTECTIVE_SL_FAILED",
                exit_premium=float(position.current_premium or position.entry_premium or position.entry_price),
            )
            log_row.trade_placed = False
            log_row.skip_reason = "protective_sl_failed"
            db.commit()
            self._notify_order(entry_order, position)
            if protective_order is not None:
                self._notify_order(protective_order, position)
            if exit_order is not None:
                self._notify_order(exit_order, position)
            return "skip:protective_sl_failed"
        log_row.trade_placed = True
        log_row.skip_reason = None
        log_row.details = {
            **(log_row.details or {}),
            "selected_strike": position.strike,
            "option_type": position.option_type,
            "expiry_date": position.expiry_date.isoformat(),
            "quantity": position.quantity,
            "chain_source": option_selection.chain_source,
            "quote_status": option_signal.get("quote_status"),
            "quote_source": option_signal.get("quote_source"),
            "quote_ts": option_signal.get("quote_ts"),
            "quote_age_seconds": option_signal.get("quote_age_seconds"),
            "instrument_key": instrument_key,
            "entry_price": entry_price,
            **(
                {
                    "sandbox_balance_before_trade": balance_before_trade,
                    "sandbox_balance_after_trade": balance_after_trade,
                }
                if self._is_sandbox_mode()
                else {
                    "paper_balance_before_trade": balance_before_trade,
                    "paper_balance_after_trade": balance_after_trade,
                }
            ),
            "sizing": sizing_meta,
        }
        self._refresh_daily_summary(db, position.trade_date)
        create_audit_log(
            db,
            action="position_opened",
            resource="position",
            resource_id=str(position.id),
            status="SUCCESS",
            message="entry_filled",
            details={
                "symbol": symbol,
                "strike": position.strike,
                "option_type": position.option_type,
                "instrument_key": instrument_key,
                "entry_price": entry_price,
                "quantity": position.quantity,
                "mode": str(self.settings.execution_mode).lower(),
            },
        )
        db.commit()
        self._notify_order(entry_order, position)
        if protective_order is not None:
            self._notify_order(protective_order, position)
        return "entered"

    def _force_square_off(self, db: Session, now: datetime, reason: str) -> dict[str, Any]:
        closed = 0
        deferred = 0
        reconciliation_pending = 0
        notifications: list[tuple[ExecutionOrder, ExecutionPosition]] = []
        for position in self._open_positions(db):
            if str(position.status).upper() != "OPEN":
                reconciliation_pending += 1
                continue
            premium = latest_option_premium(
                db,
                symbol=position.symbol,
                expiry_date=position.expiry_date,
                strike=float(position.strike),
                option_type=str(position.option_type),
                instrument_key=str((position.metadata_json or {}).get("instrument_key") or "") or None,
                settings=self.settings,
            )
            order_row = self._close_position(
                db,
                position=position,
                now=now,
                reason=reason,
                exit_premium=float(premium) if premium is not None else None,
            )
            if order_row is not None:
                notifications.append((order_row, position))
            if str(position.status).upper() == "CLOSED":
                closed += 1
            elif str(position.status).upper() == "EXIT_PENDING":
                reconciliation_pending += 1
            else:
                deferred += 1
        cancel_response = self.broker.cancel_all_pending()
        db.commit()
        for order_row, position in notifications:
            self._notify_order(order_row, position)
        return {
            "square_off_closed": closed,
            "square_off_deferred": deferred,
            "square_off_reconciliation_pending": reconciliation_pending,
            "cancel_pending_status": cancel_response.status,
        }

    def run_once(self, db: Session, now: datetime | None = None) -> dict[str, Any]:
        now = now or _now_ist()
        runtime_mode = self._sync_runtime_mode(db)
        if not bool(self.settings.execution_enabled):
            return {"status": "disabled", "at": now.isoformat()}
        broker_ready, broker_reason = self._live_broker_ready()
        if not broker_ready:
            create_audit_log(
                db,
                action="live_broker_not_ready",
                resource="broker",
                resource_id="upstox",
                status="ERROR",
                message=broker_reason,
                details={"mode": runtime_mode},
            )
            db.commit()
            return {
                "status": "live_broker_not_ready",
                "mode": runtime_mode,
                "at": now.isoformat(),
                "reason": broker_reason,
            }
        entry_reconciliation = self._reconcile_pending_entries(db, now) if self._is_live_mode() else {
            "reconciled_entries": 0,
            "pending_entry_reconciliations": 0,
            "failed_entries": 0,
        }
        reconciliation = self._reconcile_pending_exits(db, now) if self._is_live_mode() else {
            "reconciled_exits": 0,
            "pending_exit_reconciliations": 0,
            "reopened_exits": 0,
        }
        if not is_trading_day(now.date()):
            return {
                "status": "non_trading_day",
                "at": now.isoformat(),
                **entry_reconciliation,
                **reconciliation,
            }

        manage = self._manage_open_positions(db, now)
        daily_total_pnl = self._daily_total_pnl(db, now.date())
        if daily_total_pnl < -self._max_daily_loss_amount() and self._open_positions(db):
            square = self._force_square_off(db, now, "MAX_DAILY_LOSS")
            create_audit_log(
                db,
                action="max_daily_loss_squareoff",
                resource="risk",
                resource_id=now.date().isoformat(),
                status="WARN",
                message="Auto square-off triggered by max daily loss",
                details={"daily_total_pnl": daily_total_pnl, "limit": self._max_daily_loss_amount()},
            )
            return {
                "status": "max_daily_loss_squareoff",
                "mode": runtime_mode,
                "at": now.isoformat(),
                **entry_reconciliation,
                **reconciliation,
                **manage,
                **square,
            }
        if self._is_force_squareoff(now):
            square = self._force_square_off(db, now, "FORCE_SQUAREOFF")
            return {
                "status": "force_squareoff",
                "mode": runtime_mode,
                "at": now.isoformat(),
                **entry_reconciliation,
                **reconciliation,
                **manage,
                **square,
            }
        if not self._is_entry_window(now):
            db.commit()
            return {
                "status": "outside_entry_window",
                "mode": runtime_mode,
                "at": now.isoformat(),
                **entry_reconciliation,
                **reconciliation,
                **manage,
            }

        symbol_results: dict[str, str] = {}
        for symbol in self.settings.execution_symbol_list:
            try:
                symbol_results[symbol] = self._evaluate_symbol(db, now, symbol)
            except Exception as exc:
                db.rollback()
                logger.exception("Execution cycle failed for symbol=%s", symbol)
                symbol_results[symbol] = f"error:{exc}"
        return {
            "status": "ok",
            "mode": runtime_mode,
            "at": now.isoformat(),
            **entry_reconciliation,
            **reconciliation,
            **manage,
            "symbols": symbol_results,
        }

    def emergency_exit_all(self, db: Session, now: datetime | None = None) -> dict[str, Any]:
        now = now or _now_ist()
        out = self._force_square_off(db, now, "MANUAL")
        return {"status": "emergency_exit", "at": now.isoformat(), **out}

    def close_position_by_id(self, db: Session, position_id: int, now: datetime | None = None) -> dict[str, Any]:
        now = now or _now_ist()
        position = db.get(ExecutionPosition, position_id)
        if position is None:
            return {"status": "not_found", "position_id": position_id}
        if self._position_execution_mode(position) != str(self.settings.execution_mode).lower():
            return {"status": "not_found", "position_id": position_id}
        if str(position.status).upper() != "OPEN":
            return {"status": "already_closed", "position_id": position_id}
        premium = latest_option_premium(
            db,
            symbol=position.symbol,
            expiry_date=position.expiry_date,
            strike=float(position.strike),
            option_type=str(position.option_type),
            instrument_key=str((position.metadata_json or {}).get("instrument_key") or "") or None,
            settings=self.settings,
        )
        order_row = self._close_position(
            db,
            position=position,
            now=now,
            reason="MANUAL",
            exit_premium=float(premium) if premium is not None else None,
        )
        db.commit()
        if order_row is not None:
            self._notify_order(order_row, position)
        if str(position.status).upper() == "CLOSED":
            return {"status": "closed", "position_id": position_id, "exit_premium": position.exit_premium}
        if str(position.status).upper() == "EXIT_PENDING":
            return {"status": "reconciliation_pending", "position_id": position_id, "exit_premium": None}
        return {"status": "exit_deferred", "position_id": position_id, "exit_premium": None}

    def daily_report(self, db: Session, trade_date: date | None = None) -> dict[str, Any]:
        trade_date = trade_date or _now_ist().date()
        self._refresh_daily_summary(db, trade_date)
        db.commit()
        summary = db.get(DailySummary, trade_date)
        total_profit = float(summary.total_pnl if summary is not None else 0.0)
        win_rate = float((summary.win_rate / 100.0) if summary is not None else 0.0)

        positions = self._mode_positions_for_day(db, trade_date=trade_date, statuses={"CLOSED"})
        equity = float(self.settings.execution_capital)
        peak = equity
        max_drawdown_pct = 0.0
        for position in positions:
            equity += float(position.realized_pnl or position.pnl_value or 0.0)
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown_pct = max(max_drawdown_pct, ((peak - equity) / peak) * 100.0)

        position_ids = [position.id for position in self._mode_positions_for_day(db, trade_date=trade_date)]
        signal_count = 0
        if position_ids:
            signal_count = db.scalar(
                select(func.count())
                .select_from(ExecutionOrder)
                .where(
                    ExecutionOrder.trade_date == trade_date,
                    ExecutionOrder.position_id.in_(position_ids),
                )
            ) or 0
        return {
            "trade_date": trade_date.isoformat(),
            "total_trades": int(summary.total_trades if summary is not None else 0),
            "win_rate": float(win_rate),
            "max_drawdown": float(round(max_drawdown_pct, 4)),
            "total_profit": float(total_profit),
            "missed_signals": 0,
            "executed_signals": int(summary.total_trades if summary is not None else 0),
            "total_signal_events": int(signal_count),
        }
