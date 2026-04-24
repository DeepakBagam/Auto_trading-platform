from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from db.models import DailySummary, ExecutionOrder, ExecutionPosition
from execution_engine.broker import BaseBroker, BrokerOrderRequest, PaperBroker, UpstoxBroker
from execution_engine.live_service import (
    DIRECTIONAL_SIGNALS_ENABLED,
    build_option_selection,
    build_technical_signal,
    latest_option_premium,
    load_market_context,
    log_signal_decision,
)
from execution_engine.risk_manager import compute_quantity, update_risk_plan
from execution_engine.slippage_tracker import estimate_slippage
from execution_engine.strike_selector import compute_position_lots, lot_size_for_symbol
from utils.calendar_utils import is_trading_day
from utils.config import Settings, get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger
from utils.notifications import send_order_notification
from utils.symbols import normalize_symbol_key, symbol_value_filter

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
        self._last_entry_candle: dict[str, str] = {}

    def _build_broker(self) -> BaseBroker:
        if str(self.settings.execution_mode).lower() == "live":
            return UpstoxBroker(
                base_url=self.settings.upstox_base_url,
                access_token=self.settings.upstox_access_token,
            )
        return PaperBroker()

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

    def _open_positions(self, db: Session, symbol: str | None = None) -> list[ExecutionPosition]:
        query = select(ExecutionPosition).where(ExecutionPosition.status == "OPEN")
        if symbol:
            query = query.where(symbol_value_filter(ExecutionPosition.symbol, symbol))
        return db.execute(query.order_by(ExecutionPosition.opened_at.asc())).scalars().all()

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
        }
        send_order_notification(payload, settings=self.settings)

    def _refresh_daily_summary(self, db: Session, trade_date: date) -> None:
        positions = (
            db.execute(
                select(ExecutionPosition)
                .where(
                    and_(
                        ExecutionPosition.trade_date == trade_date,
                        ExecutionPosition.status == "CLOSED",
                    )
                )
                .order_by(ExecutionPosition.closed_at.asc())
            )
            .scalars()
            .all()
        )
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

    def _close_position(
        self,
        db: Session,
        *,
        position: ExecutionPosition,
        now: datetime,
        reason: str,
        exit_premium: float,
    ) -> ExecutionOrder:
        instrument_key = (position.metadata_json or {}).get("instrument_key") or ""

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
            order_type="MARKET",
            tag=f"exit_{reason.lower()}",
        )
        response = self.broker.place_order(request)

        position.status = "CLOSED"
        position.closed_at = now
        position.current_price = float(exit_premium)
        position.current_premium = float(exit_premium)
        position.exit_premium = float(exit_premium)
        realized = round(
            (float(exit_premium) - float(position.entry_premium or position.entry_price)) * int(position.quantity),
            2,
        )
        position.pnl_points = round(float(exit_premium) - float(position.entry_premium or position.entry_price), 2)
        position.pnl_value = float(realized)
        position.realized_pnl = float(realized)
        position.unrealized_pnl = 0.0
        position.exit_reason = reason
        position.current_sl = position.current_sl or position.stop_loss

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
        for position in self._open_positions(db):
            premium = latest_option_premium(
                db,
                symbol=position.symbol,
                expiry_date=position.expiry_date,
                strike=float(position.strike),
                option_type=str(position.option_type),
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
            position.current_sl = float(risk_update.current_sl)
            position.trailing_stop = float(risk_update.current_sl)
            position.tsl_active = bool(risk_update.tsl_active)
            position.peak_premium = float(risk_update.peak_price)
            self._append_position_history(position, now=now, premium=float(premium))

            exit_reason = None
            if self._is_force_squareoff(now):
                exit_reason = "FORCE_SQUAREOFF"
            elif risk_update.exit_triggered:
                exit_reason = str(risk_update.exit_reason)

            if exit_reason:
                order_row = self._close_position(
                    db,
                    position=position,
                    now=now,
                    reason=exit_reason,
                    exit_premium=float(premium),
                )
                notifications.append((order_row, position))
                closed += 1
            updated += 1
        if updated or closed:
            db.commit()
        for order_row, position in notifications:
            self._notify_order(order_row, position)
        return {"updated_positions": updated, "closed_positions": closed}

    def _daily_realized_pnl(self, db: Session, trade_date: date) -> float:
        positions = db.execute(
            select(ExecutionPosition).where(
                and_(
                    ExecutionPosition.trade_date == trade_date,
                    ExecutionPosition.status == "CLOSED",
                )
            )
        ).scalars().all()
        return sum(float(p.realized_pnl or p.pnl_value or 0.0) for p in positions)

    def _evaluate_symbol(self, db: Session, now: datetime, symbol: str) -> str:
        if not DIRECTIONAL_SIGNALS_ENABLED:
            return "skip:signals_disabled"

        # Guard: max simultaneous trades across all symbols
        all_open = self._open_positions(db)
        max_trades = int(getattr(self.settings, "execution_max_simultaneous_trades", 1))
        if len(all_open) >= max_trades:
            return "skip:max_simultaneous_trades_reached"

        # Guard: daily loss limit
        capital = float(self.settings.execution_capital)
        max_daily_loss = capital * float(getattr(self.settings, "execution_max_daily_loss_pct", 0.05))
        daily_pnl = self._daily_realized_pnl(db, now.date())
        if daily_pnl < -max_daily_loss:
            return "skip:daily_loss_limit_breached"

        # Guard: max daily trades
        daily_trade_count = db.scalar(
            select(func.count()).select_from(ExecutionPosition).where(
                and_(
                    ExecutionPosition.trade_date == now.date(),
                    ExecutionPosition.status == "CLOSED",
                )
            )
        ) or 0
        max_daily = int(getattr(self.settings, "execution_max_daily_trades", 5))
        if daily_trade_count >= max_daily:
            return "skip:max_daily_trades_reached"

        context = load_market_context(db, symbol=symbol, settings=self.settings, now=now)
        signal = build_technical_signal(db, context=context, settings=self.settings, now=now)

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
        if option_signal.get("action") != "BUY":
            log_row.skip_reason = "No liquid option contract passed the live filter."
            db.commit()
            return "skip:no_liquid_strike"

        if str(self.settings.execution_mode).lower() == "live" and option_selection.chain_source == "synthetic":
            log_row.skip_reason = "Synthetic option chain is not allowed in live mode."
            db.commit()
            return "skip:synthetic_chain_live_blocked"

        entry_price = float(option_signal["entry_price"])
        regime = str(signal.details.get("regime", "TRENDING"))
        base_lots = max(1, int(getattr(self.settings, "execution_lot_size", 1) or 1))
        max_lots = max(base_lots, int(getattr(self.settings, "execution_max_lots", 2)))
        scaled_lots = compute_position_lots(
            confidence=signal.confidence,
            regime=regime,
            base_lots=base_lots,
            max_lots=max_lots,
        )
        sizing = compute_quantity(
            capital=float(self.settings.execution_capital),
            capital_per_trade_pct=float(self.settings.execution_per_trade_risk_pct),
            entry_price=entry_price,
            lot_size=lot_size_for_symbol(symbol),
            fixed_lots=scaled_lots,
        )
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

        request = BrokerOrderRequest(
            instrument_key=instrument_key,
            option_type=str(option_signal["option_type"]),
            strike=float(option_signal["strike"]),
            expiry_date=option_selection.expiry_date.isoformat(),
            side="BUY",
            qty=int(sizing.qty),
            order_type="MARKET",
            tag="fast_live_entry",
        )
        response = self.broker.place_order(request)
        if not response.success:
            log_row.skip_reason = f"Broker rejected order: {response.message}"
            db.commit()
            return "skip:broker_reject"

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
            status="OPEN",
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
        log_row.trade_placed = True
        log_row.skip_reason = None
        log_row.details = {
            **(log_row.details or {}),
            "selected_strike": position.strike,
            "option_type": position.option_type,
            "expiry_date": position.expiry_date.isoformat(),
            "quantity": position.quantity,
            "chain_source": option_selection.chain_source,
        }
        self._refresh_daily_summary(db, position.trade_date)
        db.commit()
        self._notify_order(entry_order, position)
        return "entered"

    def _force_square_off(self, db: Session, now: datetime, reason: str) -> dict[str, Any]:
        closed = 0
        notifications: list[tuple[ExecutionOrder, ExecutionPosition]] = []
        for position in self._open_positions(db):
            premium = float(position.current_premium or position.entry_premium or position.entry_price)
            order_row = self._close_position(db, position=position, now=now, reason=reason, exit_premium=premium)
            notifications.append((order_row, position))
            closed += 1
        cancel_response = self.broker.cancel_all_pending()
        db.commit()
        for order_row, position in notifications:
            self._notify_order(order_row, position)
        return {"square_off_closed": closed, "cancel_pending_status": cancel_response.status}

    def run_once(self, db: Session, now: datetime | None = None) -> dict[str, Any]:
        now = now or _now_ist()
        if not bool(self.settings.execution_enabled):
            return {"status": "disabled", "at": now.isoformat()}
        if not is_trading_day(now.date()):
            return {"status": "non_trading_day", "at": now.isoformat()}

        manage = self._manage_open_positions(db, now)
        if self._is_force_squareoff(now):
            square = self._force_square_off(db, now, "FORCE_SQUAREOFF")
            return {"status": "force_squareoff", "at": now.isoformat(), **manage, **square}
        if not self._is_entry_window(now):
            db.commit()
            return {"status": "outside_entry_window", "at": now.isoformat(), **manage}

        symbol_results: dict[str, str] = {}
        for symbol in self.settings.execution_symbol_list:
            try:
                symbol_results[symbol] = self._evaluate_symbol(db, now, symbol)
            except Exception as exc:
                db.rollback()
                logger.exception("Execution cycle failed for symbol=%s", symbol)
                symbol_results[symbol] = f"error:{exc}"
        return {"status": "ok", "at": now.isoformat(), **manage, "symbols": symbol_results}

    def emergency_exit_all(self, db: Session, now: datetime | None = None) -> dict[str, Any]:
        now = now or _now_ist()
        out = self._force_square_off(db, now, "MANUAL")
        return {"status": "emergency_exit", "at": now.isoformat(), **out}

    def close_position_by_id(self, db: Session, position_id: int, now: datetime | None = None) -> dict[str, Any]:
        now = now or _now_ist()
        position = db.get(ExecutionPosition, position_id)
        if position is None:
            return {"status": "not_found", "position_id": position_id}
        if str(position.status).upper() != "OPEN":
            return {"status": "already_closed", "position_id": position_id}
        premium = latest_option_premium(
            db,
            symbol=position.symbol,
            expiry_date=position.expiry_date,
            strike=float(position.strike),
            option_type=str(position.option_type),
        )
        exit_premium = float(premium or position.current_premium or position.entry_premium or position.entry_price)
        order_row = self._close_position(
            db,
            position=position,
            now=now,
            reason="MANUAL",
            exit_premium=exit_premium,
        )
        db.commit()
        self._notify_order(order_row, position)
        return {"status": "closed", "position_id": position_id, "exit_premium": exit_premium}

    def daily_report(self, db: Session, trade_date: date | None = None) -> dict[str, Any]:
        trade_date = trade_date or _now_ist().date()
        summary = db.get(DailySummary, trade_date)
        if summary is None:
            self._refresh_daily_summary(db, trade_date)
            db.commit()
            summary = db.get(DailySummary, trade_date)
        total_profit = float(summary.total_pnl if summary is not None else 0.0)
        win_rate = float((summary.win_rate / 100.0) if summary is not None else 0.0)

        positions = (
            db.execute(
                select(ExecutionPosition)
                .where(
                    and_(
                        ExecutionPosition.trade_date == trade_date,
                        ExecutionPosition.status == "CLOSED",
                    )
                )
                .order_by(ExecutionPosition.closed_at.asc())
            )
            .scalars()
            .all()
        )
        equity = float(self.settings.execution_capital)
        peak = equity
        max_drawdown_pct = 0.0
        for position in positions:
            equity += float(position.realized_pnl or position.pnl_value or 0.0)
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown_pct = max(max_drawdown_pct, ((peak - equity) / peak) * 100.0)

        signal_count = db.scalar(
            select(func.count()).select_from(ExecutionOrder).where(ExecutionOrder.trade_date == trade_date)
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
