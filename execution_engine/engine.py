from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from api.routes.options import options_signal
from data_layer.collectors.upstox_option_chain import UpstoxOptionChainCollector
from db.models import DailySummary, ExecutionExternalSignal, ExecutionOrder, ExecutionPosition, OptionQuote, RawCandle
from execution_engine.broker import BaseBroker, BrokerOrderRequest, PaperBroker, UpstoxBroker
from execution_engine.risk_manager import build_risk_plan, compute_quantity, update_risk_plan
from execution_engine.strike_selector import lot_size_for_symbol, select_option_contract
from prediction_engine.consensus_engine import ConsensusResult, get_consensus_signal, log_consensus_result
from utils.calendar_utils import is_trading_day
from utils.config import Settings, get_settings
from utils.constants import IST_ZONE
from utils.logger import get_logger
from utils.notifications import send_order_notification
from utils.symbols import instrument_key_filter, normalize_symbol_key, symbol_aliases, symbol_value_filter

logger = get_logger(__name__)


def _now_ist() -> datetime:
    return datetime.now(IST_ZONE)


def _parse_time(value: str, fallback: time) -> time:
    try:
        hour, minute = str(value).split(":", 1)
        return time(int(hour), int(minute))
    except Exception:
        return fallback


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else default
    except (TypeError, ValueError):
        return default


class IntradayOptionsExecutionEngine:
    def __init__(self, settings: Settings | None = None, broker: BaseBroker | None = None) -> None:
        self.settings = settings or get_settings()
        self.broker = broker or self._build_broker()
        self._last_entry_candle: dict[str, datetime] = {}

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

    def _latest_candle_ts(self, db: Session, symbol: str, interval: str) -> datetime | None:
        return db.scalar(
            select(func.max(RawCandle.ts)).where(
                and_(
                    instrument_key_filter(RawCandle.instrument_key, symbol),
                    RawCandle.interval == interval,
                )
            )
        )

    def _open_positions(self, db: Session, symbol: str | None = None) -> list[ExecutionPosition]:
        query = select(ExecutionPosition).where(ExecutionPosition.status == "OPEN")
        if symbol:
            query = query.where(symbol_value_filter(ExecutionPosition.symbol, symbol))
        return db.execute(query.order_by(ExecutionPosition.opened_at.asc())).scalars().all()

    def _latest_option_premium(
        self,
        db: Session,
        *,
        symbol: str,
        expiry_date: date,
        strike: float,
        option_type: str,
    ) -> float | None:
        premium = db.scalar(
            select(OptionQuote.ltp)
            .where(
                and_(
                    symbol_value_filter(OptionQuote.underlying_symbol, symbol),
                    OptionQuote.expiry_date == expiry_date,
                    OptionQuote.strike == float(strike),
                    OptionQuote.option_type == option_type,
                )
            )
            .order_by(OptionQuote.ts.desc())
            .limit(1)
        )
        return float(premium) if premium is not None else None

    def _resolve_underlying_key(self, symbol: str) -> str | None:
        aliases = {normalize_symbol_key(value) for value in symbol_aliases(symbol)}
        for instrument_key in self.settings.instrument_keys:
            display = instrument_key.split("|", 1)[1] if "|" in instrument_key else instrument_key
            if normalize_symbol_key(display) in aliases:
                return instrument_key
        return None

    def _refresh_option_chain_for_symbol(self, db: Session, *, symbol: str, expiry_date: date) -> bool:
        if not self.settings.has_market_data_access:
            return False
        underlying_key = self._resolve_underlying_key(symbol)
        if underlying_key is None:
            return False
        try:
            UpstoxOptionChainCollector().sync_option_chain(
                db,
                underlying_key=underlying_key,
                underlying_symbol=symbol,
                expiry_date=expiry_date,
            )
            return True
        except Exception:
            db.rollback()
            logger.exception(
                "Option chain refresh failed for symbol=%s expiry=%s",
                symbol,
                expiry_date,
            )
            return False

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
        ml_confidence: float | None = None,
        ai_score: float | None = None,
        pine_signal: str | None = None,
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
            ml_confidence=ml_confidence,
            ai_score=ai_score,
            pine_signal=pine_signal,
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
            "capital_invested": (
                float(getattr(position, "entry_premium", 0.0) or getattr(position, "entry_price", 0.0) or 0.0)
                * float(getattr(position, "quantity", 0) or 0)
                if position is not None
                else None
            ),
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
        instrument_key = (position.metadata_json or {}).get("instrument_key") or position.symbol
        request = BrokerOrderRequest(
            symbol=str(instrument_key),
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
        realized = round((float(exit_premium) - float(position.entry_premium or position.entry_price)) * int(position.quantity), 2)
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
            ml_confidence=position.ml_confidence,
            ai_score=position.ai_score,
            pine_signal=position.pine_signal,
            consensus_reason=position.consensus_reason,
        )
        self._refresh_daily_summary(db, position.trade_date)
        return order_row

    def _manage_open_positions(self, db: Session, now: datetime) -> dict[str, int]:
        updated = 0
        closed = 0
        notifications: list[tuple[ExecutionOrder, ExecutionPosition]] = []
        refreshed_pairs: set[tuple[str, date]] = set()
        for position in self._open_positions(db):
            premium = self._latest_option_premium(
                db,
                symbol=position.symbol,
                expiry_date=position.expiry_date,
                strike=float(position.strike),
                option_type=str(position.option_type),
            )
            refresh_key = (str(position.symbol), position.expiry_date)
            if premium is None and refresh_key not in refreshed_pairs:
                refreshed_pairs.add(refresh_key)
                self._refresh_option_chain_for_symbol(
                    db,
                    symbol=str(position.symbol),
                    expiry_date=position.expiry_date,
                )
                premium = self._latest_option_premium(
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
            position.peak_premium = max(float(position.peak_premium or position.entry_premium or position.entry_price), float(premium))
            position.unrealized_pnl = round((float(premium) - float(position.entry_premium or position.entry_price)) * int(position.quantity), 2)
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
                order_row = self._close_position(db, position=position, now=now, reason=exit_reason, exit_premium=float(premium))
                notifications.append((order_row, position))
                closed += 1
            updated += 1
        if updated or closed:
            db.commit()
        for order_row, position in notifications:
            self._notify_order(order_row, position)
        return {"updated_positions": updated, "closed_positions": closed}

    def _evaluate_symbol(self, db: Session, now: datetime, symbol: str) -> str:
        interval = "1minute"
        candle_ts = self._latest_candle_ts(db, symbol=symbol, interval=interval)
        if candle_ts is None:
            return "skip:no_candle"
        if self._last_entry_candle.get(symbol) == candle_ts:
            return "skip:duplicate_candle"
        self._last_entry_candle[symbol] = candle_ts

        consensus = get_consensus_signal(
            db,
            symbol=symbol,
            interval=interval,
            now=now,
            settings=self.settings,
            persist=False,
        )
        consensus.details["candle_ts"] = candle_ts.isoformat()
        log_row = log_consensus_result(db, consensus)

        if (
            str(self.settings.execution_mode).lower() == "live"
            and normalize_symbol_key(symbol) in set(getattr(self.settings, "live_execution_blocked_symbol_list", []))
        ):
            log_row.skip_reason = "Live execution blocked for this symbol. Keep it analytics-only until an exchange-supported option contract is configured."
            db.commit()
            return "skip:live_execution_blocked"

        if self._open_positions(db, symbol=symbol):
            log_row.skip_reason = "Open position already active for symbol"
            db.commit()
            return "skip:open_position_active"

        if consensus.consensus not in {"BUY", "SELL"}:
            db.commit()
            return f"skip:{consensus.skip_reason or 'non_trade_signal'}"

        options_payload = options_signal(
            symbol=symbol,
            interval=interval,
            prediction_mode="standard",
            expiry_date=None,
            strike_mode="auto",
            strategy_mode="auto",
            manual_strike=None,
            allow_option_writing=False,
            db=db,
        )
        chain_rows = [row.model_dump(mode="json") for row in options_payload.chain]
        pick = select_option_contract(
            signal_action=consensus.consensus,  # type: ignore[arg-type]
            spot_price=float(options_payload.underlying_price),
            strike_step=int(options_payload.strike_step),
            chain_rows=chain_rows,
            confidence=float(consensus.ml_confidence),
            expected_return_pct=float(options_payload.underlying_expected_return_pct or 0.0),
            premium_min=float(self.settings.execution_premium_min),
            premium_max=float(self.settings.execution_premium_max),
            days_to_expiry=(options_payload.expiry_date - now.date()).days,
            capital_per_trade=float(self.settings.execution_capital) * float(self.settings.execution_per_trade_risk_pct),
        )
        if pick is None:
            log_row.skip_reason = "No liquid strike passed the selection rules"
            db.commit()
            return "skip:no_liquid_strike"

        risk_plan = build_risk_plan(
            entry_premium=float(pick.premium),
            tsl_activation_percent=float(self.settings.tsl_activation_percent),
            target_profit_percent=float(self.settings.target_profit_percent),
        )
        if float(consensus.ai_score) > 55.0:
            risk_amount = float(pick.premium) - float(risk_plan.initial_sl)
            risk_plan.target_price = round(float(pick.premium) + (2.5 * risk_amount), 2)
        sizing = compute_quantity(
            capital=float(self.settings.execution_capital),
            capital_per_trade_pct=float(self.settings.execution_per_trade_risk_pct),
            entry_price=float(pick.premium),
            lot_size=lot_size_for_symbol(symbol),
            fixed_lots=max(1, int(getattr(self.settings, "execution_lot_size", 1) or 1)),
            vix_level=consensus.details.get("vix_level"),
        )
        request = BrokerOrderRequest(
            symbol=str(pick.instrument_key or symbol),
            option_type=str(pick.option_type),
            strike=float(pick.strike),
            expiry_date=options_payload.expiry_date.isoformat(),
            side="BUY",
            qty=int(sizing.qty),
            order_type="MARKET",
            tag="consensus_entry",
        )
        response = self.broker.place_order(request)
        if not response.success:
            log_row.skip_reason = f"Broker rejected order: {response.message}"
            db.commit()
            return "skip:broker_reject"

        position = ExecutionPosition(
            trade_date=now.date(),
            symbol=symbol,
            interval=interval,
            strategy_name="consensus_ce_pe",
            option_type=str(pick.option_type),
            side="BUY",
            expiry_date=options_payload.expiry_date,
            strike=float(pick.strike),
            quantity=int(sizing.qty),
            status="OPEN",
            entry_price=float(pick.premium),
            entry_premium=float(pick.premium),
            stop_loss=float(risk_plan.initial_sl),
            initial_sl=float(risk_plan.initial_sl),
            current_sl=float(risk_plan.current_sl),
            trailing_stop=float(risk_plan.current_sl),
            peak_premium=float(risk_plan.peak_price),
            tsl_active=bool(risk_plan.tsl_active),
            take_profit=float(risk_plan.target_price),
            target_premium=float(risk_plan.target_price),
            current_price=float(pick.premium),
            current_premium=float(pick.premium),
            pnl_points=0.0,
            pnl_value=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            ml_confidence=float(consensus.ml_confidence),
            ai_score=float(consensus.ai_score),
            pine_signal=str(consensus.pine_signal),
            consensus_reason=str(consensus.skip_reason or f"Combined score {consensus.combined_score:.2f}"),
            entry_order_id=getattr(response, "order_id", None),
            opened_at=now,
            metadata_json={
                "instrument_key": pick.instrument_key,
                "combined_score": consensus.combined_score,
                "strike_selection_reason": pick.reason,
                "spread_rupees": pick.spread_rupees,
                "premium_history": [
                    {
                        "timestamp": now.isoformat(),
                        "premium": round(float(pick.premium), 2),
                        "current_sl": round(float(risk_plan.current_sl), 2),
                        "tsl_active": False,
                        "unrealized_pnl": 0.0,
                    }
                ],
                "ml_reasons": consensus.ml_reasons,
                "ai_reasons": consensus.ai_reasons,
                "news_sentiment": consensus.news_sentiment,
                "signal_log_id": log_row.id,
                "vix_level": consensus.details.get("vix_level"),
                "liquidity_oi": pick.oi,
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
            ml_confidence=position.ml_confidence,
            ai_score=position.ai_score,
            pine_signal=position.pine_signal,
            consensus_reason=f"Combined score {consensus.combined_score:.2f} | {pick.reason}",
        )
        log_row.trade_placed = True
        log_row.skip_reason = None
        log_row.details = {
            **(log_row.details or {}),
            "selected_strike": position.strike,
            "option_type": position.option_type,
            "expiry_date": position.expiry_date.isoformat(),
            "quantity": position.quantity,
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
        premium = self._latest_option_premium(
            db,
            symbol=position.symbol,
            expiry_date=position.expiry_date,
            strike=float(position.strike),
            option_type=str(position.option_type),
        )
        exit_premium = float(premium or position.current_premium or position.entry_premium or position.entry_price)
        order_row = self._close_position(db, position=position, now=now, reason="MANUAL", exit_premium=exit_premium)
        db.commit()
        self._notify_order(order_row, position)
        return {"status": "closed", "position_id": position_id, "exit_premium": exit_premium}

    def execute_external_signal(
        self,
        db: Session,
        *,
        symbol: str,
        signal_action: str,
        confidence: float = 0.7,
        source: str = "pine",
        now: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = now or _now_ist()
        row = ExecutionExternalSignal(
            source=str(source),
            symbol=str(symbol),
            interval="1minute",
            signal_action=str(signal_action).upper(),
            signal_ts=now,
            confidence=float(confidence),
            processed=False,
            metadata_json=metadata or {},
        )
        db.add(row)
        db.commit()
        return {"status": "accepted", "symbol": symbol, "signal_action": row.signal_action, "source": source}

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

        signal_count = db.scalar(select(func.count()).select_from(ExecutionOrder).where(ExecutionOrder.trade_date == trade_date)) or 0
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
