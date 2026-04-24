from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(slots=True)
class TradeSizingResult:
    qty: int
    lots: int
    capital_allocated: float
    vix_multiplier: float
    entry_premium: float


@dataclass(slots=True)
class RiskPlan:
    initial_sl: float
    current_sl: float
    trailing_sl: float | None
    target_price: float
    peak_price: float
    tsl_active: bool
    activation_price: float


@dataclass(slots=True)
class RiskUpdate:
    current_sl: float
    trailing_sl: float | None
    peak_price: float
    tsl_active: bool
    target_hit: bool
    exit_triggered: bool
    exit_reason: str | None


def vix_position_multiplier(vix_level: float | None) -> float:
    if vix_level is None:
        return 1.0
    vix = float(vix_level)
    if vix < 13:
        return 1.0
    if vix < 17:
        return 0.8
    if vix < 22:
        return 0.6
    return 0.4


def initial_stop_loss(entry_premium: float) -> float:
    entry = float(entry_premium)
    if entry < 50:
        return round(entry - 10.0, 2)
    if entry <= 100:
        return round(entry * 0.92, 2)
    if entry <= 200:
        return round(entry * 0.93, 2)
    return round(entry * 0.94, 2)


def build_risk_plan(
    *,
    entry_premium: float,
    tsl_activation_percent: float = 0.05,
    target_profit_percent: float = 0.30,
) -> RiskPlan:
    """Build risk plan with strict 2:1 risk-reward."""
    entry = float(entry_premium)
    initial_sl = initial_stop_loss(entry)
    risk_amount = entry - initial_sl
    target_price = round(entry + (2.0 * risk_amount), 2)
    min_target = round(entry * (1.0 + float(target_profit_percent)), 2)
    target_price = max(target_price, min_target)
    activation_price = round(entry * (1.0 + float(tsl_activation_percent)), 2)
    return RiskPlan(
        initial_sl=initial_sl,
        current_sl=initial_sl,
        trailing_sl=None,
        target_price=target_price,
        peak_price=entry,
        tsl_active=False,
        activation_price=activation_price,
    )


def compute_quantity(
    *,
    capital: float,
    capital_per_trade_pct: float | None = None,
    entry_price: float,
    lot_size: int,
    vix_level: float | None = None,
    min_lots: int = 1,
    max_lots: int | None = None,
    fixed_lots: int | None = None,
    per_trade_risk_pct: float | None = None,
    stop_loss_price: float | None = None,
) -> TradeSizingResult:
    del stop_loss_price
    lot_size = max(1, int(lot_size))
    min_lots = max(1, int(min_lots))
    if capital_per_trade_pct is None:
        capital_per_trade_pct = per_trade_risk_pct if per_trade_risk_pct is not None else 0.02
    multiplier = vix_position_multiplier(vix_level)
    price = max(0.01, float(entry_price))
    capital_allocated = max(0.0, float(capital) * max(0.0, float(capital_per_trade_pct)) * multiplier)
    if fixed_lots is not None:
        lots = max(min_lots, int(fixed_lots))
        if max_lots is not None:
            lots = min(lots, max(1, int(max_lots)))
        capital_allocated = round(price * lot_size * lots, 2)
    else:
        affordable_lots = floor(capital_allocated / (price * lot_size))
        lots = max(min_lots, affordable_lots)
        if max_lots is not None:
            lots = min(lots, max(1, int(max_lots)))
    qty = max(lot_size, lots * lot_size)
    return TradeSizingResult(
        qty=int(qty),
        lots=int(lots),
        capital_allocated=round(capital_allocated, 2),
        vix_multiplier=float(multiplier),
        entry_premium=float(entry_price),
    )


def update_risk_plan(
    *,
    entry_price: float,
    current_price: float,
    initial_sl: float,
    current_sl: float,
    peak_price: float,
    tsl_active: bool,
    target_price: float,
    tsl_activation_percent: float = 0.05,
    tsl_trail_percent: float = 0.03,
    tsl_immediate: bool = True,
) -> RiskUpdate:
    """Update risk plan with trailing stop logic.

    Two modes:

    tsl_immediate=True  (default / recommended):
        TSL is always active from the first tick.  The trail distance is the
        absolute rupee risk at entry (entry - initial_sl).
        Example: entry=100, SL=99 → trail=1.
          price→101: SL=100   price→103: SL=102
        SL only moves UP, never DOWN.

    tsl_immediate=False  (legacy percentage mode):
        TSL activates after `tsl_activation_percent` profit, then trails
        `tsl_trail_percent` below peak.  Also moves SL to breakeven at 1R.
    """
    entry = float(entry_price)
    price = float(current_price)
    peak = max(float(peak_price), price)
    active = bool(tsl_active)
    trailing_sl: float | None = None
    risk = max(entry - float(initial_sl), 1e-6)  # absolute trail amount

    if tsl_immediate:
        # --- Immediate percentage trail (configured via TSL_TRAIL_PERCENT in .env) ---
        # TSL is active from the very first tick — no waiting for a profit target.
        # Trail distance = peak × tsl_trail_percent  (scales with the premium price)
        active = True
        trailing_sl = round(peak * (1.0 - float(tsl_trail_percent)), 2)
        current_sl = round(max(float(current_sl), trailing_sl), 2)
    else:
        # --- Legacy percentage mode ---
        # Move SL to breakeven at 1R profit
        if price >= entry + risk and float(current_sl) < entry:
            current_sl = entry

        # Activate trailing stop at 1.5R profit
        if price >= entry + (1.5 * risk):
            active = True
            peak = max(peak, price)
            trailing_sl = round(peak * (1.0 - float(tsl_trail_percent)), 2)
            current_sl = round(max(float(current_sl), float(trailing_sl)), 2)
        elif active:
            peak = max(peak, price)
            trailing_sl = round(peak * (1.0 - float(tsl_trail_percent)), 2)
            current_sl = round(max(float(current_sl), float(trailing_sl)), 2)
        else:
            current_sl = round(max(float(initial_sl), float(current_sl)), 2)

    target_hit = price >= float(target_price)
    exit_triggered = False
    exit_reason: str | None = None
    if target_hit:
        exit_triggered = True
        exit_reason = "TP_HIT"
    elif price <= float(current_sl):
        exit_triggered = True
        if current_sl >= entry:
            exit_reason = "BREAKEVEN_HIT"
        elif active and current_sl > float(initial_sl):
            exit_reason = "TSL_HIT"
        else:
            exit_reason = "SL_HIT"

    return RiskUpdate(
        current_sl=float(current_sl),
        trailing_sl=(float(trailing_sl) if trailing_sl is not None else None),
        peak_price=float(peak),
        tsl_active=active,
        target_hit=target_hit,
        exit_triggered=exit_triggered,
        exit_reason=exit_reason,
    )


def update_trailing_stop(
    *,
    side: str,
    entry_metric: float,
    current_metric: float,
    existing_trailing_stop: float | None,
    hard_stop: float,
) -> float:
    if str(side).upper() == "SELL":
        if current_metric <= entry_metric * 0.95:
            candidate = round(float(current_metric) * 1.03, 2)
            base = existing_trailing_stop if existing_trailing_stop is not None else hard_stop
            return float(min(base, candidate))
        return float(existing_trailing_stop if existing_trailing_stop is not None else hard_stop)
    current_sl = existing_trailing_stop if existing_trailing_stop is not None else hard_stop
    update = update_risk_plan(
        entry_price=entry_metric,
        current_price=current_metric,
        initial_sl=hard_stop,
        current_sl=current_sl,
        peak_price=max(entry_metric, current_metric),
        tsl_active=(existing_trailing_stop is not None and existing_trailing_stop > hard_stop),
        target_price=float("inf"),
    )
    return float(update.current_sl)
