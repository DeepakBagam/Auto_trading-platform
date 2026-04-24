from __future__ import annotations

from dataclasses import dataclass


BUY = "BUY"
SELL = "SELL"


@dataclass(frozen=True, slots=True)
class ExecutionConstraints:
    stop_atr_multiplier: float = 1.10
    stop_points_min: float = 42.0
    stop_points_max: float = 42.0
    partial_exit_points: float = 50.0
    partial_exit_fraction: float = 0.30
    breakeven_trigger_points: float = 35.0
    lock_trigger_points: float = 50.0
    lock_points: float = 25.0
    trail_trigger_points: float = 50.0
    trail_distance_points: float = 20.0
    runner_target_rr: float = 2.0
    min_adx: float = 24.0
    strong_adx: float = 25.0
    min_ema_separation_atr: float = 0.35
    min_ema_separation_pct: float = 0.0005
    no_momentum_minutes: int = 75
    max_hold_minutes: int = 90


DEFAULT_EXECUTION_CONSTRAINTS = ExecutionConstraints()


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def direction_multiplier(action: str) -> int:
    return 1 if str(action).upper() == BUY else -1


def move_points(action: str, entry_price: float, current_price: float) -> float:
    mult = direction_multiplier(action)
    return round((float(current_price) - float(entry_price)) * mult, 2)


def ema_separation_floor(
    *,
    atr: float,
    close: float,
    constraints: ExecutionConstraints = DEFAULT_EXECUTION_CONSTRAINTS,
) -> float:
    return max(
        float(atr) * float(constraints.min_ema_separation_atr),
        float(close) * float(constraints.min_ema_separation_pct),
    )


def ema_separation_is_valid(
    *,
    ema_21: float,
    ema_50: float,
    atr: float,
    close: float,
    constraints: ExecutionConstraints = DEFAULT_EXECUTION_CONSTRAINTS,
) -> bool:
    return abs(float(ema_21) - float(ema_50)) >= ema_separation_floor(
        atr=atr,
        close=close,
        constraints=constraints,
    )


def adaptive_stop_points(
    *,
    atr: float,
    constraints: ExecutionConstraints = DEFAULT_EXECUTION_CONSTRAINTS,
) -> float:
    raw = float(atr) * float(constraints.stop_atr_multiplier)
    return round(
        clip(raw, constraints.stop_points_min, constraints.stop_points_max),
        2,
    )


def runner_target_points(
    *,
    stop_points: float,
    constraints: ExecutionConstraints = DEFAULT_EXECUTION_CONSTRAINTS,
) -> float:
    raw = max(
        float(stop_points) * float(constraints.runner_target_rr),
        float(constraints.partial_exit_points) + float(constraints.lock_points),
    )
    return round(raw, 2)


def structured_stop_price(
    *,
    action: str,
    entry_price: float,
    initial_stop_price: float,
    current_stop_price: float,
    best_price: float,
    constraints: ExecutionConstraints = DEFAULT_EXECUTION_CONSTRAINTS,
) -> float:
    entry = float(entry_price)
    initial_stop = float(initial_stop_price)
    current_stop = float(current_stop_price)
    best = float(best_price)
    favorable = move_points(action, entry, best)

    if favorable >= constraints.breakeven_trigger_points:
        current_stop = max(current_stop, entry) if action == BUY else min(current_stop, entry)
    if favorable >= constraints.lock_trigger_points:
        locked = entry + (direction_multiplier(action) * float(constraints.lock_points))
        current_stop = max(current_stop, locked) if action == BUY else min(current_stop, locked)
    if favorable >= constraints.trail_trigger_points:
        trailed = best - float(constraints.trail_distance_points) if action == BUY else best + float(constraints.trail_distance_points)
        current_stop = max(current_stop, trailed) if action == BUY else min(current_stop, trailed)

    if action == BUY:
        return round(max(initial_stop, current_stop), 2)
    return round(min(initial_stop, current_stop), 2)


def time_exit_reason(
    *,
    elapsed_minutes: int,
    mfe_points: float,
    current_points: float,
    partial_taken: bool,
    stop_points: float,
    constraints: ExecutionConstraints = DEFAULT_EXECUTION_CONSTRAINTS,
) -> str | None:
    elapsed = int(elapsed_minutes)
    mfe = float(mfe_points)
    current = float(current_points)
    stop = float(stop_points)

    if not partial_taken and elapsed >= 60 and mfe < constraints.breakeven_trigger_points:
        return "TIME_EXIT_NO_MOMENTUM"
    if elapsed >= constraints.no_momentum_minutes and current < constraints.lock_points and mfe < constraints.partial_exit_points:
        return "TIME_EXIT_STALL"
    if elapsed >= constraints.max_hold_minutes and current < max(constraints.partial_exit_points, stop * 1.25):
        return "TIME_EXIT_MAX_HOLD"
    return None
