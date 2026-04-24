from execution_engine.intraday_rules import (
    DEFAULT_EXECUTION_CONSTRAINTS,
    adaptive_stop_points,
    move_points,
    structured_stop_price,
    time_exit_reason,
)


def test_adaptive_stop_points_clips_into_target_band() -> None:
    assert adaptive_stop_points(atr=20.0) == 42.0
    assert adaptive_stop_points(atr=60.0) == 42.0


def test_structured_stop_price_advances_from_breakeven_to_locked_profit() -> None:
    constraints = DEFAULT_EXECUTION_CONSTRAINTS
    initial_sl = 23958.0
    entry = 24000.0

    breakeven = structured_stop_price(
        action="BUY",
        entry_price=entry,
        initial_stop_price=initial_sl,
        current_stop_price=initial_sl,
        best_price=24036.0,
        constraints=constraints,
    )
    assert breakeven == entry

    locked = structured_stop_price(
        action="BUY",
        entry_price=entry,
        initial_stop_price=initial_sl,
        current_stop_price=breakeven,
        best_price=24075.0,
        constraints=constraints,
    )
    assert locked == 24055.0


def test_time_exit_reason_enforces_max_hold_when_runner_stalls() -> None:
    reason = time_exit_reason(
        elapsed_minutes=91,
        mfe_points=48.0,
        current_points=44.0,
        partial_taken=False,
        stop_points=42.0,
    )
    assert reason == "TIME_EXIT_MAX_HOLD"


def test_move_points_handles_long_and_short_directions() -> None:
    assert move_points("BUY", 100.0, 112.5) == 12.5
    assert move_points("SELL", 100.0, 87.5) == 12.5
