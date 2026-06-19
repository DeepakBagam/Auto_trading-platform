from backend.execution_engine.risk_manager import compute_quantity, update_risk_plan


def test_update_risk_plan_keeps_fixed_sl_before_tsl_activation() -> None:
    update = update_risk_plan(
        entry_price=100.0,
        current_price=102.0,
        initial_sl=90.0,
        current_sl=90.0,
        peak_price=100.0,
        tsl_active=False,
        target_price=130.0,
        tsl_activation_percent=0.05,
        tsl_trail_percent=0.03,
    )

    assert update.tsl_active is False
    assert update.current_sl == 90.0
    assert update.trailing_sl is None


def test_update_risk_plan_activates_and_trails_after_profit_threshold() -> None:
    update = update_risk_plan(
        entry_price=100.0,
        current_price=110.0,
        initial_sl=90.0,
        current_sl=90.0,
        peak_price=106.0,
        tsl_active=False,
        target_price=140.0,
        tsl_activation_percent=0.05,
        tsl_trail_percent=0.03,
    )

    assert update.tsl_active is True
    assert update.trailing_sl == 106.7
    assert update.current_sl == 106.7


def test_compute_quantity_uses_stop_distance_when_available() -> None:
    result = compute_quantity(
        capital=100000.0,
        capital_per_trade_pct=0.02,
        entry_price=100.0,
        stop_loss_price=90.0,
        lot_size=50,
    )

    assert result.lots == 4
    assert result.qty == 200
    assert result.capital_allocated == 20000.0
    assert result.risk_budget == 2000.0
    assert result.risk_per_lot == 500.0
    assert result.estimated_risk == 2000.0
    assert result.reason == "sized"


def test_compute_quantity_rejects_invalid_stop_loss() -> None:
    result = compute_quantity(
        capital=100000.0,
        capital_per_trade_pct=0.02,
        entry_price=100.0,
        stop_loss_price=100.0,
        lot_size=50,
    )

    assert result.lots == 0
    assert result.qty == 0
    assert result.reason == "invalid_stop_loss"


def test_compute_quantity_returns_zero_when_one_lot_exceeds_risk_budget() -> None:
    result = compute_quantity(
        capital=10000.0,
        capital_per_trade_pct=0.01,
        entry_price=100.0,
        stop_loss_price=90.0,
        lot_size=50,
        fixed_lots=4,
    )

    assert result.risk_budget == 100.0
    assert result.risk_per_lot == 500.0
    assert result.lots == 0
    assert result.qty == 0
    assert result.reason == "risk_budget_below_minimum_lot"


def test_fixed_lots_is_a_cap_not_a_risk_override() -> None:
    result = compute_quantity(
        capital=100000.0,
        capital_per_trade_pct=0.01,
        entry_price=100.0,
        stop_loss_price=90.0,
        lot_size=50,
        fixed_lots=4,
        max_lots=3,
    )

    assert result.risk_limited_lots == 2
    assert result.affordable_lots == 20
    assert result.lots == 2
    assert result.qty == 100
    assert result.estimated_risk == 1000.0


def test_compute_quantity_enforces_affordability_and_max_lots() -> None:
    result = compute_quantity(
        capital=12000.0,
        capital_per_trade_pct=0.50,
        entry_price=100.0,
        stop_loss_price=99.0,
        lot_size=50,
        fixed_lots=10,
        max_lots=5,
    )

    assert result.affordable_lots == 2
    assert result.risk_limited_lots == 120
    assert result.lots == 2
    assert result.capital_allocated == 10000.0
