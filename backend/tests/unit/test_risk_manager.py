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
    assert result.capital_allocated == 2000.0


def test_compute_quantity_falls_back_to_premium_when_stop_is_invalid() -> None:
    result = compute_quantity(
        capital=100000.0,
        capital_per_trade_pct=0.02,
        entry_price=100.0,
        stop_loss_price=100.0,
        lot_size=50,
    )

    assert result.lots == 1
    assert result.qty == 50
