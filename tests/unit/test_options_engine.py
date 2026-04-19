from datetime import date, timedelta

from prediction_engine.options_engine import (
    build_chain_rows,
    build_option_signal,
    strike_step_for_symbol,
    synthetic_option_chain,
)


def test_synthetic_option_chain_builds_ce_pe_quotes() -> None:
    expiry = date.today() + timedelta(days=7)
    quotes = synthetic_option_chain(
        symbol="Nifty 50",
        underlying_price=24050.0,
        expiry_date=expiry,
        strike_step=50,
        levels=3,
    )
    assert len(quotes) == 14
    assert any(q.option_type == "CE" for q in quotes)
    assert any(q.option_type == "PE" for q in quotes)


def test_build_option_signal_maps_bullish_to_buy_ce() -> None:
    expiry = date.today() + timedelta(days=7)
    strike_step = strike_step_for_symbol("Nifty 50")
    quotes = synthetic_option_chain(
        symbol="Nifty 50",
        underlying_price=24120.0,
        expiry_date=expiry,
        strike_step=strike_step,
        levels=4,
    )
    chain = build_chain_rows(quotes)
    out = build_option_signal(
        symbol="Nifty 50",
        interval="1minute",
        expiry_date=expiry,
        underlying_price=24120.0,
        underlying_signal_action="BUY",
        underlying_conviction="medium",
        underlying_confidence=0.74,
        underlying_expected_return_pct=0.013,
        chain_rows=chain,
        strike_step=strike_step,
        strike_mode="auto",
        manual_strike=None,
        allow_option_writing=False,
    )
    signal = out["option_signal"]
    assert signal["action"] == "BUY"
    assert signal["option_type"] == "CE"
    assert signal["strike"] is not None
    assert signal["entry_price"] is not None
    assert signal["stop_loss"] is not None
    assert signal["trailing_stop_loss"] is not None


def test_build_option_signal_maps_hold_to_no_trade() -> None:
    expiry = date.today() + timedelta(days=7)
    quotes = synthetic_option_chain(
        symbol="Nifty 50",
        underlying_price=24000.0,
        expiry_date=expiry,
        strike_step=50,
        levels=2,
    )
    chain = build_chain_rows(quotes)
    out = build_option_signal(
        symbol="Nifty 50",
        interval="day",
        expiry_date=expiry,
        underlying_price=24000.0,
        underlying_signal_action="HOLD",
        underlying_conviction="low",
        underlying_confidence=0.45,
        underlying_expected_return_pct=0.0,
        chain_rows=chain,
        strike_step=50,
        strike_mode="auto",
        manual_strike=None,
        allow_option_writing=False,
    )
    signal = out["option_signal"]
    assert signal["action"] == "HOLD"
    assert signal["entry_price"] is None


def test_build_option_signal_auto_sideways_selects_iron_condor_when_writing_enabled() -> None:
    expiry = date.today() + timedelta(days=7)
    quotes = synthetic_option_chain(
        symbol="Nifty 50",
        underlying_price=24100.0,
        expiry_date=expiry,
        strike_step=50,
        levels=4,
    )
    chain = build_chain_rows(quotes)
    out = build_option_signal(
        symbol="Nifty 50",
        interval="1minute",
        expiry_date=expiry,
        underlying_price=24100.0,
        underlying_signal_action="HOLD",
        underlying_conviction="medium",
        underlying_confidence=0.62,
        underlying_expected_return_pct=0.0004,
        chain_rows=chain,
        strike_step=50,
        strike_mode="auto",
        manual_strike=None,
        allow_option_writing=True,
        strategy_mode="auto",
        technical_context={},
    )
    signal = out["option_signal"]
    assert signal["strategy"] == "iron_condor"
    assert signal["action"] == "SELL"
    assert len(signal["legs"]) == 4


def test_build_option_signal_long_straddle_builds_two_leg_debit_trade() -> None:
    expiry = date.today() + timedelta(days=7)
    quotes = synthetic_option_chain(
        symbol="Nifty 50",
        underlying_price=24000.0,
        expiry_date=expiry,
        strike_step=50,
        levels=4,
    )
    chain = build_chain_rows(quotes)
    out = build_option_signal(
        symbol="Nifty 50",
        interval="30minute",
        expiry_date=expiry,
        underlying_price=24000.0,
        underlying_signal_action="HOLD",
        underlying_conviction="medium",
        underlying_confidence=0.70,
        underlying_expected_return_pct=0.012,
        chain_rows=chain,
        strike_step=50,
        strike_mode="auto",
        manual_strike=None,
        allow_option_writing=False,
        strategy_mode="long_straddle",
        technical_context={},
    )
    signal = out["option_signal"]
    assert signal["strategy"] == "long_straddle"
    assert signal["action"] == "BUY"
    assert len(signal["legs"]) == 2
    assert signal["entry_price"] is not None
