from datetime import datetime

from backend.execution_engine.live_service import (
    TechnicalSignal,
    _option_liquidity_failures,
    _option_risk_plan,
)
from backend.utils.config import Settings
from backend.utils.constants import IST_ZONE


def _signal(atr: float) -> TechnicalSignal:
    return TechnicalSignal(
        symbol="Nifty 50",
        interval="1minute",
        timestamp=datetime(2026, 6, 9, 10, 0, tzinfo=IST_ZONE),
        action="BUY",
        bias="BUY",
        score=80,
        confidence=0.8,
        conviction="medium",
        entry_price=23000,
        stop_loss=22950,
        take_profit=23100,
        cooldown_seconds=0,
        max_signals_reached=False,
        reasons=[],
        details={"atr_14": atr},
    )


def test_option_liquidity_rejects_low_volume_oi_and_wide_spread() -> None:
    settings = Settings(_env_file=None, option_min_volume=500, option_min_oi=1000, option_max_spread_pct=0.05)

    failures = _option_liquidity_failures(
        {
            "instrument_key": "NSE_FO|12345",
            "ltp": 100,
            "bid": 90,
            "ask": 110,
            "volume": 100,
            "oi": 250,
        },
        settings,
    )

    assert any("volume" in item for item in failures)
    assert any("OI" in item for item in failures)
    assert any("spread" in item for item in failures)


def test_atr_aware_option_risk_widens_in_volatile_market() -> None:
    settings = Settings(_env_file=None, enhanced_risk_enabled=True, atr_sl_multiplier=1.8, target_rr_ratio=2.0)

    calm = _option_risk_plan(100, _signal(atr=6), settings)
    volatile = _option_risk_plan(100, _signal(atr=25), settings)

    assert calm["stop_pct"] < volatile["stop_pct"]
    assert calm["stop_loss"] > volatile["stop_loss"]
    assert volatile["target_pct"] >= calm["target_pct"]
