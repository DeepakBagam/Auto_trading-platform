from datetime import datetime

from backend.data_layer.instrument_metadata import _contract_map, _normalize_tick_size
from backend.utils.constants import IST_ZONE


def test_upstox_json_tick_size_is_normalized_from_paise() -> None:
    assert _normalize_tick_size(5) == 0.05
    assert _normalize_tick_size(0.05) == 0.05


def test_contract_metadata_parses_current_lot_and_tick() -> None:
    fetched_at = datetime(2026, 6, 21, 12, 0, tzinfo=IST_ZONE)
    contracts = _contract_map(
        [
            {
                "instrument_key": "NSE_FO|123",
                "lot_size": 30,
                "minimum_lot": 30,
                "tick_size": 5,
                "freeze_quantity": 900,
            }
        ],
        fetched_at=fetched_at,
    )

    contract = contracts["NSE_FO|123"]
    assert contract.lot_size == 30
    assert contract.minimum_lot == 30
    assert contract.tick_size == 0.05
    assert contract.freeze_quantity == 900
