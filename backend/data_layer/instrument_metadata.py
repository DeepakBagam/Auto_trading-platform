from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from threading import Lock
from typing import Any

from backend.data_layer.collectors.upstox_option_chain import UpstoxOptionChainCollector
from backend.utils.config import Settings
from backend.utils.constants import IST_ZONE


@dataclass(frozen=True, slots=True)
class OptionContractMetadata:
    instrument_key: str
    lot_size: int
    minimum_lot: int
    tick_size: float
    freeze_quantity: int | None
    fetched_at: datetime


_CACHE_TTL = timedelta(minutes=15)
_CACHE: dict[tuple[str, date], tuple[datetime, dict[str, OptionContractMetadata]]] = {}
_CACHE_LOCK = Lock()


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalize_tick_size(value: Any) -> float | None:
    try:
        tick = Decimal(str(value))
    except Exception:
        return None
    if tick <= 0:
        return None
    # Upstox JSON instrument metadata may express paise as 5 for a Rs 0.05 tick.
    if tick >= 1:
        tick = tick / Decimal("100")
    return float(tick)


def _contract_map(rows: list[dict[str, Any]], *, fetched_at: datetime) -> dict[str, OptionContractMetadata]:
    out: dict[str, OptionContractMetadata] = {}
    for row in rows:
        instrument_key = str(row.get("instrument_key") or row.get("instrument_token") or "").strip()
        lot_size = _positive_int(row.get("lot_size"))
        minimum_lot = _positive_int(row.get("minimum_lot")) or lot_size
        tick_size = _normalize_tick_size(row.get("tick_size"))
        if not instrument_key or lot_size is None or minimum_lot is None or tick_size is None:
            continue
        if instrument_key in out:
            continue
        out[instrument_key] = OptionContractMetadata(
            instrument_key=instrument_key,
            lot_size=lot_size,
            minimum_lot=minimum_lot,
            tick_size=tick_size,
            freeze_quantity=_positive_int(row.get("freeze_quantity")),
            fetched_at=fetched_at,
        )
    return out


def resolve_option_contract_metadata(
    *,
    settings: Settings,
    underlying_key: str,
    expiry_date: date,
    instrument_key: str,
) -> OptionContractMetadata | None:
    cache_key = (underlying_key, expiry_date)
    now = datetime.now(IST_ZONE)
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None and now - cached[0] <= _CACHE_TTL:
            return cached[1].get(instrument_key)

    rows = UpstoxOptionChainCollector(settings=settings).fetch_option_contracts(
        underlying_key,
        expiry_date,
    )
    contracts = _contract_map(rows, fetched_at=now)
    with _CACHE_LOCK:
        _CACHE[cache_key] = (now, contracts)
    return contracts.get(instrument_key)
