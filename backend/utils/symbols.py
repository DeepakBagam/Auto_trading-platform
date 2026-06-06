from __future__ import annotations

from sqlalchemy import or_


_SYMBOL_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "NIFTY50": ("Nifty 50", "NIFTY 50", "NIFTY50"),
    "INDIAVIX": ("India VIX", "INDIA VIX", "INDIAVIX"),
    "SENSEX": ("SENSEX",),
    "BANKNIFTY": ("Bank Nifty", "BANK NIFTY", "Nifty Bank", "NIFTY BANK", "BANKNIFTY", "NIFTYBANK"),
    "NIFTYBANK": ("Bank Nifty", "BANK NIFTY", "Nifty Bank", "NIFTY BANK", "BANKNIFTY", "NIFTYBANK"),
}

_PREFERRED_DISPLAY: dict[str, str] = {
    "NIFTY50": "Nifty 50",
    "INDIAVIX": "India VIX",
    "SENSEX": "SENSEX",
    "BANKNIFTY": "Bank Nifty",
    "NIFTYBANK": "Bank Nifty",
}

_DISPLAY_ORDER = {
    "Nifty 50": 0,
    "Bank Nifty": 1,
    "SENSEX": 2,
    "India VIX": 3,
}


def normalize_symbol_key(symbol: str) -> str:
    return "".join(ch for ch in str(symbol or "").upper() if ch.isalnum())


def symbol_aliases(symbol: str) -> list[str]:
    raw = str(symbol or "").strip()
    if not raw:
        return []
    normalized = normalize_symbol_key(raw)
    aliases = list(_SYMBOL_ALIAS_GROUPS.get(normalized, (raw,)))
    if raw not in aliases:
        aliases.insert(0, raw)
    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        key = alias.upper()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(alias)
    return deduped


def canonical_symbol_name(symbol: str) -> str:
    raw = str(symbol or "").strip()
    if not raw:
        return raw
    return _PREFERRED_DISPLAY.get(normalize_symbol_key(raw), raw)


def display_symbol_from_instrument_key(instrument_key: str) -> str:
    raw = str(instrument_key).split("|", 1)[1] if "|" in str(instrument_key) else str(instrument_key)
    return canonical_symbol_name(raw)


def instrument_key_filter(column, symbol: str):
    aliases = symbol_aliases(symbol)
    return or_(*([column == alias for alias in aliases] + [column.like(f"%|{alias}") for alias in aliases]))


def symbol_value_filter(column, symbol: str):
    return column.in_(symbol_aliases(symbol))


def sort_display_symbols(symbols: list[str]) -> list[str]:
    normalized = {canonical_symbol_name(symbol) for symbol in symbols if str(symbol or "").strip()}
    return sorted(normalized, key=lambda value: (_DISPLAY_ORDER.get(value, 99), value))
