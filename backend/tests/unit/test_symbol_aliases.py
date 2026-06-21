from backend.utils.symbols import (
    canonical_symbol_name,
    is_option_execution_symbol,
    sort_display_symbols,
    symbol_aliases,
)


def test_bank_nifty_aliases_are_normalized() -> None:
    aliases = symbol_aliases("Bank Nifty")
    assert "Nifty Bank" in aliases
    assert canonical_symbol_name("Nifty Bank") == "Bank Nifty"


def test_display_symbols_are_sorted_for_dashboard() -> None:
    symbols = sort_display_symbols(["India VIX", "SENSEX", "Bank Nifty", "Nifty 50"])
    assert symbols == ["Nifty 50", "Bank Nifty", "SENSEX", "India VIX"]


def test_india_vix_is_market_context_not_option_execution_underlying() -> None:
    assert is_option_execution_symbol("India VIX") is False
    assert is_option_execution_symbol("Bank Nifty") is True
