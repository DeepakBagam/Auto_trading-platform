from utils.symbols import canonical_symbol_name, sort_display_symbols, symbol_aliases


def test_bank_nifty_aliases_are_normalized() -> None:
    aliases = symbol_aliases("Bank Nifty")
    assert "Nifty Bank" in aliases
    assert canonical_symbol_name("Nifty Bank") == "Bank Nifty"


def test_display_symbols_are_sorted_for_dashboard() -> None:
    symbols = sort_display_symbols(["India VIX", "SENSEX", "Bank Nifty", "Nifty 50"])
    assert symbols == ["Nifty 50", "Bank Nifty", "SENSEX", "India VIX"]
