import pytest

from backend.utils.intervals import normalize_interval


@pytest.mark.parametrize(
    ("raw_interval", "expected"),
    [
        ("1m", "1minute"),
        ("1min", "1minute"),
        ("1minute", "1minute"),
        ("2m", "2minute"),
        ("75 minutes", "75minute"),
        ("300m", "300minute"),
        ("1h", "1hour"),
        ("5 hours", "5hour"),
        ("1d", "day"),
        ("1w", "week"),
        ("1mo", "month"),
    ],
)
def test_normalize_interval_accepts_api_aliases(raw_interval: str, expected: str) -> None:
    assert normalize_interval(raw_interval) == expected


def test_normalize_interval_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        normalize_interval("301minute")
