import pytest

from utils.intervals import normalize_interval


@pytest.mark.parametrize(
    ("raw_interval", "expected"),
    [
        ("1m", "1minute"),
        ("1min", "1minute"),
        ("1minute", "1minute"),
    ],
)
def test_normalize_interval_accepts_api_aliases(raw_interval: str, expected: str) -> None:
    assert normalize_interval(raw_interval) == expected


def test_normalize_interval_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        normalize_interval("30minute")
