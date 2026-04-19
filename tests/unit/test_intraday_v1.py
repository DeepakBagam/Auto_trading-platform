import pandas as pd

from models.intraday_v1 import (
    _build_seasonal_profile,
    _clamp_close_by_volatility,
    _session_slot_index,
    _stabilize_return,
)


def test_stabilize_return_decays_towards_mean_for_long_horizon() -> None:
    early = _stabilize_return(raw_ret=-0.004, step=1, mu=0.0, sigma=0.001, interval="1minute")
    late = _stabilize_return(raw_ret=-0.004, step=180, mu=0.0, sigma=0.001, interval="1minute")
    assert early < 0
    assert abs(late) < abs(early)


def test_clamp_close_by_volatility_limits_far_outlier() -> None:
    # 30% jump at step=1 should be clamped heavily for low observed sigma.
    clamped = _clamp_close_by_volatility(
        close_candidate=130.0,
        anchor_close=100.0,
        step=1,
        sigma_close=0.002,
    )
    assert 99.0 <= clamped <= 101.0


def test_session_slot_index_aligns_expected_slots() -> None:
    assert _session_slot_index(pd.Timestamp("2026-04-07 09:15:00+05:30").to_pydatetime(), "1minute") == 0
    assert _session_slot_index(pd.Timestamp("2026-04-07 09:16:00+05:30").to_pydatetime(), "1minute") == 1
    assert _session_slot_index(pd.Timestamp("2026-04-07 09:45:00+05:30").to_pydatetime(), "30minute") == 1


def test_build_seasonal_profile_captures_slot_volatility() -> None:
    ts = pd.to_datetime(
        [
            "2026-04-06 09:15:00+05:30",
            "2026-04-06 09:16:00+05:30",
            "2026-04-06 09:17:00+05:30",
            "2026-04-07 09:15:00+05:30",
            "2026-04-07 09:16:00+05:30",
            "2026-04-07 09:17:00+05:30",
        ]
    )
    df = pd.DataFrame(
        {
            "ts": ts,
            "open": [100.0, 101.0, 99.5, 102.0, 100.5, 101.5],
            "high": [101.5, 102.5, 100.5, 103.0, 101.8, 102.4],
            "low": [99.8, 100.2, 98.9, 101.2, 99.9, 100.7],
            "close": [101.0, 99.8, 100.4, 100.8, 101.7, 100.9],
            "volume": [10, 11, 12, 13, 14, 15],
        }
    )
    profile, floor = _build_seasonal_profile(df, "1minute")
    assert set(profile.keys()) == {0, 1, 2}
    assert floor > 0.0
    assert profile[1]["abs_close_ret"] > 0.0
    assert profile[1]["range_pct"] >= profile[1]["body_abs_pct"]
