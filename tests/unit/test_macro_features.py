import pandas as pd

from feature_engine.macro_features import build_macro_features


def _make_target_frame(symbol: str, start: str = "2026-01-01", periods: int = 30) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    base = 100.0 if symbol == "India VIX" else 50000.0
    return pd.DataFrame(
        {
            "ts": dates,
            "open": [base + i * 10 for i in range(periods)],
            "high": [base + i * 10 + 20 for i in range(periods)],
            "low": [base + i * 10 - 20 for i in range(periods)],
            "close": [base + i * 10 + 5 for i in range(periods)],
            "volume": [1000 + i * 10 for i in range(periods)],
            "ret_1d": pd.Series([base + i * 10 + 5 for i in range(periods)]).pct_change().fillna(0.0),
            "ret_5d": pd.Series([base + i * 10 + 5 for i in range(periods)]).pct_change(5).fillna(0.0),
        }
    )


def test_build_macro_features_adds_cross_asset_context():
    target = _make_target_frame("SENSEX")
    market_df = pd.concat(
        [
            _make_target_frame("SENSEX").assign(symbol="SENSEX"),
            _make_target_frame("Nifty 50").assign(symbol="Nifty 50"),
            _make_target_frame("India VIX").assign(symbol="India VIX"),
        ],
        ignore_index=True,
    )

    out = build_macro_features(target, symbol="SENSEX", market_df=market_df)

    assert "peer_ret_1d" in out.columns
    assert "india_vix_ret_1d" in out.columns
    assert "rel_strength_1d" in out.columns
    assert "peer_corr_20" in out.columns
    assert float(out["peer_ret_1d"].abs().sum()) > 0.0
    assert float(out["india_vix_close_level"].abs().sum()) > 0.0
    assert out["risk_off_score"].notna().all()


def test_build_macro_features_falls_back_to_zero_without_market_data():
    target = _make_target_frame("SENSEX")

    out = build_macro_features(target, symbol="SENSEX", market_df=pd.DataFrame())

    for col in [
        "peer_ret_1d",
        "india_vix_ret_1d",
        "peer_corr_20",
        "peer_beta_20",
    ]:
        assert col in out.columns
        assert float(out[col].abs().sum()) == 0.0
    assert "rel_strength_1d" in out.columns
    assert out["rel_strength_1d"].equals(target["ret_1d"])
