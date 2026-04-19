from prediction_engine.confidence_engine import (
    compute_confidence_v3,
    data_freshness_score,
    interval_quality_score,
)


def test_interval_quality_score_bounds():
    s = interval_quality_score(coverage=0.8, width_pct=0.02)
    assert 0.0 <= s <= 1.0


def test_data_freshness_score_decreases():
    assert data_freshness_score(1) >= data_freshness_score(30)
    assert data_freshness_score(30) >= data_freshness_score(200)


def test_compute_confidence_v3_shape():
    out = compute_confidence_v3(
        calibrated_direction_prob=0.7,
        interval_coverage=0.81,
        interval_width_pct=0.03,
        staleness_minutes=2,
    )
    assert 0.0 <= out["confidence_score"] <= 1.0
    assert out["confidence_bucket"] in {"high", "medium", "low"}
    assert "components" in out
