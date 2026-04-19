from __future__ import annotations


def compute_confidence(direction_prob: float, metrics: dict | None = None) -> float:
    metrics = metrics or {}
    if "mae_close_pct" in metrics:
        mae_pct = float(metrics.get("mae_close_pct", 0.0))
        mae_penalty = 1.0 / (1.0 + mae_pct * 10.0)
    else:
        mae_close = float(metrics.get("mae_close", 0.0))
        mae_penalty = 1.0 / (1.0 + mae_close / 100.0)
    raw = float(direction_prob) * mae_penalty
    return max(0.0, min(1.0, raw))


def confidence_bucket(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


def interval_quality_score(coverage: float, width_pct: float, target_coverage: float = 0.80) -> float:
    cov = max(0.0, min(1.0, float(coverage)))
    width = max(0.0, float(width_pct))
    coverage_score = max(0.0, 1.0 - abs(cov - target_coverage) / max(target_coverage, 1e-9))
    width_score = 1.0 / (1.0 + width * 20.0)
    return max(0.0, min(1.0, 0.6 * coverage_score + 0.4 * width_score))


def data_freshness_score(staleness_minutes: float) -> float:
    lag = max(0.0, float(staleness_minutes))
    if lag <= 2:
        return 1.0
    if lag <= 10:
        return 0.85
    if lag <= 30:
        return 0.65
    if lag <= 120:
        return 0.35
    return 0.10


def compute_confidence_v3(
    calibrated_direction_prob: float,
    interval_coverage: float,
    interval_width_pct: float,
    staleness_minutes: float,
) -> dict:
    p_dir = max(0.0, min(1.0, float(calibrated_direction_prob)))
    p_interval = interval_quality_score(interval_coverage, interval_width_pct)
    p_fresh = data_freshness_score(staleness_minutes)
    score = 0.55 * p_dir + 0.30 * p_interval + 0.15 * p_fresh
    score = max(0.0, min(1.0, float(score)))
    return {
        "confidence_score": score,
        "confidence_bucket": confidence_bucket(score),
        "components": {
            "direction_prob_calibrated": p_dir,
            "interval_quality_score": p_interval,
            "data_freshness_score": p_fresh,
        },
    }
