from __future__ import annotations

from prediction_engine.confidence_engine import compute_confidence


def combine_predictions(xgb_out: dict) -> dict:
    if "confidence_score" in xgb_out:
        score = float(xgb_out.get("confidence_score", 0.0))
        meta = {
            "direction_prob": float(xgb_out.get("direction_prob", 0.0)),
            "direction_prob_calibrated": float(xgb_out.get("direction_prob_calibrated", 0.0)),
            "metrics": xgb_out.get("metrics", {}),
            "confidence_components": xgb_out.get("confidence_components", {}),
            "confidence_score": score,
            "confidence_bucket": xgb_out.get("confidence_bucket"),
            "pred_interval": xgb_out.get("pred_interval", {}),
            "model_family": xgb_out.get("model_family", "unknown"),
            "calibration_version": xgb_out.get("calibration_version"),
        }
        return {
            "pred_open": float(xgb_out["pred_open"]),
            "pred_high": float(xgb_out["pred_high"]),
            "pred_low": float(xgb_out["pred_low"]),
            "pred_close": float(xgb_out["pred_close"]),
            "direction": xgb_out["direction"],
            "confidence": score,
            "model_version": xgb_out["model_version"],
            "meta": meta,
        }

    confidence = compute_confidence(
        direction_prob=float(xgb_out.get("direction_prob", 0.5)),
        metrics=xgb_out.get("metrics", {}),
    )
    return {
        "pred_open": float(xgb_out["pred_open"]),
        "pred_high": float(xgb_out["pred_high"]),
        "pred_low": float(xgb_out["pred_low"]),
        "pred_close": float(xgb_out["pred_close"]),
        "direction": xgb_out["direction"],
        "confidence": confidence,
        "model_version": xgb_out["model_version"],
        "meta": {
            "direction_prob": float(xgb_out.get("direction_prob", 0.0)),
            "metrics": xgb_out.get("metrics", {}),
        },
    }
