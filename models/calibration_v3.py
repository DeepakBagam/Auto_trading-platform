from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from joblib import dump, load
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y = (y_true > 0).astype(float)
    p = np.clip(p.astype(float), 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = (y_true > 0).astype(float)
    p = np.clip(p.astype(float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(p)
    if n == 0:
        return 1.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        m = int(mask.sum())
        if m == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += (m / n) * abs(acc - conf)
    return float(ece)


def fit_direction_calibrator(raw_prob: np.ndarray, y_dir: np.ndarray):
    x = np.clip(raw_prob.astype(float), 0.0, 1.0)
    y = (y_dir > 0).astype(int)
    if len(np.unique(x)) >= 5:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x, y)
        p_cal = np.clip(iso.predict(x), 0.0, 1.0)
        return "isotonic", iso, p_cal
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(x.reshape(-1, 1), y)
    p_cal = lr.predict_proba(x.reshape(-1, 1))[:, 1]
    return "platt", lr, p_cal


def save_calibrator(artifact_dir: Path, method: str, model) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "calibrator.joblib"
    meta_path = artifact_dir / "calibration_meta.json"
    dump(model, model_path)
    meta_path.write_text(json.dumps({"method": method}), encoding="utf-8")
    return model_path


def load_calibrator(artifact_dir: Path):
    model = load(artifact_dir / "calibrator.joblib")
    meta = json.loads((artifact_dir / "calibration_meta.json").read_text(encoding="utf-8"))
    return meta.get("method", "unknown"), model


def apply_calibrator(method: str, model, raw_prob: np.ndarray) -> np.ndarray:
    x = np.clip(raw_prob.astype(float), 0.0, 1.0)
    if method == "isotonic":
        return np.clip(model.predict(x), 0.0, 1.0)
    if method == "platt":
        return np.clip(model.predict_proba(x.reshape(-1, 1))[:, 1], 0.0, 1.0)
    return x
