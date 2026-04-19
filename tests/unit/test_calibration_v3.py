import numpy as np

from models.calibration_v3 import expected_calibration_error, fit_direction_calibrator


def test_fit_direction_calibrator_output_range():
    raw = np.linspace(0.05, 0.95, 100)
    y = (raw > 0.5).astype(int)
    method, _, p = fit_direction_calibrator(raw, y)
    assert method in {"isotonic", "platt"}
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_ece_zero_for_perfect():
    y = np.array([0, 0, 1, 1], dtype=int)
    p = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    assert expected_calibration_error(y, p, bins=4) == 0.0
