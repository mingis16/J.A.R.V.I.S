from __future__ import annotations

import numpy as np

from signal_engine.calibration import (
    brier_score,
    expected_calibration_error,
    fit_calibrator,
    reliability_diagram_data,
)


def test_brier_score_perfect_predictions_is_zero():
    y = np.array([1, 0, 1, 0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y, p) == 0.0


def test_brier_score_worst_case_predictions_is_one():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y, p) == 1.0


def test_ece_is_zero_for_perfectly_calibrated_buckets():
    rng = np.random.default_rng(0)
    p = np.repeat([0.2, 0.8], 1000)
    y = rng.binomial(1, p)
    # Use bucket means of p rather than raw p as "true" calibration target to
    # keep this a check on the ECE machinery, not on binomial noise.
    ece = expected_calibration_error(y, p, n_buckets=10)
    assert ece < 0.05  # small sample noise only, should be well-calibrated by construction


def test_ece_is_large_for_badly_miscalibrated_predictions():
    y = np.zeros(1000)  # always the negative class
    p = np.full(1000, 0.9)  # model always confidently predicts positive
    ece = expected_calibration_error(y, p, n_buckets=10)
    assert ece > 0.5


def test_reliability_diagram_bucket_counts_sum_to_total():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.05, 0.15, 0.55, 0.65, 0.95])
    buckets = reliability_diagram_data(y, p, n_buckets=10)
    assert sum(b.count for b in buckets) == len(y)


def test_fit_calibrator_sigmoid_improves_calibration():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, size=2000)
    # Badly miscalibrated raw scores: correlated with y but squashed near 0.9/0.1
    raw_p = np.where(y == 1, rng.uniform(0.6, 0.95, size=2000), rng.uniform(0.05, 0.4, size=2000))
    calibrator = fit_calibrator(raw_p, y, method="sigmoid")
    calibrated = calibrator.calibrate(raw_p)
    assert brier_score(y, calibrated) <= brier_score(y, raw_p)


def test_fit_calibrator_isotonic_runs():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.4, size=500)
    raw_p = rng.uniform(0, 1, size=500)
    calibrator = fit_calibrator(raw_p, y, method="isotonic")
    calibrated = calibrator.calibrate(raw_p)
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0
