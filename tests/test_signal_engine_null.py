"""Null test at the unit level: shuffling labels before fitting must destroy
any apparent edge. This mirrors the null test run_walk_forward performs on
real data (see signal_engine/backtest.py), isolated here as a fast,
deterministic check on the model+calibration+gating machinery itself.

Important design point: the label's marginal P(Y=1) is centered exactly at
the RR=2 breakeven probability (1/3). If it weren't, an uninformative model
that just learns the *overall* base rate could still look profitable once
calibrated — not because it found real per-case signal, but because the
unconditional rate alone clears breakeven. Centering at breakeven means only
genuine per-case discrimination (correlation with X[:, 0]) can produce
gated trades with positive expectancy.
"""
from __future__ import annotations

import numpy as np

from signal_engine.calibration import fit_calibrator
from signal_engine.costs import expectancy_r
from signal_engine.model import train_baseline

FEATURE_NAMES = [f"f{i}" for i in range(5)]
RISK_REWARD = 2.0
BREAKEVEN = 1 / (1 + RISK_REWARD)
THRESHOLD = 0.4  # above breakeven, so only real discrimination clears it


def _gated_expectancy(model, calibrator, X_test, threshold=THRESHOLD):
    p = calibrator.calibrate(model.predict_proba(X_test))
    take = p > threshold
    if take.sum() == 0:
        return 0.0, 0
    return float(np.mean([expectancy_r(pi, RISK_REWARD, 0.0) for pi in p[take]])), int(take.sum())


def test_shuffled_labels_produce_no_edge():
    rng = np.random.default_rng(3)
    n = 3000
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    # Real per-case signal in X[:, 0], marginal rate centered at breakeven.
    p_true = np.clip(BREAKEVEN + 0.25 * X[:, 0], 0.02, 0.98)
    y = rng.binomial(1, p_true)

    split = int(n * 0.7)
    val_split = split - 400
    X_train, y_train = X[:val_split], y[:val_split]
    X_val, y_val = X[val_split:split], y[val_split:split]
    X_test = X[split:]

    # Real model: trains and calibrates on the true labels.
    real_model = train_baseline(X_train, y_train, FEATURE_NAMES)
    real_calibrator = fit_calibrator(real_model.predict_proba(X_val), y_val, method="sigmoid")
    real_expectancy, real_n_trades = _gated_expectancy(real_model, real_calibrator, X_test)

    # Null model: ONE consistent shuffle applied across both training and
    # calibration labels (not the true labels re-shuffled twice independently).
    shuffled_y = rng.permutation(y[:split])
    shuffled_y_train, shuffled_y_val = shuffled_y[:val_split], shuffled_y[val_split:]
    null_model = train_baseline(X_train, shuffled_y_train, FEATURE_NAMES)
    null_calibrator = fit_calibrator(null_model.predict_proba(X_val), shuffled_y_val, method="sigmoid")
    null_expectancy, null_n_trades = _gated_expectancy(null_model, null_calibrator, X_test)

    assert real_n_trades > 50, f"expected the real-signal model to gate in a meaningful number of trades, got {real_n_trades}"
    assert real_expectancy > 0.1, f"expected the real-signal model to show positive gated expectancy, got {real_expectancy:.3f}"

    # A shuffled-label model has no real per-case skill: either it gates in
    # ~nothing (its calibrated probabilities hover near the breakeven-ish
    # marginal rate, rarely clearing the threshold), or if noise happens to
    # gate a few trades in, their expectancy should be small.
    assert null_n_trades < real_n_trades / 3, f"shuffled model gated in too many trades ({null_n_trades}) — check for leakage"
    if null_n_trades > 0:
        assert abs(null_expectancy) < 0.3, f"shuffled-label model's gated trades should show ~no edge, got {null_expectancy:.3f}"
