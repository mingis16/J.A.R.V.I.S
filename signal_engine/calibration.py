"""Probability calibration, fit on a validation split held out from both
the training data the base model saw and the test fold it's evaluated on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

Method = Literal["sigmoid", "isotonic"]


@dataclass
class Calibrator:
    method: Method
    model: Any

    def calibrate(self, raw_p: np.ndarray) -> np.ndarray:
        raw_p = np.asarray(raw_p)
        if self.method == "sigmoid":
            return self.model.predict_proba(raw_p.reshape(-1, 1))[:, 1]
        return self.model.predict(raw_p)


def fit_calibrator(raw_p_val: np.ndarray, y_val: np.ndarray, method: Method = "sigmoid") -> Calibrator:
    if method == "sigmoid":
        lr = LogisticRegression()
        lr.fit(np.asarray(raw_p_val).reshape(-1, 1), y_val)
        return Calibrator(method="sigmoid", model=lr)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_p_val, y_val)
        return Calibrator(method="isotonic", model=iso)
    raise ValueError(f"Unknown calibration method: {method}")


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y_true, p))


@dataclass
class ReliabilityBucket:
    bucket_low: float
    bucket_high: float
    mean_predicted: float
    observed_rate: float
    count: int


def reliability_diagram_data(y_true: np.ndarray, p: np.ndarray, n_buckets: int = 10) -> list[ReliabilityBucket]:
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    edges = np.linspace(0, 1, n_buckets + 1)
    buckets = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_buckets - 1 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            buckets.append(ReliabilityBucket(lo, hi, float("nan"), float("nan"), 0))
            continue
        buckets.append(
            ReliabilityBucket(
                bucket_low=lo,
                bucket_high=hi,
                mean_predicted=float(p[mask].mean()),
                observed_rate=float(y_true[mask].mean()),
                count=count,
            )
        )
    return buckets


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_buckets: int = 10) -> float:
    buckets = reliability_diagram_data(y_true, p, n_buckets)
    total = sum(b.count for b in buckets)
    if total == 0:
        return float("nan")
    ece = 0.0
    for b in buckets:
        if b.count == 0:
            continue
        ece += (b.count / total) * abs(b.mean_predicted - b.observed_rate)
    return ece
