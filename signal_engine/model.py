"""Transparent baseline model: logistic regression over the engineered
features. Per the spec — no deep learning until this is beaten; REPORT.md
notes gradient boosting as the documented next step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineModel:
    scaler: StandardScaler
    classifier: LogisticRegression
    feature_names: list[str]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.scaler.transform(X))[:, 1]


def train_baseline(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> BaselineModel:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_scaled, y)
    return BaselineModel(scaler=scaler, classifier=clf, feature_names=feature_names)
