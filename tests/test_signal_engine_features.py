"""The spec's explicitly required leakage test: shifting features forward
relative to labels must collapse performance to chance. If a shifted
version still predicts well, some feature is leaking future information.

Uses a synthetic price series with an engineered, aperiodic trending
structure (random-length up/down blocks) so there is genuine, learnable
signal in the *correctly aligned* features/labels — otherwise a "collapse
to chance" result would be meaningless (there'd be nothing to collapse).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from signal_engine.backtest import usable_rows
from signal_engine.features import FEATURE_COLUMNS, build_features
from signal_engine.labeling import triple_barrier_labels
from signal_engine.model import train_baseline

SHIFT = 600  # much larger than any block length or the horizon, so shifted
             # features land in an unrelated, uncorrelated regime


def _synthetic_trending_prices(n_bars: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 100.0
    prices = [price]
    bar = 0
    while bar < n_bars:
        block_len = int(rng.integers(50, 150))
        direction = rng.choice([-1, 1])
        drift = direction * 0.08
        for _ in range(block_len):
            price += drift + rng.normal(0, 0.05)
            prices.append(price)
            bar += 1
            if bar >= n_bars:
                break

    prices = np.array(prices[: n_bars + 1])
    close = prices[1:]
    open_ = prices[:-1]
    high = np.maximum(open_, close) + rng.uniform(0, 0.05, size=n_bars)
    low = np.minimum(open_, close) - rng.uniform(0, 0.05, size=n_bars)
    times = pd.date_range("2020-01-01", periods=n_bars, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": rng.integers(50, 200, size=n_bars),
        }
    )


def _auc_for_features(feat_matrix: np.ndarray, labels: np.ndarray) -> float:
    n = len(labels)
    split = int(n * 0.7)
    X_train, X_test = feat_matrix[:split], feat_matrix[split:]
    y_train, y_test = labels[:split], labels[split:]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return 0.5
    model = train_baseline(X_train, y_train, FEATURE_COLUMNS)
    p_test = model.predict_proba(X_test)
    return roc_auc_score(y_test, p_test)


def test_properly_aligned_features_beat_chance_and_shifted_features_collapse():
    df = _synthetic_trending_prices(4000)
    feats = build_features(df)
    side = np.where(feats["ema_fast"] > feats["ema_slow"], 1, -1)
    labels = triple_barrier_labels(df, side, feats["atr_14"].to_numpy(), atr_sl_mult=1.0, atr_tp_mult=2.0, horizon_bars=10)

    feats = feats.copy()
    feats["label"] = labels.label
    feats["outcome"] = labels.outcome
    usable = usable_rows(feats).reset_index(drop=True)

    aligned_X = usable[FEATURE_COLUMNS].to_numpy()
    aligned_y = usable["label"].to_numpy()
    aligned_auc = _auc_for_features(aligned_X, aligned_y)

    # Shift features forward relative to label by SHIFT rows, dropping the
    # tail rows that no longer have a valid shifted feature vector.
    shifted_feats = usable[FEATURE_COLUMNS].shift(-SHIFT)
    valid = shifted_feats.dropna().index
    shifted_X = shifted_feats.loc[valid].to_numpy()
    shifted_y = usable.loc[valid, "label"].to_numpy()
    shifted_auc = _auc_for_features(shifted_X, shifted_y)

    assert aligned_auc > 0.6, f"expected genuine signal in aligned features, got AUC={aligned_auc:.3f}"
    assert abs(shifted_auc - 0.5) < 0.1, f"shifted features should collapse to chance, got AUC={shifted_auc:.3f}"
    assert aligned_auc - shifted_auc > 0.15
