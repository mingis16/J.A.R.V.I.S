"""Purged, embargoed, anchored walk-forward folds.

Anchored: fold i's training set is everything chronologically before its
test block (not just a fixed-size rolling window) — more data each fold.
Purged: the last `horizon_bars` of the training set are dropped, because a
label that far into the past could look forward into the test block it's
supposed to be excluded from.
Embargoed: an additional `embargo_bars` buffer is dropped on top of the
purge, as extra insurance against boundary leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def purged_walk_forward_folds(
    n_bars: int,
    n_folds: int,
    horizon_bars: int,
    embargo_bars: int,
    min_train_bars: int,
) -> list[Fold]:
    usable_end = n_bars - horizon_bars  # bars at/after this can't be labeled
    test_region_len = usable_end - min_train_bars
    if test_region_len <= 0:
        raise ValueError(f"Not enough bars ({n_bars}) for min_train_bars={min_train_bars} + horizon={horizon_bars}")

    fold_size = test_region_len // n_folds
    if fold_size <= 0:
        raise ValueError(f"n_folds={n_folds} too high for {test_region_len} usable bars")

    folds = []
    for i in range(n_folds):
        test_start = min_train_bars + i * fold_size
        test_end = usable_end if i == n_folds - 1 else min_train_bars + (i + 1) * fold_size

        purge_boundary = test_start - horizon_bars - embargo_bars
        train_end = max(purge_boundary, 0)

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        folds.append(Fold(train_idx=train_idx, test_idx=test_idx))

    return folds
