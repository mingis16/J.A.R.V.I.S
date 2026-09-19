from __future__ import annotations

import pytest

from signal_engine.splits import purged_walk_forward_folds


def test_folds_are_chronological_and_non_overlapping():
    folds = purged_walk_forward_folds(n_bars=1000, n_folds=4, horizon_bars=10, embargo_bars=5, min_train_bars=200)
    assert len(folds) == 4
    for i in range(len(folds) - 1):
        assert folds[i].test_idx[-1] < folds[i + 1].test_idx[0]


def test_no_train_index_within_horizon_plus_embargo_of_test_start():
    horizon, embargo = 24, 12
    folds = purged_walk_forward_folds(n_bars=2000, n_folds=5, horizon_bars=horizon, embargo_bars=embargo, min_train_bars=500)
    for fold in folds:
        if len(fold.train_idx) == 0:
            continue
        test_start = fold.test_idx[0]
        max_allowed_train_idx = test_start - horizon - embargo
        assert fold.train_idx.max() <= max_allowed_train_idx


def test_train_set_grows_each_fold_anchored_walk_forward():
    folds = purged_walk_forward_folds(n_bars=2000, n_folds=5, horizon_bars=10, embargo_bars=5, min_train_bars=300)
    train_sizes = [len(f.train_idx) for f in folds]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[-1] > train_sizes[0]


def test_raises_when_not_enough_bars():
    with pytest.raises(ValueError):
        purged_walk_forward_folds(n_bars=100, n_folds=5, horizon_bars=24, embargo_bars=24, min_train_bars=500)
