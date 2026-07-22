import numpy as np
import pandas as pd
import pytest

from backtest.walk_forward import purged_walk_forward_splits
from vol.forecast import build_dataset


def _dataset(n: int = 300, horizon: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="America/Sao_Paulo")
    prices = pd.Series(5.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    return build_dataset(prices, horizon=horizon)


def test_purged_walk_forward_splits_no_overlap_between_train_and_test():
    dataset = _dataset()
    folds = purged_walk_forward_splits(dataset, n_splits=4, horizon=21)

    assert len(folds) == 4
    for train, test in folds:
        assert train.index.max() < test.index.min()


def test_purge_removes_rows_within_horizon_of_test_start():
    dataset = _dataset()
    horizon = 21
    folds_no_purge = purged_walk_forward_splits(dataset, n_splits=4, horizon=0)
    folds_purged = purged_walk_forward_splits(dataset, n_splits=4, horizon=horizon)

    for (train_np, _), (train_p, _) in zip(folds_no_purge, folds_purged):
        assert len(train_p) == len(train_np) - horizon


def test_embargo_removes_additional_rows():
    dataset = _dataset()
    folds_no_embargo = purged_walk_forward_splits(dataset, n_splits=4, horizon=21, embargo_days=0)
    folds_embargo = purged_walk_forward_splits(dataset, n_splits=4, horizon=21, embargo_days=10)

    for (train_a, _), (train_b, _) in zip(folds_no_embargo, folds_embargo):
        assert len(train_b) == len(train_a) - 10


def test_folds_are_chronological_and_non_overlapping_across_folds():
    dataset = _dataset()
    folds = purged_walk_forward_splits(dataset, n_splits=4, horizon=21)

    prev_test_end = None
    for _, test in folds:
        if prev_test_end is not None:
            assert test.index.min() > prev_test_end
        prev_test_end = test.index.max()


def test_raises_when_dataset_too_small():
    dataset = _dataset(n=5, horizon=1)
    with pytest.raises(ValueError):
        purged_walk_forward_splits(dataset, n_splits=10, horizon=1)
