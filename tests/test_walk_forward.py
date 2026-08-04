import numpy as np
import pandas as pd
import pytest

from backtest.walk_forward import purged_walk_forward_splits, purged_walk_forward_splits_by_step
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


def test_by_step_produces_many_more_folds_than_fixed_n_splits():
    dataset = _dataset(n=700)
    folds = purged_walk_forward_splits_by_step(dataset, min_train_size=252, step_size=21, horizon=21, embargo_days=5)

    # ~700-252 linhas restantes / passo de 21 -> bem mais que os 5 folds do metodo antigo
    assert len(folds) > 15


def test_by_step_first_fold_respects_min_train_size():
    dataset = _dataset(n=700)
    folds = purged_walk_forward_splits_by_step(dataset, min_train_size=252, step_size=21, horizon=21, embargo_days=5)

    first_train, first_test = folds[0]
    assert len(first_train) <= 252 - (21 + 5)  # purgado, entao um pouco menor que o minimo bruto
    assert first_test.index.min() >= dataset.index[252]


def test_by_step_no_overlap_between_train_and_test():
    dataset = _dataset(n=700)
    folds = purged_walk_forward_splits_by_step(dataset, min_train_size=200, step_size=21, horizon=21, embargo_days=5)

    for train, test in folds:
        assert train.index.max() < test.index.min()


def test_by_step_folds_are_chronological_and_non_overlapping_across_folds():
    dataset = _dataset(n=700)
    folds = purged_walk_forward_splits_by_step(dataset, min_train_size=200, step_size=21, horizon=21, embargo_days=5)

    prev_test_end = None
    for _, test in folds:
        if prev_test_end is not None:
            assert test.index.min() > prev_test_end
        prev_test_end = test.index.max()


def test_by_step_train_set_expands_across_folds():
    dataset = _dataset(n=700)
    folds = purged_walk_forward_splits_by_step(dataset, min_train_size=200, step_size=21, horizon=21, embargo_days=5)

    train_sizes = [len(train) for train, _ in folds]
    assert train_sizes == sorted(train_sizes)  # treino so cresce (janela expansiva)
    assert train_sizes[-1] > train_sizes[0]
