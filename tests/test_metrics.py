import numpy as np
import pandas as pd
import pytest

from backtest import metrics


def test_sharpe_ratio_zero_when_std_zero():
    assert metrics.sharpe_ratio([0.01, 0.01, 0.01]) == 0.0


def test_sharpe_ratio_positive_for_positive_mean_returns():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, 500)
    assert metrics.sharpe_ratio(returns) > 0


def test_psr_is_half_when_sr_hat_equals_benchmark():
    assert metrics.probabilistic_sharpe_ratio(sr_hat=1.2, benchmark_sr=1.2, n_obs=252) == pytest.approx(0.5)


def test_psr_increases_with_sr_hat():
    low = metrics.probabilistic_sharpe_ratio(sr_hat=0.5, benchmark_sr=0.0, n_obs=252)
    high = metrics.probabilistic_sharpe_ratio(sr_hat=1.5, benchmark_sr=0.0, n_obs=252)
    assert high > low


def test_expected_max_sharpe_increases_with_n_trials():
    small_n = metrics.expected_max_sharpe_ratio(sr_trials_std=0.5, n_trials=5)
    large_n = metrics.expected_max_sharpe_ratio(sr_trials_std=0.5, n_trials=500)
    assert large_n > small_n


def test_expected_max_sharpe_increases_with_trials_std():
    low_std = metrics.expected_max_sharpe_ratio(sr_trials_std=0.1, n_trials=50)
    high_std = metrics.expected_max_sharpe_ratio(sr_trials_std=1.0, n_trials=50)
    assert high_std > low_std


def test_expected_max_sharpe_raises_for_n_trials_below_two():
    with pytest.raises(ValueError):
        metrics.expected_max_sharpe_ratio(sr_trials_std=0.5, n_trials=1)


def test_deflated_sharpe_decreases_as_n_trials_increases():
    # Guardrail central do CLAUDE.md: testar mais configuracoes deve exigir
    # um Sharpe observado maior para ser considerado nao-sorte.
    dsr_few_trials = metrics.deflated_sharpe_ratio(
        sr_hat=1.0, sr_trials_std=0.3, n_trials=5, n_obs=252
    )
    dsr_many_trials = metrics.deflated_sharpe_ratio(
        sr_hat=1.0, sr_trials_std=0.3, n_trials=500, n_obs=252
    )
    assert dsr_many_trials < dsr_few_trials


def test_directional_accuracy_perfect_when_forecast_always_right_side():
    idx = pd.date_range("2026-01-01", periods=5)
    reference = pd.Series([10.0] * 5, index=idx)
    forecast = pd.Series([12.0, 8.0, 15.0, 5.0, 11.0], index=idx)
    actual = pd.Series([13.0, 7.0, 20.0, 2.0, 10.5], index=idx)  # mesmo lado de 10.0 que o forecast

    result = metrics.directional_accuracy(forecast, actual, reference)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["n"] == 5
    assert result["hits"] == 5


def test_directional_accuracy_zero_when_forecast_always_wrong_side():
    idx = pd.date_range("2026-01-01", periods=4)
    reference = pd.Series([10.0] * 4, index=idx)
    forecast = pd.Series([12.0, 8.0, 15.0, 5.0], index=idx)
    actual = pd.Series([7.0, 13.0, 2.0, 20.0], index=idx)  # sempre o lado OPOSTO do forecast

    result = metrics.directional_accuracy(forecast, actual, reference)
    assert result["accuracy"] == pytest.approx(0.0)


def test_directional_accuracy_pvalue_small_for_strong_signal():
    idx = pd.date_range("2026-01-01", periods=40)
    reference = pd.Series(10.0, index=idx)
    rng = np.random.default_rng(0)
    forecast = pd.Series(reference.to_numpy() + rng.normal(2, 0.1, 40), index=idx)
    actual = pd.Series(reference.to_numpy() + rng.normal(2, 0.1, 40), index=idx)  # mesmo lado quase sempre

    result = metrics.directional_accuracy(forecast, actual, reference)
    assert result["accuracy"] > 0.9
    assert result["pvalue_vs_50pct"] < 0.01


def test_directional_accuracy_aligns_by_index_and_drops_missing():
    idx = pd.date_range("2026-01-01", periods=5)
    reference = pd.Series(10.0, index=idx)
    forecast = pd.Series([12.0, 8.0, np.nan, 5.0, 11.0], index=idx)
    actual = pd.Series([13.0, 7.0, 20.0, 2.0, 10.5], index=idx)

    result = metrics.directional_accuracy(forecast, actual, reference)
    assert result["n"] == 4  # a linha com NaN no forecast e descartada
