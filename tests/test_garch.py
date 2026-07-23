import numpy as np
import pandas as pd
import pytest

from vol import garch
from vol.forecast import forward_target


def _price_series(n: int, seed: int = 0, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = 5.0 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    return pd.Series(prices, index=idx)


def test_fit_garch_returns_expected_params():
    prices = _price_series(300, seed=1)
    from vol.realized import log_returns

    fitted = garch.fit_garch(log_returns(prices))
    assert set(["omega", "alpha[1]", "beta[1]"]).issubset(set(fitted.params.index))


def test_forecast_garch_daily_variances_are_positive():
    prices = _price_series(300, seed=2)
    from vol.realized import log_returns

    returns = log_returns(prices)
    fitted = garch.fit_garch(returns.iloc[:200])
    variances = garch.forecast_garch_daily_variances(
        returns, fitted.params, start=returns.index[200], horizon=5
    )
    assert (variances.to_numpy() > 0).all()


def test_garch_forward_target_forecast_matches_test_index_length():
    prices = _price_series(300, seed=3)
    train_end = prices.index[199]
    test_index = prices.index[210:230]

    pred = garch.garch_forward_target_forecast(prices, train_end, test_index, horizon=10)

    assert len(pred) == len(test_index)
    assert (pred > 0).all()
    assert pred.index.equals(test_index)


def test_garch_forecast_has_no_lookahead():
    prices = _price_series(300, seed=4)
    train_end = prices.index[199]
    test_index_short = prices.index[210:215]

    pred_short_universe = garch.garch_forward_target_forecast(
        prices.iloc[:220], train_end, test_index_short, horizon=5
    )
    pred_long_universe = garch.garch_forward_target_forecast(
        prices, train_end, test_index_short, horizon=5
    )

    # A previsao pros mesmos dias de teste nao pode mudar so porque a serie
    # de precos tem MAIS dados no futuro alem do periodo de teste.
    pd.testing.assert_series_equal(pred_short_universe, pred_long_universe, rtol=1e-6)


def test_evaluate_garch_fold_returns_expected_keys():
    prices = _price_series(300, seed=5)
    train = pd.DataFrame(index=prices.index[:200])
    test_index = prices.index[210:280]
    target = forward_target(prices, horizon=10).reindex(test_index)
    test = pd.DataFrame({"target": target}, index=test_index).dropna()

    result = garch.evaluate_garch_fold(prices, train, test, horizon=10)

    assert set(result.keys()) == {"rmse", "mae", "r2_oos"}
    assert result["rmse"] >= 0
