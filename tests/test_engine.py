import numpy as np
import pandas as pd
import pytest

from backtest import engine
from strategy.signal import LONG_VOL, SHORT_VOL


def _price_series(n: int, seed: int = 0, vol: float = 0.005) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = 5.0 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    return pd.Series(prices, index=idx)


def test_calibrate_risk_premium_matches_manual_ratio():
    premium = engine.calibrate_risk_premium(rv_at_calibration_date=8.0, iv_at_calibration_date=10.0)
    assert premium == pytest.approx(1.25)


def test_proxy_iv_scales_rv_by_premium():
    rv = pd.Series([5.0, 10.0, 15.0])
    iv = engine.proxy_iv(rv, risk_premium=1.2)
    assert iv.tolist() == pytest.approx([6.0, 12.0, 18.0])


def test_generate_oos_rv_forecast_covers_test_folds_only():
    from vol.forecast import BASELINE_FEATURES, build_dataset

    prices = _price_series(400, seed=1)
    dataset = build_dataset(prices, horizon=21)

    forecast_series = engine.generate_oos_rv_forecast(
        dataset, BASELINE_FEATURES, horizon=21, n_splits=4, embargo_days=5
    )

    assert len(forecast_series) > 0
    assert forecast_series.index.isin(dataset.index).all()
    assert forecast_series.notna().all()


def test_run_backtest_no_trades_when_band_covers_everything():
    prices = _price_series(200, seed=2)
    idx = prices.index
    rv_forecast = pd.Series(10.0, index=idx)
    iv = pd.Series(10.0, index=idx)  # rv == iv sempre -> dentro de qualquer banda > 0

    trades = engine.run_backtest(prices, rv_forecast, iv, horizon=21, band_pct=1.0)
    assert trades.empty


def test_run_backtest_generates_long_vol_trade_when_rv_forecast_high():
    prices = _price_series(200, seed=3)
    idx = prices.index
    rv_forecast = pd.Series(20.0, index=idx)  # RV prevista bem acima da IV
    iv = pd.Series(9.0, index=idx)

    trades = engine.run_backtest(prices, rv_forecast, iv, horizon=21, band_pct=1.0)

    assert len(trades) > 0
    assert (trades["signal"] == LONG_VOL).all()
    assert (trades["n_contracts"] > 0).all()
    assert (trades["premium"] > 0).all()


def test_run_backtest_generates_short_vol_trade_when_rv_forecast_low():
    prices = _price_series(200, seed=4)
    idx = prices.index
    rv_forecast = pd.Series(3.0, index=idx)  # RV prevista bem abaixo da IV
    iv = pd.Series(15.0, index=idx)

    trades = engine.run_backtest(prices, rv_forecast, iv, horizon=21, band_pct=1.0)

    assert len(trades) > 0
    assert (trades["signal"] == SHORT_VOL).all()


def test_run_backtest_trades_do_not_overlap():
    prices = _price_series(300, seed=5)
    idx = prices.index
    rv_forecast = pd.Series(20.0, index=idx)
    iv = pd.Series(9.0, index=idx)

    trades = engine.run_backtest(prices, rv_forecast, iv, horizon=21, band_pct=1.0)

    assert len(trades) > 1
    for i in range(len(trades) - 1):
        assert trades["exit_date"].iloc[i] <= trades["entry_date"].iloc[i + 1]


def test_run_backtest_pnl_matches_manual_calc_for_single_trade():
    prices = _price_series(60, seed=6)
    idx = prices.index
    rv_forecast = pd.Series(np.nan, index=idx)
    rv_forecast.iloc[0] = 20.0  # so um sinal, no primeiro dia
    iv = pd.Series(9.0, index=idx)

    trades = engine.run_backtest(prices, rv_forecast, iv, horizon=21, band_pct=1.0, spread_pct=0.0)

    assert len(trades) == 1
    trade = trades.iloc[0]

    from strategy.sizing import size_straddle
    from vol.black76 import call_price, put_price

    K = prices.iloc[0]
    premium = call_price(K, K, 21 / 365, 0.09, 0.0) + put_price(K, K, 21 / 365, 0.09, 0.0)
    n = size_straddle(target_vega=1000.0, spot=K, ttm_days=21, iv_pct=9.0)
    payoff = abs(prices.iloc[21] - K)
    expected_pnl = n * (payoff - premium)

    assert trade["pnl_gross"] == pytest.approx(expected_pnl)
    assert trade["pnl_net"] == pytest.approx(expected_pnl)  # spread_pct=0 -> custo zero


def test_summarize_backtest_empty_trades():
    result = engine.summarize_backtest(pd.DataFrame())
    assert result == {"n_trades": 0}


def test_summarize_backtest_computes_expected_fields():
    trades = pd.DataFrame(
        {
            "signal": [LONG_VOL, SHORT_VOL, LONG_VOL],
            "n_contracts": [100.0, 100.0, 100.0],
            "premium": [0.10, 0.10, 0.10],
            "pnl_net": [5.0, -3.0, 2.0],
        }
    )
    result = engine.summarize_backtest(trades, horizon=21)

    assert result["n_trades"] == 3
    assert result["win_rate"] == pytest.approx(2 / 3)
    assert result["total_pnl"] == pytest.approx(4.0)
    assert result["n_long_vol"] == 2
    assert result["n_short_vol"] == 1
    assert "sharpe" in result
