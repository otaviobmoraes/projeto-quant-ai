import numpy as np
import pandas as pd
import pytest

from vol import realized


def test_log_returns_matches_manual_calc():
    prices = pd.Series([5.0, 5.1, 5.0, 5.2])
    r = realized.log_returns(prices)

    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(5.1 / 5.0))
    assert r.iloc[2] == pytest.approx(np.log(5.0 / 5.1))
    assert r.iloc[3] == pytest.approx(np.log(5.2 / 5.0))


def test_realized_vol_constant_returns_is_zero():
    # Retornos log constantes -> desvio padrao zero -> RV zero (a partir da janela).
    prices = pd.Series([5.0 * (1.01**i) for i in range(10)])
    rv = realized.realized_vol(prices, window=3)

    assert rv.iloc[3:].abs().max() < 1e-8


def test_realized_vol_scales_with_annualization_factor():
    rng = np.random.default_rng(42)
    prices = pd.Series(5.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))))

    rv_252 = realized.realized_vol(prices, window=21, annualization_factor=252)
    rv_1 = realized.realized_vol(prices, window=21, annualization_factor=1)

    ratio = (rv_252 / rv_1).dropna()
    assert ratio.apply(lambda x: x == pytest.approx(np.sqrt(252), rel=1e-6)).all()


def test_load_ptax_realized_vol_reads_processed_parquet(tmp_path, monkeypatch):
    dates = pd.date_range("2026-01-01", periods=30, freq="B", tz="America/Sao_Paulo")
    rng = np.random.default_rng(0)
    prices = 5.0 * np.exp(np.cumsum(rng.normal(0, 0.005, len(dates))))
    df = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "value": list(prices) + list(prices * 1.001),
            "tipo": ["venda"] * len(dates) + ["compra"] * len(dates),
        }
    )
    path = tmp_path / "ptax.parquet"
    df.to_parquet(path)
    monkeypatch.setattr(realized, "PTAX_PROCESSED_PATH", path)

    rv = realized.load_ptax_realized_vol(window=5, tipo="venda")

    assert list(rv.columns) == ["rv_pct"]
    assert len(rv) == len(dates)
    assert rv["rv_pct"].iloc[:5].isna().all() or rv["rv_pct"].isna().sum() >= 4


def test_load_ptax_realized_vol_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(realized, "PTAX_PROCESSED_PATH", tmp_path / "missing.parquet")

    with pytest.raises(FileNotFoundError):
        realized.load_ptax_realized_vol()


def test_parkinson_daily_variance_zero_when_high_equals_low():
    high = pd.Series([5.0, 5.1, 5.2])
    low = high.copy()
    variance = realized.parkinson_daily_variance(high, low)
    assert (variance == 0).all()


def test_parkinson_daily_variance_matches_manual_calc():
    high = pd.Series([5.10])
    low = pd.Series([5.00])
    variance = realized.parkinson_daily_variance(high, low)

    expected = np.log(5.10 / 5.00) ** 2 / (4 * np.log(2))
    assert variance.iloc[0] == pytest.approx(expected)


def test_parkinson_daily_variance_less_noisy_than_squared_return():
    # Mesma serie de precos: a variancia diaria via Parkinson (usa o range
    # intradiario simulado) tem desvio-padrao MENOR que o quadrado do
    # retorno de fechamento, ao longo de uma amostra grande -- e a razao de
    # usar Parkinson em primeiro lugar (estimador menos ruidoso).
    rng = np.random.default_rng(7)
    n = 500
    close = pd.Series(5.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    # Simula um range intradiario plausivel em torno do close (sempre high >= low).
    intraday_noise = np.abs(rng.normal(0, 0.003, n))
    high = close * (1 + intraday_noise)
    low = close * (1 - intraday_noise)

    close_to_close_var = realized.log_returns(close) ** 2
    parkinson_var = realized.parkinson_daily_variance(high, low)

    assert parkinson_var.std() < close_to_close_var.std()


def test_parkinson_vol_scales_with_annualization_factor():
    rng = np.random.default_rng(1)
    n = 60
    close = pd.Series(5.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    intraday_noise = np.abs(rng.normal(0, 0.003, n))
    high = close * (1 + intraday_noise)
    low = close * (1 - intraday_noise)

    vol_252 = realized.parkinson_vol(high, low, window=10, annualization_factor=252)
    vol_1 = realized.parkinson_vol(high, low, window=10, annualization_factor=1)

    ratio = (vol_252 / vol_1).dropna()
    assert ratio.apply(lambda x: x == pytest.approx(np.sqrt(252), rel=1e-6)).all()


def test_load_parkinson_prices_and_variance_reads_fx_spot(tmp_path, monkeypatch):
    idx = pd.date_range("2024-01-01", periods=30, tz="America/Sao_Paulo")
    rng = np.random.default_rng(0)
    close = 5.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 30)))
    noise = np.abs(rng.normal(0, 0.003, 30))
    fx_df = pd.DataFrame(
        {
            "date": idx,
            "open": close,
            "high": close * (1 + noise),
            "low": close * (1 - noise),
            "close": close,
        }
    )
    path = tmp_path / "fx_spot.parquet"
    fx_df.to_parquet(path)
    monkeypatch.setattr(realized, "FX_SPOT_PROCESSED_PATH", path)

    close_out, variance = realized.load_parkinson_prices_and_variance()

    assert len(close_out) == 30
    assert len(variance) == 30
    assert (variance >= 0).all()


def test_load_parkinson_prices_and_variance_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(realized, "FX_SPOT_PROCESSED_PATH", tmp_path / "missing.parquet")

    with pytest.raises(FileNotFoundError):
        realized.load_parkinson_prices_and_variance()


def test_load_b3_futures_prices_and_variance_rescales_and_computes_parkinson(tmp_path, monkeypatch):
    idx = pd.date_range("2024-01-01", periods=30, tz="America/Sao_Paulo")
    rng = np.random.default_rng(1)
    settlement = 5000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 30)))  # BRL por 1000 USD
    noise = np.abs(rng.normal(0, 0.003, 30))
    fut_df = pd.DataFrame(
        {
            "date": idx,
            "ticker": ["DOLF24"] * 30,
            "settlement": settlement,
            "open": settlement,
            "last": settlement,
            "high": settlement * (1 + noise),
            "low": settlement * (1 - noise),
            "contract_changed": [False] * 30,
        }
    )
    path = tmp_path / "b3_dol_futures.parquet"
    fut_df.to_parquet(path)
    monkeypatch.setattr(realized, "B3_FUTURES_PROCESSED_PATH", path)

    close_out, variance = realized.load_b3_futures_prices_and_variance()

    assert len(close_out) == 30
    # ajuste vem em BRL/1000 USD -> deve sair na escala do spot (~5, nao ~5000)
    assert 3.0 < close_out.mean() < 8.0
    assert (variance.dropna() >= 0).all()


def test_load_b3_futures_variance_is_scale_invariant(tmp_path, monkeypatch):
    """Parkinson usa ln(high/low), entao dividir por 1000 nao pode alterar a
    variancia -- garante que o reescalonamento nao contaminou o estimador."""
    idx = pd.date_range("2024-01-01", periods=10, tz="America/Sao_Paulo")
    base = np.linspace(5000, 5100, 10)
    fut_df = pd.DataFrame(
        {
            "date": idx,
            "ticker": ["DOLF24"] * 10,
            "settlement": base,
            "open": base,
            "last": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "contract_changed": [False] * 10,
        }
    )
    path = tmp_path / "b3_dol_futures.parquet"
    fut_df.to_parquet(path)
    monkeypatch.setattr(realized, "B3_FUTURES_PROCESSED_PATH", path)

    _, variance = realized.load_b3_futures_prices_and_variance()

    expected = realized.parkinson_daily_variance(
        pd.Series(base * 1.01, index=idx), pd.Series(base * 0.99, index=idx)
    )
    # check_freq=False: o round-trip pelo parquet perde o atributo `freq` do
    # indice; o que importa aqui sao os valores.
    pd.testing.assert_series_equal(variance, expected, check_names=False, check_freq=False)


def test_forward_realized_skewness_uses_exactly_the_next_h_returns():
    idx = pd.date_range("2024-01-01", periods=12, freq="B")
    returns = pd.Series(np.arange(12, dtype=float), index=idx)

    result = realized.forward_realized_skewness(returns, horizon=3)

    # No indice t, a janela deve ser exatamente returns[t+1 .. t+3].
    esperado_t0 = pd.Series([1.0, 2.0, 3.0]).skew()
    assert result.iloc[0] == pytest.approx(esperado_t0)
    esperado_t4 = pd.Series([5.0, 6.0, 7.0]).skew()
    assert result.iloc[4] == pytest.approx(esperado_t4)


def test_forward_realized_skewness_is_positive_for_right_tail():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    # retornos pequenos com um salto grande POSITIVO adiante -> cauda direita
    r = np.full(40, 0.001)
    r[5:8] = [0.001, 0.05, 0.001]
    returns = pd.Series(r, index=idx)

    result = realized.forward_realized_skewness(returns, horizon=10)

    # em t=0, a janela [1..10] contem o salto positivo -> assimetria a direita
    assert result.iloc[0] > 1.0


def test_forward_realized_skewness_has_no_lookahead_beyond_horizon():
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0, 0.01, 60), index=idx)

    completo = realized.forward_realized_skewness(returns, horizon=5)
    # truncar a serie DEPOIS de t+5 nao pode alterar o valor em t
    truncado = realized.forward_realized_skewness(returns.iloc[:30], horizon=5)

    comum = completo.iloc[:24].dropna().index.intersection(truncado.dropna().index)
    assert len(comum) > 0
    pd.testing.assert_series_equal(
        completo.loc[comum], truncado.loc[comum], check_names=False, check_freq=False
    )


def test_forward_realized_skewness_tolerates_roll_gaps():
    """Rolagens sao mensais: com min_periods estrito, quase toda janela de 21
    dias conteria um NaN e o alvo inteiro viraria vazio."""
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0, 0.01, 120), index=idx)
    returns.iloc[::21] = np.nan  # um "dia de rolagem" a cada 21 pregoes

    tolerante = realized.forward_realized_skewness(returns, horizon=21)
    estrito = realized.forward_realized_skewness(returns, horizon=21, min_periods=21)

    assert tolerante.notna().sum() > 50
    assert estrito.notna().sum() == 0  # confirma que o problema seria real


def test_load_b3_futures_prices_and_variance_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(realized, "B3_FUTURES_PROCESSED_PATH", tmp_path / "missing.parquet")

    with pytest.raises(FileNotFoundError):
        realized.load_b3_futures_prices_and_variance()
