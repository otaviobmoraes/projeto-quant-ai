"""Testes do dimensionamento por RISCO (strategy/sizing.py) e da sua ligacao
com o backtest delta-neutro."""

import numpy as np
import pandas as pd
import pytest

from backtest import engine
from strategy.sizing import rv_dispersion, size_by_risk_target, size_straddle

ARGS = dict(spot=5.0, ttm_days=21, iv_pct=12.0)


def test_dispersao_usa_so_janela_passada():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rv = pd.Series(np.arange(400.0), index=idx)
    d = rv_dispersion(rv, window=252)

    # sem janela minima cumprida, NaN -- nunca um numero vindo do futuro
    assert d.iloc[:125].isna().all()
    assert d.iloc[-1] == pytest.approx(rv.iloc[-252:].std())


def test_tamanho_cai_quando_a_vol_da_vol_sobe():
    """O comportamento central: dispersao maior => posicao menor."""
    calmo = size_by_risk_target(target_risk=1000.0, rv_dispersion_pct=1.0, **ARGS)
    turbulento = size_by_risk_target(target_risk=1000.0, rv_dispersion_pct=4.0, **ARGS)
    assert turbulento < calmo
    assert turbulento == pytest.approx(calmo / 4)


def test_risco_por_contrato_fica_constante():
    """Vega x dispersao (o desvio-padrao aproximado do P&L) tem que ser o mesmo
    nos dois regimes -- e a definicao de risco alvo constante."""
    from strategy.sizing import straddle_vega_per_contract

    vega = straddle_vega_per_contract(**ARGS)
    for disp in (0.8, 2.0, 3.5):
        n = size_by_risk_target(target_risk=1000.0, rv_dispersion_pct=disp, **ARGS)
        assert n * vega * disp == pytest.approx(1000.0)


def test_teto_de_alavancagem_protege_dispersao_quase_zero():
    """Em periodo muito calmo a formula pediria posicao ilimitada -- isso e
    divisao por numero pequeno, nao resultado de modelo."""
    enorme = size_by_risk_target(
        target_risk=1000.0, rv_dispersion_pct=1e-9, max_leverage=5.0, **ARGS
    )
    vega_constante = size_straddle(target_vega=1000.0, **ARGS)
    assert enorme == pytest.approx(5 * vega_constante)

    assert size_by_risk_target(
        target_risk=1000.0, rv_dispersion_pct=0.0, max_leverage=3.0, **ARGS
    ) == pytest.approx(3 * vega_constante)


def test_dispersao_negativa_ou_zero_nao_quebra():
    n = size_by_risk_target(target_risk=1000.0, rv_dispersion_pct=-1.0, **ARGS)
    assert np.isfinite(n) and n > 0


# ---------------------------------------------------------------------------
# Ligacao com o motor
# ---------------------------------------------------------------------------

HORIZON = 5


def _cenario():
    idx = pd.date_range("2024-01-01", periods=HORIZON + 3, freq="B")
    prices = pd.Series(5.0 + np.linspace(0, 0.2, HORIZON + 3), index=idx)
    rv_forecast = pd.Series(5.0, index=idx)
    iv = pd.Series(20.0, index=idx)
    return prices, rv_forecast, iv


def test_serie_de_tamanho_substitui_o_vega_constante():
    prices, fc, iv = _cenario()
    kwargs = dict(horizon=HORIZON, band_pct=1.0, spread_pct=0.0, futures_spread_pct=0.0)

    padrao = engine.run_backtest_delta_hedged(prices, fc, iv, **kwargs).iloc[0]
    dobro = pd.Series(2 * padrao["n_contracts"], index=prices.index)
    forcado = engine.run_backtest_delta_hedged(
        prices, fc, iv, n_contracts_series=dobro, **kwargs
    ).iloc[0]

    assert forcado["n_contracts"] == pytest.approx(2 * padrao["n_contracts"])
    # P&L e linear no tamanho da posicao
    assert forcado["pnl_gross"] == pytest.approx(2 * padrao["pnl_gross"])


def test_datas_ausentes_caem_no_vega_constante():
    prices, fc, iv = _cenario()
    kwargs = dict(horizon=HORIZON, band_pct=1.0, spread_pct=0.0, futures_spread_pct=0.0)

    padrao = engine.run_backtest_delta_hedged(prices, fc, iv, **kwargs).iloc[0]
    vazia = pd.Series(dtype=float)
    caiu_no_padrao = engine.run_backtest_delta_hedged(
        prices, fc, iv, n_contracts_series=vazia, **kwargs
    ).iloc[0]

    assert caiu_no_padrao["n_contracts"] == pytest.approx(padrao["n_contracts"])


def test_tamanho_invalido_e_ignorado():
    prices, fc, iv = _cenario()
    kwargs = dict(horizon=HORIZON, band_pct=1.0, spread_pct=0.0, futures_spread_pct=0.0)
    padrao = engine.run_backtest_delta_hedged(prices, fc, iv, **kwargs).iloc[0]

    for ruim in (np.nan, 0.0, -3.0):
        serie = pd.Series(ruim, index=prices.index)
        linha = engine.run_backtest_delta_hedged(
            prices, fc, iv, n_contracts_series=serie, **kwargs
        ).iloc[0]
        assert linha["n_contracts"] == pytest.approx(padrao["n_contracts"])