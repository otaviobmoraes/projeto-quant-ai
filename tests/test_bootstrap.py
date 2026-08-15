"""Testes da inferencia sobre o Sharpe com operacoes sobrepostas
(backtest/bootstrap.py) e do modo `allow_overlap` do motor.

Os dois testes que carregam o argumento:
- `test_hac_alarga_o_erro_padrao_com_autocorrelacao`: dado autocorrelacionado
  tem MENOS informacao que o n bruto sugere, e o erro padrao precisa refletir
  isso. Se nao refletisse, seria a armadilha nº 2 do projeto de volta.
- `test_cobertura_do_intervalo_em_serie_sobreposta`: o IC tem que cobrir o
  Sharpe verdadeiro perto de 95% das vezes numa serie com a MESMA estrutura de
  sobreposicao do backtest real.
"""

import numpy as np
import pandas as pd
import pytest

from backtest import engine
from backtest.bootstrap import (
    block_bootstrap_sharpe,
    effective_sample_size,
    sharpe_se_hac,
)

ANUAL = 12.0  # 252/21, a convencao do projeto


def _ma_sobreposta(n: int, horizon: int, mu: float, sigma: float, seed: int) -> np.ndarray:
    """Serie com a estrutura do backtest sobreposto: cada 'trade' e a media
    movel de `horizon` choques diarios, entao trades vizinhos compartilham
    horizon-1 choques -- exatamente MA(horizon-1)."""
    rng = np.random.default_rng(seed)
    choques = rng.normal(mu, sigma, n + horizon - 1)
    return np.convolve(choques, np.ones(horizon) / horizon, mode="valid")


# ---------------------------------------------------------------------------
# Erro padrao HAC
# ---------------------------------------------------------------------------


def test_hac_bate_com_o_ingenuo_quando_nao_ha_autocorrelacao():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1.0, 4000)
    r = sharpe_se_hac(x, annualization=ANUAL, lags=0)
    ingenuo = np.sqrt((1 + r["sharpe"] ** 2 / (2 * ANUAL)) / x.size) * np.sqrt(ANUAL)
    assert r["se"] == pytest.approx(ingenuo, rel=0.15)


def test_hac_alarga_o_erro_padrao_com_autocorrelacao():
    """O TESTE CENTRAL. Numa serie sobreposta o erro padrao HAC tem que ser
    bem MAIOR que o ingenuo -- e a correcao que impede o p-valor inflado."""
    x = _ma_sobreposta(400, horizon=21, mu=0.05, sigma=1.0, seed=7)

    com_hac = sharpe_se_hac(x, annualization=ANUAL)
    sem_hac = sharpe_se_hac(x, annualization=ANUAL, lags=0)

    assert com_hac["se"] > 2 * sem_hac["se"]
    assert com_hac["sharpe"] == pytest.approx(sem_hac["sharpe"])  # o ponto nao muda


def test_hac_sharpe_bate_com_a_formula_direta():
    rng = np.random.default_rng(3)
    x = rng.normal(0.2, 1.5, 500)
    r = sharpe_se_hac(x, annualization=ANUAL)
    esperado = x.mean() / x.std(ddof=0) * np.sqrt(ANUAL)
    assert r["sharpe"] == pytest.approx(esperado, rel=1e-6)


def test_hac_lida_com_amostra_minuscula_e_serie_constante():
    assert np.isnan(sharpe_se_hac([1.0, 2.0], annualization=ANUAL)["se"])
    r = sharpe_se_hac(np.ones(50), annualization=ANUAL)
    assert r["sharpe"] == 0.0


# ---------------------------------------------------------------------------
# Bootstrap de bloco
# ---------------------------------------------------------------------------


def test_bootstrap_detecta_sharpe_claramente_positivo():
    rng = np.random.default_rng(11)
    x = rng.normal(0.5, 1.0, 600)  # SR diario alto, sem autocorrelacao
    r = block_bootstrap_sharpe(x, annualization=ANUAL, n_boot=800, seed=1)

    assert r["ic_baixo"] > 0
    assert r["p_value_bootstrap"] < 0.05
    assert r["significante_5pct"]


def test_bootstrap_nao_rejeita_ruido_puro():
    rng = np.random.default_rng(12)
    x = rng.normal(0.0, 1.0, 600)
    r = block_bootstrap_sharpe(x, annualization=ANUAL, n_boot=800, seed=2)

    assert r["ic_baixo"] < 0 < r["ic_alto"]
    assert r["p_value_bootstrap"] > 0.05
    assert not r["significante_5pct"]


def test_intervalo_e_mais_largo_em_serie_sobreposta():
    """Mesmo Sharpe pontual, mas com sobreposicao o intervalo tem que abrir."""
    rng = np.random.default_rng(13)
    independente = rng.normal(0.05, 1.0 / np.sqrt(21), 400)
    sobreposta = _ma_sobreposta(400, horizon=21, mu=0.05, sigma=1.0, seed=13)

    a = block_bootstrap_sharpe(independente, annualization=ANUAL, n_boot=600, seed=3)
    b = block_bootstrap_sharpe(sobreposta, annualization=ANUAL, n_boot=600, seed=3)

    assert (b["ic_alto"] - b["ic_baixo"]) > (a["ic_alto"] - a["ic_baixo"])


def test_cobertura_do_intervalo_em_serie_sobreposta():
    """O IC de 95% tem que cobrir o Sharpe verdadeiro na maioria larga das
    replicacoes. Sem isso, o intervalo seria decorativo."""
    horizon, mu, sigma = 21, 0.05, 1.0
    sr_verdadeiro = mu / (sigma / np.sqrt(horizon)) * np.sqrt(ANUAL)

    cobriu = 0
    n_rep = 40
    for s in range(n_rep):
        x = _ma_sobreposta(300, horizon=horizon, mu=mu, sigma=sigma, seed=100 + s)
        r = block_bootstrap_sharpe(x, annualization=ANUAL, n_boot=400, seed=s)
        if r["ic_baixo"] <= sr_verdadeiro <= r["ic_alto"]:
            cobriu += 1

    # tolerante (bootstrap com n_boot modesto), mas exige cobertura real
    assert cobriu >= 0.80 * n_rep, f"cobertura {cobriu}/{n_rep} baixa demais"


def test_bootstrap_e_reprodutivel():
    rng = np.random.default_rng(14)
    x = rng.normal(0.1, 1.0, 300)
    a = block_bootstrap_sharpe(x, annualization=ANUAL, n_boot=400, seed=99)
    b = block_bootstrap_sharpe(x, annualization=ANUAL, n_boot=400, seed=99)
    assert a["p_value_bootstrap"] == b["p_value_bootstrap"]
    assert a["ic_baixo"] == pytest.approx(b["ic_baixo"])


# ---------------------------------------------------------------------------
# Tamanho efetivo de amostra
# ---------------------------------------------------------------------------


def test_tamanho_efetivo_limitado_pelo_span():
    r = effective_sample_size(n_trades=224, span_days=1000, horizon=21)
    assert r["n_efetivo"] == 47
    assert r["fator_de_sobreposicao"] == pytest.approx(224 / 47)


def test_tamanho_efetivo_igual_ao_bruto_quando_nao_ha_sobreposicao():
    r = effective_sample_size(n_trades=40, span_days=1000, horizon=21)
    assert r["n_efetivo"] == 40
    assert r["fator_de_sobreposicao"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Modo sobreposto do motor
# ---------------------------------------------------------------------------

HORIZON = 5


def test_allow_overlap_gera_muito_mais_trades():
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    prices = pd.Series(5.0 + np.linspace(0, 0.4, 60), index=idx)
    fc = pd.Series(5.0, index=idx)
    iv = pd.Series(20.0, index=idx)
    kwargs = dict(horizon=HORIZON, band_pct=1.0, spread_pct=0.0, futures_spread_pct=0.0)

    sem = engine.run_backtest_delta_hedged(prices, fc, iv, **kwargs)
    com = engine.run_backtest_delta_hedged(prices, fc, iv, allow_overlap=True, **kwargs)

    assert len(com) > 4 * len(sem)
    # sobrepostos: entra em toda data que ainda tem horizonte pela frente
    assert len(com) == 60 - HORIZON
    # e as entradas do modo nao-sobreposto sao um subconjunto das sobrepostas
    assert set(sem["entry_date"]).issubset(set(com["entry_date"]))


def test_trades_sobrepostos_de_fato_se_sobrepoem():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    prices = pd.Series(5.0 + np.linspace(0, 0.3, 40), index=idx)
    fc, iv = pd.Series(5.0, index=idx), pd.Series(20.0, index=idx)

    tr = engine.run_backtest_delta_hedged(
        prices, fc, iv, horizon=HORIZON, band_pct=1.0, spread_pct=0.0,
        futures_spread_pct=0.0, allow_overlap=True,
    )
    assert (tr["entry_date"].iloc[1] < tr["exit_date"].iloc[0])