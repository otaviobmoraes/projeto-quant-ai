"""Testes dos diagnosticos de limite (backtest/capacity.py).

A validacao que importa nao e "roda sem erro" -- e que os diagnosticos
DISTINGAM os dois casos que precisam ser distinguidos. Por isso os testes
constroem dois datasets sinteticos: um com sinal forte de verdade e outro que
e ruido puro, e exigem que cada diagnostico aponte para o lado certo.
"""

import numpy as np
import pandas as pd
import pytest

from backtest import capacity

FEATURES = ["rv_d", "rv_w", "rv_m"]


def _dataset(n: int, seed: int, forca_do_sinal: float) -> pd.DataFrame:
    """`forca_do_sinal=0` => alvo e ruido puro (nao ha o que aprender).
    `forca_do_sinal=1` => alvo e quase inteiramente explicado pelas features."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "rv_d": rng.uniform(1, 5, n),
            "rv_w": rng.uniform(1, 5, n),
            "rv_m": rng.uniform(1, 5, n),
        },
        index=idx,
    )
    sinal = 2.0 * df["rv_d"] + 1.0 * df["rv_w"]
    ruido = rng.normal(0, 3.0, n)
    bruto = forca_do_sinal * sinal + (1 - forca_do_sinal) * ruido
    df["target"] = 10 + bruto - bruto.min()  # positivo, para o log
    return df


# ---------------------------------------------------------------------------
# Curva de aprendizado
# ---------------------------------------------------------------------------


def test_curva_usa_sempre_o_mesmo_bloco_de_teste():
    """Sem isso, a comparacao entre tamanhos de treino seria invalida
    (armadilha 4: R2 nao e comparavel entre amostras diferentes)."""
    ds = _dataset(1500, seed=1, forca_do_sinal=0.8)
    curva = capacity.learning_curve(ds, FEATURES, horizon=21, train_sizes=[100, 400, 900])
    assert len(curva) == 3
    assert curva["n_teste"].nunique() == 1


def test_curva_sobe_quando_ha_sinal_e_falta_dado():
    """Com sinal real, treinar com mais dados TEM que melhorar -- se nao
    melhorasse, o diagnostico nao serviria para nada."""
    ds = _dataset(2000, seed=2, forca_do_sinal=0.9)
    curva = capacity.learning_curve(ds, FEATURES, horizon=21, train_sizes=[60, 1200])
    assert curva["r2_oos"].iloc[-1] > curva["r2_oos"].iloc[0]


def test_curva_converge_para_zero_por_baixo_quando_nao_ha_sinal():
    """Sem sinal, mais dados NAO compram poder preditivo -- so param de
    prejudicar. O R2 sobe de muito negativo em direcao a ZERO e estaciona la;
    nunca cruza para positivo.

    Medido no sintetico: -0.127 -> -0.056 -> -0.046. O primeiro salto e grande
    (menos variancia de estimacao), o segundo e residual. Confundir esse
    primeiro salto com "o modelo esta aprendendo" e o erro que este teste
    existe para travar.
    """
    ds = _dataset(2000, seed=3, forca_do_sinal=0.0)
    curva = capacity.learning_curve(ds, FEATURES, horizon=21, train_sizes=[200, 600, 1200])

    # nunca vira previsao util
    assert curva["r2_oos"].max() < 0.05
    # e o ganho marginal decai: o segundo passo melhora bem menos que o primeiro
    ganhos = curva["r2_oos"].diff().dropna().to_numpy()
    assert ganhos[-1] < ganhos[0] / 3


def test_treino_e_purgado_do_teste():
    ds = _dataset(1200, seed=4, forca_do_sinal=0.5)
    train, test = capacity._split_fixo(
        ds, test_size=300, train_size=None, horizon=21, embargo_days=5
    )
    assert train.index.max() < test.index.min()
    # a lacuna tem que ter pelo menos horizon + embargo pregoes
    lacuna = ds.index.get_loc(test.index[0]) - ds.index.get_loc(train.index[-1])
    assert lacuna >= 21 + 5


def test_split_levanta_se_nao_couber():
    ds = _dataset(60, seed=5, forca_do_sinal=0.5)
    with pytest.raises(ValueError, match="pequeno demais"):
        capacity._split_fixo(ds, test_size=50, train_size=None, horizon=21, embargo_days=5)


# ---------------------------------------------------------------------------
# Teto em amostra
# ---------------------------------------------------------------------------


def test_teto_alto_quando_ha_relacao():
    ds = _dataset(1200, seed=6, forca_do_sinal=1.0)
    teto = capacity.in_sample_ceiling(ds, FEATURES, horizon=21, test_size=300)
    assert teto["r2_oos"] > 0.9


def test_teto_baixo_quando_o_alvo_e_ruido():
    """O diagnostico decisivo: se nem colando o modelo explica o alvo, nao ha
    o que nenhuma variante honesta possa extrair."""
    ds = _dataset(1200, seed=7, forca_do_sinal=0.0)
    teto = capacity.in_sample_ceiling(ds, FEATURES, horizon=21, test_size=300)
    assert teto["r2_oos"] < 0.10


# ---------------------------------------------------------------------------
# Varredura de capacidade
# ---------------------------------------------------------------------------


def test_capacidade_maior_memoriza_mas_nao_generaliza_em_ruido():
    """O CONTRASTE QUE DECIDE. Em dado sem sinal, aumentar capacidade faz o R2
    DENTRO da amostra subir muito e o de FORA nao acompanhar. Isso e memorizar
    ruido, nao 'treinar pouco'."""
    ds = _dataset(1200, seed=8, forca_do_sinal=0.0)
    grid = [
        {"rotulo": "rasa", "max_depth": 2, "n_estimators": 100},
        {"rotulo": "funda", "max_depth": 12, "n_estimators": 800},
    ]
    r = capacity.capacity_sweep(ds, FEATURES, horizon=21, test_size=300, grid=grid)

    assert r["r2_dentro"].iloc[-1] > r["r2_dentro"].iloc[0] + 0.3  # memoriza mais
    assert r["r2_fora"].iloc[-1] <= r["r2_fora"].iloc[0] + 0.05    # e nao generaliza


def test_varredura_reporta_dentro_e_fora():
    ds = _dataset(900, seed=9, forca_do_sinal=0.7)
    grid = [{"rotulo": "rasa", "max_depth": 2, "n_estimators": 50}]
    r = capacity.capacity_sweep(ds, FEATURES, horizon=21, test_size=250, grid=grid)
    assert {"r2_dentro", "r2_fora", "capacidade"}.issubset(r.columns)


# ---------------------------------------------------------------------------
# Autocorrelacao do alvo
# ---------------------------------------------------------------------------


def test_autocorrelacao_detecta_memoria_e_sua_ausencia():
    idx = pd.date_range("2020-01-01", periods=1000, freq="B")
    rng = np.random.default_rng(10)

    ruido = pd.Series(rng.normal(0, 1, 1000), index=idx)
    persistente = ruido.rolling(30, min_periods=1).mean()

    a_ruido = capacity.target_autocorrelation(ruido, [1, 21])
    a_pers = capacity.target_autocorrelation(persistente, [1, 21])

    assert abs(a_ruido["autocorrelacao"].iloc[1]) < 0.1
    assert a_pers["autocorrelacao"].iloc[1] > 0.2