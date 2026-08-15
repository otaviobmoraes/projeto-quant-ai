"""Testes do estimador de spread efetivo (data/option_spread.py).

O teste que carrega o argumento e `test_recupera_spread_conhecido_por_simulacao`:
simula negocios saltando entre ponta de compra e de venda com um spread
CONHECIDO e exige que o estimador o recupere. Sem isso, o numero medido seria
apenas plausivel, nao validado.
"""

import numpy as np
import pandas as pd
import pytest

from data import option_spread as osp


def _ranges(linhas: list[dict]) -> pd.DataFrame:
    base = {"date": pd.Timestamp("2024-01-02"), "ticker": "DOLF24C005000",
            "maturity": "F24", "option_type": "C", "strike": 5000.0}
    return pd.DataFrame([{**base, **linha} for linha in linhas])


# ---------------------------------------------------------------------------
# Correcao de poucos negocios
# ---------------------------------------------------------------------------


def test_probabilidade_de_revelar_o_spread():
    f = osp.reveal_probability(np.array([2, 3, 4, 10]))
    assert f[0] == pytest.approx(0.50)
    assert f[1] == pytest.approx(0.75)
    assert f[2] == pytest.approx(0.875)
    assert f[3] == pytest.approx(1 - 2**-9)


def test_probabilidade_cresce_e_satura_em_um():
    f = osp.reveal_probability(np.arange(2, 40))
    assert np.all(np.diff(f) > 0)
    assert f[-1] == pytest.approx(1.0, abs=1e-6)


def test_series_com_um_negocio_sao_descartadas():
    """Com 1 negocio min==max por construcao: isso e 'nao medido', nao
    'spread zero'. Incluir zeraria a mediana."""
    r = _ranges([
        {"price_min": 10.0, "price_max": 10.0, "price_avg": 10.0, "n_trades": 1},
        {"price_min": 9.5, "price_max": 10.5, "price_avg": 10.0, "n_trades": 4},
    ])
    out = osp.effective_spread(r)
    assert len(out) == 1
    assert out.iloc[0]["n_trades"] == 4


# ---------------------------------------------------------------------------
# Validacao por simulacao -- o teste central
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spread_verdadeiro", [0.02, 0.05, 0.10])
def test_recupera_spread_conhecido_por_simulacao(spread_verdadeiro):
    """Simula series cujos negocios saltam entre bid e ask com spread
    CONHECIDO (sem movimento do subjacente) e verifica que a mediana estimada
    bate com o valor verdadeiro."""
    rng = np.random.default_rng(7)
    mid = 20.0
    meia = spread_verdadeiro * mid  # meia-amplitude em unidades de preco
    linhas = []
    for _ in range(4000):
        n = int(rng.integers(2, 9))
        lados = rng.integers(0, 2, n)
        precos = mid + np.where(lados == 1, meia, -meia)
        linhas.append({
            "price_min": precos.min(), "price_max": precos.max(),
            "price_avg": precos.mean(), "n_trades": float(n),
        })
    out = osp.effective_spread(_ranges(linhas))
    resumo = osp.summarize_spread(out)
    assert resumo["mediana"] == pytest.approx(spread_verdadeiro, rel=0.15)


def test_no_pior_caso_de_dois_negocios_ainda_recupera():
    """N=2 e o pior caso: metade das series nao revela spread nenhum. As que
    revelam ainda entregam o spread EXATO, e e por isso que a estimacao
    condiciona em vez de corrigir."""
    rng = np.random.default_rng(11)
    mid, meia = 20.0, 0.05 * 20.0
    linhas = []
    for _ in range(3000):
        precos = mid + np.where(rng.integers(0, 2, 2) == 1, meia, -meia)
        linhas.append({"price_min": precos.min(), "price_max": precos.max(),
                       "price_avg": precos.mean(), "n_trades": 2.0})

    out = osp.effective_spread(_ranges(linhas))
    # ~metade das series e descartada por amplitude nula...
    assert 0.4 < len(out) / 3000 < 0.6
    # ...e as que sobram dao o spread verdadeiro, sem fator nenhum
    assert osp.summarize_spread(out)["mediana"] == pytest.approx(0.05, rel=0.02)


def test_aplicar_o_fator_junto_com_o_filtro_dobraria_o_spread():
    """REGRESSAO do erro que os testes pegaram: usar reveal_probability na
    estimacao, junto com o descarte de amplitude nula, conta o mesmo efeito
    duas vezes e dobra o resultado em N=2."""
    rng = np.random.default_rng(11)
    mid, meia = 20.0, 0.05 * 20.0
    linhas = []
    for _ in range(2000):
        precos = mid + np.where(rng.integers(0, 2, 2) == 1, meia, -meia)
        linhas.append({"price_min": precos.min(), "price_max": precos.max(),
                       "price_avg": precos.mean(), "n_trades": 2.0})

    out = osp.effective_spread(_ranges(linhas))
    correto = osp.summarize_spread(out)["mediana"]
    errado = float(
        (out["spread_pct"] / osp.reveal_probability(out["n_trades"])).median()
    )
    assert correto == pytest.approx(0.05, rel=0.02)
    assert errado == pytest.approx(0.10, rel=0.02)  # exatamente o dobro


# ---------------------------------------------------------------------------
# Desconto do movimento do subjacente
# ---------------------------------------------------------------------------


def test_movimento_do_subjacente_e_descontado():
    """Parte da amplitude vem do futuro andar, nao do spread. Descontar tem
    que REDUZIR o spread estimado."""
    r = _ranges([{"price_min": 9.0, "price_max": 11.0, "price_avg": 10.0, "n_trades": 6.0}])
    sem = osp.effective_spread(r)
    com = osp.effective_spread(
        r,
        underlying_range=pd.Series({pd.Timestamp("2024-01-02"): 2.0}),
        deltas=pd.Series([0.5], index=r.index),
    )
    # amplitude 2.0, movimento esperado 0.5*2.0 = 1.0 => sobra 1.0 de spread
    assert com.iloc[0]["spread_pct"] == pytest.approx(0.05)
    assert sem.iloc[0]["spread_pct"] == pytest.approx(0.10)


def test_desconto_nao_produz_spread_negativo():
    r = _ranges([{"price_min": 9.9, "price_max": 10.1, "price_avg": 10.0, "n_trades": 5.0}])
    com = osp.effective_spread(
        r,
        underlying_range=pd.Series({pd.Timestamp("2024-01-02"): 1000.0}),
        deltas=pd.Series([0.9], index=r.index),
    )
    assert com.empty or (com["spread_pct"] >= 0).all()


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------


def test_resumo_usa_mediana_e_reporta_dispersao():
    rng = np.random.default_rng(3)
    linhas = [{"price_min": 10 - d, "price_max": 10 + d, "price_avg": 10.0, "n_trades": 8.0}
              for d in rng.uniform(0.1, 1.0, 500)]
    resumo = osp.summarize_spread(osp.effective_spread(_ranges(linhas)))
    assert resumo["n"] == 500
    assert resumo["p25"] < resumo["mediana"] < resumo["p75"] < resumo["p90"]


def test_regressao_separa_spread_de_movimento_do_subjacente():
    """Simula amplitude = spread FIXO + parcela proporcional ao movimento do
    futuro, e exige que a regressao recupere os dois."""
    rng = np.random.default_rng(21)
    n, mid = 600, 20.0
    spread_total, beta = 1.0, 0.4  # 1.0 de spread (5% do premio total), 40% da faixa
    datas = pd.date_range("2024-01-02", periods=n, freq="B")
    faixa_fut = pd.Series(rng.uniform(10, 60, n), index=datas)
    delta = 0.5

    linhas = []
    for d in datas:
        amp = spread_total + beta * delta * faixa_fut[d] * (mid / 100) / 10
        linhas.append({"date": d, "ticker": "X", "maturity": "F24", "option_type": "C",
                       "strike": 5000.0, "price_min": mid - amp / 2,
                       "price_max": mid + amp / 2, "price_avg": mid, "n_trades": 4.0})
    r = pd.DataFrame(linhas)

    out = osp.spread_by_regression(
        r, underlying_range=faixa_fut * (mid / 100) / 10 / 1.0,
        deltas=pd.Series(delta, index=r.index),
    )
    assert out["spread_total_pct"] == pytest.approx(spread_total / mid, rel=0.05)
    assert out["beta_movimento"] == pytest.approx(beta, rel=0.10)
    assert out["spread_pct"] == pytest.approx(spread_total / mid / 2, rel=0.05)


def test_regressao_devolve_so_n_com_amostra_minuscula():
    r = _ranges([{"price_min": 9.0, "price_max": 11.0, "price_avg": 10.0, "n_trades": 3.0}])
    out = osp.spread_by_regression(
        r, underlying_range=pd.Series({pd.Timestamp("2024-01-02"): 10.0}),
        deltas=pd.Series([0.5], index=r.index),
    )
    assert out == {"n": 1}


def test_resumo_de_amostra_vazia():
    assert osp.summarize_spread(pd.DataFrame()) == {"n": 0}


def test_descarta_amplitude_absurda():
    r = _ranges([
        {"price_min": 0.01, "price_max": 90.0, "price_avg": 10.0, "n_trades": 3.0},
        {"price_min": 9.5, "price_max": 10.5, "price_avg": 10.0, "n_trades": 3.0},
    ])
    assert len(osp.effective_spread(r)) == 1