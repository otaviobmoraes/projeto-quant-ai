"""Testes do delta-hedge: gregas do Black-76 (vol/black76.py) e do backtest
delta-neutro (backtest.engine.run_backtest_delta_hedged).

O teste que carrega o argumento e
`test_hedge_separa_volatilidade_de_direcao`: dois caminhos de preco com o
MESMO ponto final e volatilidade realizada muito diferente. Sem hedge os dois
dao o mesmo P&L (o backtest so olha |S_T - K|); com hedge eles se separam.
E a demonstracao de que o hedge muda o que esta sendo medido.
"""

import numpy as np
import pandas as pd
import pytest

from backtest import engine
from strategy.signal import SHORT_VOL
from vol import black76

# ---------------------------------------------------------------------------
# Gregas
# ---------------------------------------------------------------------------


def test_paridade_call_put_do_delta():
    """delta_call - delta_put = exp(-rT), a paridade no Black-76."""
    for r in (0.0, 0.05):
        dc = black76.call_delta(F=5.0, K=5.2, T=0.25, sigma=0.15, r=r)
        dp = black76.put_delta(F=5.0, K=5.2, T=0.25, sigma=0.15, r=r)
        assert dc - dp == pytest.approx(np.exp(-r * 0.25))


def test_delta_do_straddle_atm_e_quase_zero():
    """No dinheiro as duas pernas quase se cancelam -- e a razao de o straddle
    ser a estrutura natural pra operar nivel de vol."""
    d = black76.straddle_delta(F=5.0, K=5.0, T=21 / 365, sigma=0.12, r=0.0)
    assert abs(d) < 0.03


def test_delta_do_straddle_cresce_ao_sair_do_dinheiro():
    """Longe do strike o straddle vira aposta direcional: delta -> +1 (call
    fundo dentro) ou -1 (put fundo dentro)."""
    args = dict(K=5.0, T=21 / 365, sigma=0.12, r=0.0)
    assert black76.straddle_delta(F=7.0, **args) == pytest.approx(1.0, abs=1e-6)
    assert black76.straddle_delta(F=3.5, **args) == pytest.approx(-1.0, abs=1e-6)


def test_delta_bate_com_a_derivada_numerica_do_premio():
    F, K, T, sigma = 5.0, 5.1, 0.2, 0.14
    h = 1e-6
    pares = (
        (black76.call_price, black76.call_delta),
        (black76.put_price, black76.put_delta),
    )
    for pricer, greek in pares:
        numerica = (pricer(F + h, K, T, sigma) - pricer(F - h, K, T, sigma)) / (2 * h)
        assert greek(F, K, T, sigma) == pytest.approx(numerica, rel=1e-5)


def test_gamma_bate_com_a_derivada_numerica_do_delta():
    F, K, T, sigma = 5.0, 5.1, 0.2, 0.14
    h = 1e-5
    numerica = (
        black76.straddle_delta(F + h, K, T, sigma) - black76.straddle_delta(F - h, K, T, sigma)
    ) / (2 * h)
    assert black76.straddle_gamma(F, K, T, sigma) == pytest.approx(numerica, rel=1e-4)


# ---------------------------------------------------------------------------
# Backtest delta-neutro
# ---------------------------------------------------------------------------

HORIZON = 5


def _serie(valores: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(valores), freq="B")
    return pd.Series(valores, index=idx, dtype=float)


def _sinal_de_venda(prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    """RV prevista bem abaixo da IV -> sinal de VENDA de vol em todo dia."""
    rv_forecast = pd.Series(5.0, index=prices.index)
    iv = pd.Series(20.0, index=prices.index)
    return rv_forecast, iv


def test_preco_constante_zera_o_pnl_do_hedge():
    """Sem movimento no futuro nao ha nada a hedgear: o P&L do hedge e zero e
    a venda de straddle embolsa o premio inteiro (bruto)."""
    prices = _serie([5.0] * (HORIZON + 3))
    rv_forecast, iv = _sinal_de_venda(prices)

    trades = engine.run_backtest_delta_hedged(
        prices, rv_forecast, iv, horizon=HORIZON, band_pct=1.0, spread_pct=0.0,
        futures_spread_pct=0.0,
    )

    assert len(trades) == 1
    linha = trades.iloc[0]
    assert linha["signal"] == SHORT_VOL
    assert linha["pnl_hedge"] == pytest.approx(0.0, abs=1e-9)
    # payoff = 0 (preco nao mudou), entao a venda ganha exatamente o premio
    assert linha["pnl_gross"] == pytest.approx(linha["n_contracts"] * linha["premium"])


def test_hedge_separa_volatilidade_de_direcao():
    """O TESTE QUE JUSTIFICA O HEDGE.

    Dois caminhos com o MESMO preco inicial e o MESMO preco final:
      - `calmo`: sobe em linha reta, pouca vol realizada;
      - `agitado`: sobe, desaba e volta -- mesmo destino, muito mais vol.

    Sem hedge o P&L depende so de |S_T - K|, entao os dois sao IDENTICOS: o
    backtest nao consegue distinguir um mercado calmo de um agitado. Com
    delta-hedge o caminho agitado custa muito mais caro para quem esta VENDIDO
    em vol, que e o resultado economicamente correto.
    """
    calmo = _serie([5.00, 5.02, 5.04, 5.06, 5.08, 5.10, 5.10, 5.10])
    agitado = _serie([5.00, 5.40, 4.70, 5.35, 4.75, 5.10, 5.10, 5.10])
    assert calmo.iloc[0] == agitado.iloc[0]
    assert calmo.iloc[HORIZON] == agitado.iloc[HORIZON]

    sem_hedge, com_hedge = {}, {}
    for nome, prices in (("calmo", calmo), ("agitado", agitado)):
        rv_forecast, iv = _sinal_de_venda(prices)
        kwargs = dict(horizon=HORIZON, band_pct=1.0, spread_pct=0.0)
        sem_hedge[nome] = engine.run_backtest(prices, rv_forecast, iv, **kwargs).iloc[0]["pnl_net"]
        com_hedge[nome] = engine.run_backtest_delta_hedged(
            prices, rv_forecast, iv, futures_spread_pct=0.0, **kwargs
        ).iloc[0]["pnl_net"]

    # sem hedge: mesmo ponto final => mesmo resultado. O caminho e invisivel.
    assert sem_hedge["calmo"] == pytest.approx(sem_hedge["agitado"])

    # com hedge: o caminho agitado (vol realizada alta) pune a posicao vendida.
    assert com_hedge["agitado"] < com_hedge["calmo"]


def test_hedge_reduz_a_dispersao_do_pnl_entre_caminhos():
    """Propriedade central: hedgear reduz a variancia do P&L entre realizacoes
    diferentes do preco, porque remove a componente direcional."""
    pnl_sem, pnl_com = [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        caminho = 5.0 * np.exp(np.cumsum(rng.normal(0, 0.01, HORIZON + 3)))
        prices = _serie(list(caminho))
        rv_forecast, iv = _sinal_de_venda(prices)
        kwargs = dict(horizon=HORIZON, band_pct=1.0, spread_pct=0.0)
        pnl_sem.append(engine.run_backtest(prices, rv_forecast, iv, **kwargs).iloc[0]["pnl_net"])
        pnl_com.append(
            engine.run_backtest_delta_hedged(
                prices, rv_forecast, iv, futures_spread_pct=0.0, **kwargs
            ).iloc[0]["pnl_net"]
        )

    # reducao substancial, nao marginal
    assert np.std(pnl_com) < 0.7 * np.std(pnl_sem)


def test_rebalance_every_controla_o_numero_de_ajustes():
    prices = _serie(list(5.0 + np.linspace(0, 0.2, HORIZON + 3)))
    rv_forecast, iv = _sinal_de_venda(prices)
    kwargs = dict(horizon=HORIZON, band_pct=1.0, spread_pct=0.0, futures_spread_pct=0.0)

    diario = engine.run_backtest_delta_hedged(prices, rv_forecast, iv, rebalance_every=1, **kwargs)
    esparso = engine.run_backtest_delta_hedged(prices, rv_forecast, iv, rebalance_every=3, **kwargs)

    assert diario.iloc[0]["n_rebalances"] == HORIZON
    assert esparso.iloc[0]["n_rebalances"] < diario.iloc[0]["n_rebalances"]


def test_custo_do_hedge_reduz_o_pnl_liquido():
    prices = _serie(list(5.0 + np.linspace(0, 0.3, HORIZON + 3)))
    rv_forecast, iv = _sinal_de_venda(prices)
    kwargs = dict(horizon=HORIZON, band_pct=1.0, spread_pct=0.0)

    sem_custo = engine.run_backtest_delta_hedged(
        prices, rv_forecast, iv, futures_spread_pct=0.0, **kwargs
    ).iloc[0]
    com_custo = engine.run_backtest_delta_hedged(
        prices, rv_forecast, iv, futures_spread_pct=0.001, **kwargs
    ).iloc[0]

    assert sem_custo["hedge_cost"] == pytest.approx(0.0)
    assert com_custo["hedge_cost"] > 0
    assert com_custo["pnl_net"] < sem_custo["pnl_net"]
    # o bruto nao muda: custo nao e P&L
    assert com_custo["pnl_gross"] == pytest.approx(sem_custo["pnl_gross"])


def test_pnl_gross_e_soma_de_opcao_e_hedge():
    prices = _serie(list(5.0 + np.linspace(0, 0.25, HORIZON + 3)))
    rv_forecast, iv = _sinal_de_venda(prices)
    linha = engine.run_backtest_delta_hedged(
        prices, rv_forecast, iv, horizon=HORIZON, band_pct=1.0, spread_pct=0.0,
        futures_spread_pct=0.0,
    ).iloc[0]
    assert linha["pnl_gross"] == pytest.approx(linha["pnl_option"] + linha["pnl_hedge"])


def test_rebalance_every_invalido_levanta():
    prices = _serie([5.0] * (HORIZON + 3))
    rv_forecast, iv = _sinal_de_venda(prices)
    with pytest.raises(ValueError, match="rebalance_every"):
        engine.run_backtest_delta_hedged(
            prices, rv_forecast, iv, horizon=HORIZON, rebalance_every=0
        )