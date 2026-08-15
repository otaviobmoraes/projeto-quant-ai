"""Dimensionamento de posicao por vega (straddle ATM).

Em vez de operar um numero fixo de contratos sempre, o tamanho da posicao e
escolhido para entregar uma exposicao a vega alvo (`target_vega`) constante
em R$ por ponto percentual de vol -- assim o risco assumido nao varia so
porque a vol de mercado (e portanto o vega por contrato) esta alta ou baixa
naquele dia.

Estrutura inicial: straddle ATM (CLAUDE.md) -- strike = forward/spot do dia
(aproximacao: ainda nao ha curva de juros propria coletada para achar o
forward "de verdade" via paridade coberta de juros; usar o spot/PTAX como
proxy do forward e uma simplificacao explicita desta fase).
"""

from __future__ import annotations

import pandas as pd

from vol.black76 import straddle_vega


def straddle_vega_per_contract(spot: float, ttm_days: int, iv_pct: float, r: float = 0.0) -> float:
    """Vega (por ponto percentual de vol) de 1 straddle ATM (K = spot, usado
    como proxy do forward -- ver docstring do modulo).
    """
    T = ttm_days / 365
    sigma = iv_pct / 100
    return straddle_vega(F=spot, K=spot, T=T, sigma=sigma, r=r) * 0.01


def size_straddle(
    target_vega: float, spot: float, ttm_days: int, iv_pct: float, r: float = 0.0
) -> float:
    """Numero de straddles (pode ser fracionario -- arredondar/lotear fica a
    cargo de quem for executar) para atingir `target_vega` (R$ por ponto
    percentual de vol) de exposicao.
    """
    vega_per_contract = straddle_vega_per_contract(spot, ttm_days, iv_pct, r)
    return target_vega / vega_per_contract


def rv_dispersion(rv_trailing: pd.Series, window: int = 252) -> pd.Series:
    """"Vol da vol": desvio-padrao movel da RV corrente, em pontos percentuais.

    Estimador da INCERTEZA sobre onde a RV vai parar -- que e o risco de fato
    corrido por quem opera o spread RV vs IV. Usa somente janela PASSADA
    (`rolling`), entao pode ser calculado em t sem olhar o futuro.

    Janela de 252 pregoes (1 ano): longa o bastante para a estimativa nao ser
    dominada por ruido e curta o bastante para acompanhar mudanca de regime.
    Nao foi varrida -- varrer o parametro e a busca em grade que o CLAUDE.md
    proibe e que inflaria o contador do Deflated Sharpe.
    """
    return rv_trailing.rolling(window, min_periods=window // 2).std()


def size_by_risk_target(
    target_risk: float,
    spot: float,
    ttm_days: int,
    iv_pct: float,
    rv_dispersion_pct: float,
    r: float = 0.0,
    max_leverage: float = 5.0,
) -> float:
    """Numero de straddles para atingir um RISCO alvo constante, em vez de um
    VEGA alvo constante.

    A DIFERENCA, que e o ponto. `size_straddle` entrega exposicao constante a
    um ponto percentual de vol -- mas o RISCO de fato corrido e vega x quanto a
    vol pode se mover, e "quanto a vol pode se mover" varia muito no tempo. Com
    vega constante, a posicao carrega risco alto justamente quando a vol da vol
    esta alta, que e quando a venda de vol quebra. Aqui o tamanho e dividido
    pela dispersao esperada da RV, entao o risco fica constante e a posicao
    ENCOLHE em regime turbulento.

    P&L de um straddle delta-neutro ~ vega x (RV - IV) para diferencas
    pequenas, entao o desvio-padrao do P&L por contrato ~ vega x dispersao da
    RV -- e essa e a normalizacao aplicada.

    LITERATURA E RESSALVA HONESTA. Moreira & Muir (2017, Journal of Finance
    72(4)) e Harvey, Hoyle, Korgaonkar, Rattray, Sargaison & Van Hemert (2018,
    Journal of Portfolio Management 45(1):14-33) documentam que escalar pelo
    inverso da variancia recente aumenta o Sharpe. MAS Harvey et al. sao
    explicitos: o ganho de Sharpe vale para "risk assets" (acoes, credito), via
    efeito de alavancagem, e e DESPREZIVEL para moedas. A razao para esperar
    efeito aqui mesmo assim e que o objeto nao e a moeda, e uma carteira
    VENDIDA EM VOL, que tem a propriedade relevante por construcao: as perdas
    se concentram exatamente quando a vol sobe. O que Harvey et al. encontram
    em TODA classe de ativo -- reducao da severidade da cauda esquerda --
    tambem se aplica, e importa aqui (a assimetria do backtest e -1,60).
    Se o ganho nao aparecer, a leitura correta e que o caso "moeda" prevaleceu.

    `max_leverage`: teto de quantas vezes o tamanho pode exceder o do vega
    constante equivalente. Existe porque em periodo MUITO calmo a dispersao vai
    a quase zero e a formula pediria posicao ilimitada -- o que nao e resultado
    de modelo, e divisao por numero pequeno.
    """
    vega_por_contrato = straddle_vega_per_contract(spot, ttm_days, iv_pct, r)
    if rv_dispersion_pct <= 0:
        return target_risk / vega_por_contrato * max_leverage
    n = target_risk / (vega_por_contrato * rv_dispersion_pct)
    teto = target_risk / vega_por_contrato * max_leverage
    return min(n, teto)
