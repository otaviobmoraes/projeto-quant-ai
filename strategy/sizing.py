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
