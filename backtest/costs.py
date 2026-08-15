"""Custos de transacao para operacoes de opcao de dolar.

CLAUDE.md: opcao de dolar tem spread bid-ask largo -- incluir custos e
slippage em TODO backtest, sempre. Sem book (bid/ask) coletado ainda, o
custo e aproximado como uma fracao do premio (`spread_pct`), cobrada toda
vez que a posicao muda (abertura OU fechamento) -- simplificacao explicita
e deliberadamente conservadora (default 5% do premio).
"""

from __future__ import annotations


def transaction_cost(premium: float, n_contracts: float, spread_pct: float = 0.05) -> float:
    """Custo de uma unica transacao (abrir OU fechar posicao): metade do
    spread bid-ask (aproximado por `spread_pct` do premio) vezes o notional
    operado (|n_contracts| x premio).
    """
    return abs(n_contracts) * premium * spread_pct


def round_trip_cost(
    premium_open: float, premium_close: float, n_contracts: float, spread_pct: float = 0.05
) -> float:
    """Custo total de abrir E depois fechar a mesma posicao."""
    return transaction_cost(premium_open, n_contracts, spread_pct) + transaction_cost(
        premium_close, n_contracts, spread_pct
    )


# Spread do FUTURO de dolar, nao da opcao. Ordem de grandeza deliberadamente
# diferente: o DOL e um dos futuros mais liquidos da B3 (spread tipico de 0,5
# a 1 ponto num preco de ~5.000, isto e ~1-2 bp), enquanto a opcao de dolar
# quase nao negocia -- a propria coleta deste projeto achou 3 series com
# negocio por pregao em 2026. Usar os 5% da opcao para custear o hedge
# inviabilizaria artificialmente qualquer rebalanceamento diario.
FUTURES_SPREAD_PCT = 0.0001


def futures_hedge_cost(
    delta_contracts: float, futures_price: float, spread_pct: float = FUTURES_SPREAD_PCT
) -> float:
    """Custo de ajustar a posicao de hedge no futuro em `delta_contracts`
    contratos (positivo ou negativo -- o custo depende do |tamanho| girado).

    Cobrado a cada rebalanceamento do delta-hedge. E o custo que decide se
    hedgear diariamente compensa: hedge mais frequente reduz a variancia do
    P&L mas paga spread toda vez.
    """
    return abs(delta_contracts) * futures_price * spread_pct
