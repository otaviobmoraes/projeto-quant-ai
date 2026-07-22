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
