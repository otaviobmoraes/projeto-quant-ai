"""Regra de sinal: comparar a previsao de RV (vol/forecast.py) com a IV de
mercado (vol/implied.py). Compra vol quando RV previsto > IV; vende vol
quando RV previsto < IV.

Guardrails do CLAUDE.md aplicados aqui:
- Sem look-ahead: a decisao de hoje (linha em t) so pode usar RV previsto e
  IV conhecidos ate o fechamento de hoje.
- Custos realistas: o spread bid-ask de opcao de dolar e largo, entao uma
  banda morta (`band_pct`) em torno de RV=IV evita operar em diferencas
  pequenas demais para pagar o custo de transacao.
"""

from __future__ import annotations

import pandas as pd

LONG_VOL = 1
SHORT_VOL = -1
NO_TRADE = 0


def generate_signal(rv_forecast_pct: float, iv_pct: float, band_pct: float = 0.0) -> int:
    """Sinal para um unico dia. `band_pct` e uma banda morta simetrica (em
    pontos percentuais de vol) em torno de RV=IV: dentro dela, NO_TRADE.
    """
    spread = rv_forecast_pct - iv_pct
    if spread > band_pct:
        return LONG_VOL
    if spread < -band_pct:
        return SHORT_VOL
    return NO_TRADE


def signal_series(rv_forecast: pd.Series, iv: pd.Series, band_pct: float = 0.0) -> pd.Series:
    """Versao vetorizada, alinhada por data (join interno -- so gera sinal
    onde ha RV previsto e IV para o mesmo dia).
    """
    aligned = pd.concat([rv_forecast.rename("rv_forecast"), iv.rename("iv")], axis=1, join="inner")
    return aligned.apply(
        lambda row: generate_signal(row["rv_forecast"], row["iv"], band_pct), axis=1
    ).rename("signal")
