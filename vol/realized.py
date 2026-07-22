"""Volatilidade realizada (RV) do USD/BRL a partir da serie de PTAX ja coletada.

Separacao I/O (le o parquet processado por data/ptax.py) vs logica pura
(retornos e RV rolante), para poder testar a matematica sem tocar disco.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.ptax import PROCESSED_PATH as PTAX_PROCESSED_PATH

TRADING_DAYS_PER_YEAR = 252


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff()


def realized_vol(
    prices: pd.Series, window: int, annualization_factor: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """RV anualizada (close-to-close), em pontos percentuais, numa janela
    rolante de `window` dias uteis de retornos log.

    Usa dias uteis (nao corridos) porque a serie de entrada ja vem alinhada
    ao calendario de pregao da B3 (sem linhas em fins de semana/feriados).
    """
    returns = log_returns(prices)
    return returns.rolling(window).std() * np.sqrt(annualization_factor) * 100


def load_ptax_realized_vol(window: int = 21, tipo: str = "venda") -> pd.DataFrame:
    """Le o parquet processado do PTAX (ja coletado por data.ptax.load_ptax_processed)
    e monta a serie de RV realizada de `window` dias uteis, anualizada.

    Colunas: date (index), rv_pct.
    """
    if not PTAX_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PTAX_PROCESSED_PATH} nao encontrado -- rode "
            "data.ptax.load_ptax_processed(...) primeiro para coletar o PTAX."
        )
    df = pd.read_parquet(PTAX_PROCESSED_PATH)
    prices = df[df["tipo"] == tipo].set_index("date")["value"].sort_index()
    rv = realized_vol(prices, window)
    return rv.rename("rv_pct").to_frame()
