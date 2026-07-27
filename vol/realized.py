"""Volatilidade realizada (RV) do USD/BRL a partir da serie de PTAX ja coletada.

Separacao I/O (le o parquet processado por data/ptax.py) vs logica pura
(retornos e RV rolante), para poder testar a matematica sem tocar disco.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.fx_spot import PROCESSED_PATH as FX_SPOT_PROCESSED_PATH
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


def parkinson_daily_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    """Variancia diaria "instantanea" via estimador de Parkinson (1980),
    usando o range intradiario (High/Low) em vez do retorno de fechamento.

    RV_d = ln(High/Low)^2 / (4*ln2). E MUITO menos ruidoso que o quadrado do
    retorno de fechamento como proxy de variancia de 1 dia -- um unico
    retorno ao quadrado tem variancia do proprio estimador enorme (kurtose
    alta), enquanto o range intradiario usa mais informacao do dia (o
    caminho percorrido, nao so o ponto final). Efficiency teorica ~5x maior
    que o estimador close-to-close (Parkinson, 1980).

    Drop-in replacement pra `log_returns(prices)**2` em qualquer lugar que
    espere uma serie de variancia diaria (ex.: vol.forecast.build_dataset
    via o parametro `daily_variance`).
    """
    return (np.log(high / low) ** 2) / (4 * np.log(2))


def parkinson_vol(
    high: pd.Series, low: pd.Series, window: int, annualization_factor: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """RV anualizada (Parkinson), em pontos percentuais, numa janela rolante
    de `window` dias uteis de variancia diaria tipo Parkinson."""
    daily_var = parkinson_daily_variance(high, low)
    return np.sqrt(daily_var.rolling(window).mean() * annualization_factor) * 100


def load_parkinson_prices_and_variance() -> tuple[pd.Series, pd.Series]:
    """Le o fx_spot ja coletado (data/fx_spot.py, com OHLC) e devolve
    (close, daily_variance) usando o estimador de Parkinson em vez do proxy
    de retorno de fechamento -- ADOTADO como baseline preferencial: RMSE
    consistentemente menor em TODOS os folds testados no diagnostico do
    backtest (ver commit do diagnostico), nao so na media.
    """
    if not FX_SPOT_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{FX_SPOT_PROCESSED_PATH} nao encontrado -- rode data.fx_spot primeiro."
        )
    fx = pd.read_parquet(FX_SPOT_PROCESSED_PATH).set_index("date").sort_index()
    variance = parkinson_daily_variance(fx["high"], fx["low"])
    return fx["close"], variance


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
