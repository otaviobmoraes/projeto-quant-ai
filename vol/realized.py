"""Volatilidade realizada (RV) do USD/BRL a partir da serie de PTAX ja coletada.

Separacao I/O (le o parquet processado por data/ptax.py) vs logica pura
(retornos e RV rolante), para poder testar a matematica sem tocar disco.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.b3_futures import PROCESSED_PATH as B3_FUTURES_PROCESSED_PATH
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


def forward_realized_skewness(
    returns: pd.Series, horizon: int, min_periods: int | None = None
) -> pd.Series:
    """Assimetria (skewness) dos retornos nos `horizon` dias APOS cada data.

    Contraparte REALIZADA do skew implicito. Motivacao (credibility/CLAUDE.md):
    a teoria de Barro-Gordon preve que perda de credibilidade eleva
    ASSIMETRICAMENTE o risco de depreciacao do real -- ou seja, deveria
    alargar a cauda DIREITA da distribuicao de USD/BRL, nao apenas o nivel
    de volatilidade. Testar theta_t contra RV futura (um alvo de NIVEL) nao
    responde a essa previsao; testar contra assimetria futura, sim.

    O teste ideal usaria skew IMPLICITO (da superficie de opcoes), mas a B3
    sobrescreve a superficie diariamente e so ha 1 dia de historico -- ver
    limitacoes no relatorio. Esta e a melhor aproximacao possivel com os
    dados disponiveis.

    Olha para frente por construcao (shift negativo): serve como ALVO
    supervisionado, nunca como feature de entrada.

    `min_periods`: quantos retornos validos a janela precisa ter. Importa
    quando a serie tem buracos deliberados -- e o caso do futuro de dolar,
    onde o retorno do dia de ROLAGEM e descartado (emenda entre contratos
    diferentes nao e movimento de preco real). Como as rolagens sao MENSAIS,
    praticamente toda janela de 21 dias contem uma; com o padrao do pandas
    (min_periods = horizon) quase tudo viraria NaN. Default aqui: tolera ate
    3 ausencias na janela.
    """
    if min_periods is None:
        min_periods = max(3, horizon - 3)
    # rolling ANTES do shift (mesma ordem de forward_target_from_variance):
    # a ordem inversa (shift antes) daria a mesma janela, mas descartaria os
    # primeiros horizon-1 valores sem necessidade.
    fwd = returns.rolling(horizon, min_periods=min_periods).skew().shift(-horizon)
    return fwd.rename("forward_skewness")


def load_b3_futures_prices_and_variance() -> tuple[pd.Series, pd.Series]:
    """Le o futuro de dolar da B3 ja coletado (data/b3_futures.py) e devolve
    (preco de ajuste, variancia diaria de Parkinson).

    Fonte PREFERENCIAL do ponto de vista economico: a estrategia negocia
    opcao SOBRE ESSE FUTURO (Black-76), entao a RV relevante e a do futuro,
    nao a do spot. Alem disso o alinhamento de datas foi validado contra o
    PTAX (corr 0.768 no mesmo dia), enquanto o `close` do yfinance so alinha
    com defasagem de 1 dia -- ver data/b3_futures.py.

    O preco de ajuste vem cotado em BRL por 1000 USD; e dividido por 1000
    pra ficar na mesma escala do spot/PTAX. Parkinson usa ln(high/low), que
    e invariante a escala -- a divisao nao afeta a variancia.

    A serie troca de contrato a cada mes (rolagem). Parkinson e calculado
    DENTRO de cada dia (high/low do mesmo pregao), entao e imune a emenda --
    diferente de retorno de fechamento, que exigiria descartar o dia da
    virada (coluna `contract_changed`).
    """
    if not B3_FUTURES_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{B3_FUTURES_PROCESSED_PATH} nao encontrado -- rode "
            "data.b3_futures.load_dol_futures_processed(...) primeiro."
        )
    fut = pd.read_parquet(B3_FUTURES_PROCESSED_PATH).set_index("date").sort_index()
    variance = parkinson_daily_variance(fut["high"], fut["low"])
    return fut["settlement"] / 1000.0, variance


PRICE_SOURCES = ("ptax", "yfinance", "b3")


def load_prices_and_variance(
    source: str = "b3", tipo: str = "venda"
) -> tuple[pd.Series, pd.Series | None]:
    """Despachante unico de fonte de preco -> (precos, variancia diaria).

    Existe para que as ablacoes (backtest/, credibility/) escolham a fonte
    por um parametro, em vez de repetirem o mesmo if/else. Fontes:

    - "b3"       : futuro de dolar da B3, variancia de Parkinson. PADRAO e
                   fonte PREFERENCIAL -- e o instrumento sobre o qual a opcao
                   e escrita, com preco oficial da bolsa e datas validadas
                   contra o PTAX (ver data/b3_futures.py).
    - "yfinance" : spot BRL=X, variancia de Parkinson. Mantida so para
                   reproduzir resultados historicos do projeto; as barras
                   tem abertura ~= fechamento e o `close` fica defasado 1 dia
                   (o high/low, usado pelo Parkinson, esta correto).
    - "ptax"     : PTAX do BCB, sem OHLC -- devolve variancia None, deixando
                   o chamador cair no proxy de retorno de fechamento ao
                   quadrado.
    """
    if source not in PRICE_SOURCES:
        raise ValueError(f"fonte desconhecida: {source!r} -- use uma de {PRICE_SOURCES}")

    if source == "b3":
        return load_b3_futures_prices_and_variance()
    if source == "yfinance":
        return load_parkinson_prices_and_variance()

    if not PTAX_PROCESSED_PATH.exists():
        raise FileNotFoundError(f"{PTAX_PROCESSED_PATH} nao encontrado -- rode data.ptax primeiro.")
    ptax_df = pd.read_parquet(PTAX_PROCESSED_PATH)
    prices = ptax_df[ptax_df["tipo"] == tipo].set_index("date")["value"].sort_index()
    return prices, None


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
