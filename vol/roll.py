"""Correcao do ciclo de rolagem do futuro de dolar da B3.

MOTIVACAO (diagnostico que originou este modulo): a serie oficial de preco do
projeto e o contrato MAIS LIQUIDO de cada dia (data/b3_futures.front_month_series),
que troca a cada ~20,2 pregoes -- praticamente o mesmo periodo do horizonte alvo
da estrategia (21 dias). Medindo a vol de Parkinson por posicao no ciclo do
contrato, aparece um padrao monotonico: ~10,9% de vol anualizada num contrato
recem-promovido contra ~9,5% num contrato prestes a ser rolado (queda de ~15%).

O sentido do padrao e o INVERSO do efeito Samuelson (que preveria vol SUBINDO
perto do vencimento), o que sugere artefato de medicao e nao fenomeno economico:
conforme a liquidez migra para o vencimento seguinte, menos negocios exploram o
range intradiario, e o ln(high/low) do estimador de Parkinson encolhe
mecanicamente.

Consequencia testavel: isso injeta no alvo um "dente de serra" deterministico
com periodo ~= o horizonte de previsao, que o HAR-RV nao tem nenhuma feature
para capturar -- candidato a contribuir para o R2 negativo em h=21 documentado
no relatorio (Secao 5.7).

SEM LOOK-AHEAD: o vencimento e derivado do TICKER (DOLK23 -> maio/2023), nao da
data observada de rolagem. O calendario de vencimentos da B3 e publicado com
anos de antecedencia, entao "dias ate o vencimento" e conhecido ex-ante em
qualquer data. O fator sazonal, quando usado em previsao, precisa ser estimado
SOMENTE no periodo de treino de cada fold -- ver `seasonal_factor`.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# Codigos de mes de futuros (padrao CME/B3).
MONTH_CODES: dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

# Faixas de dias corridos ate o vencimento usadas para estimar o fator sazonal.
# Escolhidas ANTES de olhar o resultado da ablacao (pre-registro): 5 faixas
# aproximadamente equipopulosas dado o ciclo de ~30 dias corridos do contrato.
DEFAULT_BUCKET_EDGES: tuple[int, ...] = (0, 7, 14, 21, 28)


def contract_expiry(ticker: str) -> date:
    """Vencimento do futuro de dolar a partir do ticker (ex.: "DOLK23" ->
    2023-05-01, primeiro dia util de maio/2023).

    O futuro de dolar da B3 vence no PRIMEIRO DIA UTIL do mes do contrato.
    Aproximamos "dia util" por "primeiro dia de semana" -- sem calendario de
    feriados, o erro e de 1-2 dias em poucos contratos (quando o dia 1 cai em
    feriado), imaterial para agrupar em faixas de ~7 dias.
    """
    if len(ticker) < 5 or ticker[3] not in MONTH_CODES:
        raise ValueError(f"ticker de futuro de dolar nao reconhecido: {ticker!r}")

    month = MONTH_CODES[ticker[3]]
    year = 2000 + int(ticker[4:])

    day = date(year, month, 1)
    while day.weekday() >= 5:  # 5=sabado, 6=domingo
        day = date(year, month, day.day + 1)
    return day


def days_to_expiry(dates: pd.Series, tickers: pd.Series) -> pd.Series:
    """Dias CORRIDOS entre cada data e o vencimento do contrato daquele dia.

    Dias corridos (nao uteis) de proposito: dispensa calendario de feriados,
    e a variavel so serve para posicionar a observacao dentro do ciclo do
    contrato, onde a resolucao de 1 dia e irrelevante.
    """
    expiries = tickers.map(contract_expiry)
    naive = pd.to_datetime(pd.Series(dates).to_numpy(), utc=True).tz_localize(None)
    delta = pd.to_datetime(expiries.to_numpy()) - pd.to_datetime(naive.date)
    return pd.Series(delta.days, index=pd.Series(dates).index, name="days_to_expiry")


def bucket_days_to_expiry(
    dte: pd.Series, edges: tuple[int, ...] = DEFAULT_BUCKET_EDGES
) -> pd.Series:
    """Discretiza `days_to_expiry` em faixas (indice inteiro da faixa).

    Faixas em vez do valor continuo porque o efeito nao e linear em dias e a
    amostra por dia individual e pequena (~27 observacoes por dia do ciclo).
    """
    return pd.Series(
        np.digitize(dte.to_numpy(), edges), index=dte.index, name="dte_bucket"
    )


def seasonal_factor(
    daily_variance: pd.Series, dte: pd.Series, edges: tuple[int, ...] = DEFAULT_BUCKET_EDGES
) -> pd.Series:
    """Fator MULTIPLICATIVO medio da variancia diaria por faixa de dias-ate-o-
    vencimento, normalizado para media geometrica 1 no periodo estimado.

    Estimado em ESPACO LOG (media geometrica): a variancia realizada e
    fortemente assimetrica a direita, entao a media aritmetica seria dominada
    por poucos dias de estresse e o fator refletiria esses outliers em vez do
    nivel tipico da faixa.

    IMPORTANTE -- uso em previsao: passe aqui SOMENTE o periodo de treino do
    fold. Estimar o fator na amostra completa e depois aplicar no teste e
    look-ahead (o fator carrega informacao sobre o nivel de vol do periodo de
    teste). Ver `backtest.roll_ablation`.

    Retorna uma Series indexada pelo indice da faixa, para ser aplicada via
    `deseasonalize`.
    """
    log_var = np.log(daily_variance)
    buckets = bucket_days_to_expiry(dte, edges)
    per_bucket = log_var.groupby(buckets).mean()
    return np.exp(per_bucket - log_var.mean()).rename("seasonal_factor")


def deseasonalize(
    daily_variance: pd.Series,
    dte: pd.Series,
    factor: pd.Series,
    edges: tuple[int, ...] = DEFAULT_BUCKET_EDGES,
) -> pd.Series:
    """Divide a variancia diaria pelo fator sazonal da faixa de cada dia.

    `factor`: saida de `seasonal_factor`, estimada no TREINO. Faixas ausentes
    em `factor` (possivel num treino curto) recebem fator 1,0 -- ou seja, nao
    corrigem nada, que e o comportamento conservador correto.
    """
    buckets = bucket_days_to_expiry(dte, edges)
    applied = buckets.map(factor).fillna(1.0).astype(float)
    return (daily_variance / applied.to_numpy()).rename("variance_deseasonalized")


def load_b3_futures_with_dte() -> tuple[pd.Series, pd.Series, pd.Series]:
    """I/O: le o futuro de dolar ja coletado (data/b3_futures.py) e devolve
    (precos de ajuste, variancia diaria de Parkinson, dias ate o vencimento),
    todos indexados por data -- as tres pecas que as ablacoes de rolagem
    precisam.
    """
    from data.b3_futures import PROCESSED_PATH
    from vol.realized import parkinson_daily_variance

    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} nao encontrado -- rode data.b3_futures primeiro."
        )

    fut = pd.read_parquet(PROCESSED_PATH).sort_values("date").reset_index(drop=True)
    dte = days_to_expiry(fut["date"], fut["ticker"])

    index = pd.DatetimeIndex(fut["date"])
    prices = pd.Series(fut["settlement"].to_numpy(), index=index, name="settlement")
    variance = pd.Series(
        parkinson_daily_variance(fut["high"], fut["low"]).to_numpy(),
        index=index,
        name="daily_variance",
    )
    return prices, variance, pd.Series(dte.to_numpy(), index=index, name="days_to_expiry")
