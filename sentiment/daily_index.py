"""Indice diario de sentimento/estresse a partir dos corpora BR ja coletados
(Copom em data/copom.py, manchetes em data/rss_news.py), pontuados pelo
FinBERT-PT-BR (sentiment/finbert.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from data.copom import PROCESSED_PATH as COPOM_PROCESSED_PATH
from data.gdelt_news import FISCAL_RISK_QUERY, HEADLINES_PROCESSED_PATH
from data.rss_news import PROCESSED_PATH as RSS_PROCESSED_PATH
from sentiment.finbert import score_texts

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "sentiment_index.parquet"
FISCAL_SENTIMENT_PROCESSED_PATH = PROCESSED_DIR / "fiscal_sentiment_index.parquet"


def daily_sentiment_index(
    df: pd.DataFrame,
    date_col: str,
    text_col: str,
    pipeline_fn: Callable[[list[str]], list[dict]] | None = None,
) -> pd.DataFrame:
    """Pontua cada texto de `df` e agrega por dia.

    Colunas: date, sentiment_mean (media do signed_score, indice de
    sentimento), share_negative (fracao de textos NEGATIVE, proxy de
    estresse), n_textos.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "sentiment_mean", "share_negative", "n_textos"])

    scored = score_texts(df[text_col].tolist(), pipeline_fn=pipeline_fn)
    scored["date"] = pd.to_datetime(df[date_col]).dt.date.values

    daily = (
        scored.groupby("date")
        .agg(
            sentiment_mean=("signed_score", "mean"),
            share_negative=("label", lambda s: (s == "NEGATIVE").mean()),
            n_textos=("label", "count"),
        )
        .reset_index()
    )
    return daily


def load_combined_daily_sentiment(
    pipeline_fn: Callable[[list[str]], list[dict]] | None = None,
) -> pd.DataFrame:
    """Le os corpora BR ja coletados (Copom + RSS), pontua com o FinBERT-PT-BR
    e monta o indice diario combinado, salvando o parquet processado.
    """
    frames = []
    if COPOM_PROCESSED_PATH.exists():
        copom_df = pd.read_parquet(COPOM_PROCESSED_PATH)[["data_referencia", "texto"]]
        copom_df = copom_df.rename(columns={"data_referencia": "date", "texto": "text"})
        frames.append(copom_df)
    if RSS_PROCESSED_PATH.exists():
        rss_df = pd.read_parquet(RSS_PROCESSED_PATH)[["date", "title"]]
        rss_df = rss_df.rename(columns={"title": "text"})
        frames.append(rss_df)

    if not frames:
        raise FileNotFoundError(
            f"Nem {COPOM_PROCESSED_PATH} nem {RSS_PROCESSED_PATH} existem -- "
            "rode os coletores de data.copom / data.rss_news primeiro."
        )

    combined = pd.concat(frames, ignore_index=True)
    daily = daily_sentiment_index(combined, date_col="date", text_col="text", pipeline_fn=pipeline_fn)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(PROCESSED_PATH, index=False)
    return daily


def load_fiscal_risk_sentiment_index(
    query: str | None = None,
    language: str = "Portuguese",
    pipeline_fn: Callable[[list[str]], list[dict]] | None = None,
) -> pd.DataFrame:
    """Le as manchetes de risco fiscal ja coletadas (data.gdelt_news,
    load_gdelt_headlines_processed_chunked), pontua com o FinBERT-PT-BR e
    monta o indice diario de sentimento/estresse fiscal.

    `language="Portuguese"`: FILTRO OBRIGATORIO -- FinBERT-PT-BR so serve
    para texto em portugues. A query de risco fiscal usa palavras-chave em
    ingles ("fiscal", "budget", ...) e o GDELT monitora cobertura GLOBAL: uma
    amostra real (checada manualmente) trouxe so ~32% das manchetes em
    portugues, o resto espalhado entre ingles, espanhol e mais de 8 outros
    idiomas. Aplicar o modelo ao lote inteiro sem filtrar daria pontuacao de
    sentimento sem sentido pra maior parte dos textos.
    """
    if not HEADLINES_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{HEADLINES_PROCESSED_PATH} nao encontrado -- rode "
            "data.gdelt_news.load_gdelt_headlines_processed_chunked primeiro."
        )
    query = query or FISCAL_RISK_QUERY
    headlines = pd.read_parquet(HEADLINES_PROCESSED_PATH)
    headlines = headlines[(headlines["query"] == query) & (headlines["language"] == language)]
    if headlines.empty:
        raise ValueError(
            f"nenhuma manchete em '{language}' encontrada para a query de risco fiscal -- "
            "verifique se a coleta ja rodou e se ha cobertura nesse idioma."
        )

    daily = daily_sentiment_index(headlines, date_col="date", text_col="title", pipeline_fn=pipeline_fn)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(FISCAL_SENTIMENT_PROCESSED_PATH, index=False)
    return daily
