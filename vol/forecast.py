"""Previsao de RV: HAR-RV baseline + versao aumentada com noticia (tom do
GDELT), com o estudo de ablacao obrigatorio do CLAUDE.md (toda avaliacao de
RV compara com noticia vs baseline sem noticia -- prova causal do valor da IA).

Sem dados intraday, a variancia realizada diaria e aproximada pelo retorno
log diario ao quadrado -- a simplificacao padrao do HAR-RV (Corsi, 2009)
quando so ha preco de fechamento.

O split cronologico usado aqui e uma avaliacao PRELIMINAR fora da amostra,
so para a ablacao fazer sentido sem o pior tipo de look-ahead. O walk-forward
purgado com embargo (Lopez de Prado) fica para backtest/, que e onde a
avaliacao final e rigorosa de fato acontece.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data.gdelt_news import TONE_PROCESSED_PATH
from data.ptax import PROCESSED_PATH as PTAX_PROCESSED_PATH
from vol.realized import TRADING_DAYS_PER_YEAR, log_returns

BASELINE_FEATURES = ["rv_d", "rv_w", "rv_m"]
NEWS_FEATURES = BASELINE_FEATURES + ["news"]


def har_features(prices: pd.Series) -> pd.DataFrame:
    """Features HAR-RV (diaria, semanal=5d, mensal=22d) de variancia realizada
    diaria (proxy: retorno log diario ao quadrado). Cada linha usa somente
    dados ate aquele dia (sem look-ahead).
    """
    r2 = log_returns(prices) ** 2
    df = pd.DataFrame(index=prices.index)
    df["rv_d"] = r2
    df["rv_w"] = r2.rolling(5).mean()
    df["rv_m"] = r2.rolling(22).mean()
    return df


def forward_target(prices: pd.Series, horizon: int) -> pd.Series:
    """RV anualizada realizada nos `horizon` dias APOS cada data -- o alvo a
    prever. Olha para frente por construcao (shift negativo): usado somente
    para montar o dataset supervisionado, nunca como feature de entrada.
    """
    r2 = log_returns(prices) ** 2
    fwd_var = r2.rolling(horizon).mean().shift(-horizon)
    return np.sqrt(fwd_var * TRADING_DAYS_PER_YEAR) * 100


def build_dataset(prices: pd.Series, news: pd.Series | None = None, horizon: int = 21) -> pd.DataFrame:
    """Monta o dataset supervisionado: features HAR (+ noticia, se fornecida)
    e o alvo (RV futura de `horizon` dias). A serie de noticia e alinhada por
    ultimo valor conhecido (ffill) -- ela pode nao ter uma observacao exata em
    todo dia util, mas nunca usa informacao futura.
    """
    df = har_features(prices)
    df["target"] = forward_target(prices, horizon)
    if news is not None:
        df["news"] = news.reindex(df.index, method="ffill")
    return df.dropna()


def chronological_split(dataset: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cronologico simples (sem embaralhar, treino sempre antes do teste
    no tempo). Avaliacao preliminar -- o walk-forward purgado fica em backtest/.
    """
    n_test = max(1, int(len(dataset) * test_size))
    return dataset.iloc[:-n_test], dataset.iloc[-n_test:]


def fit_har(train: pd.DataFrame, feature_cols: list[str]):
    X = sm.add_constant(train[feature_cols])
    y = train["target"]
    return sm.OLS(y, X).fit()


def evaluate(model, test: pd.DataFrame, feature_cols: list[str]) -> dict:
    X_test = sm.add_constant(test[feature_cols], has_constant="add")
    pred = model.predict(X_test)
    err = test["target"].to_numpy() - pred.to_numpy()
    ss_res = float((err**2).sum())
    ss_tot = float(((test["target"] - test["target"].mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2_oos": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def run_ablation(
    prices: pd.Series, news: pd.Series, horizon: int = 21, test_size: float = 0.2
) -> dict:
    """Ajusta o HAR-RV baseline e a versao com noticia (mesmo split de
    treino/teste para os dois) e devolve as metricas fora da amostra lado a
    lado -- a comparacao com noticia vs sem noticia exigida pelo CLAUDE.md.
    """
    dataset = build_dataset(prices, news=news, horizon=horizon)
    train, test = chronological_split(dataset, test_size)

    baseline_model = fit_har(train, BASELINE_FEATURES)
    news_model = fit_har(train, NEWS_FEATURES)

    return {
        "baseline": evaluate(baseline_model, test, BASELINE_FEATURES),
        "com_noticia": evaluate(news_model, test, NEWS_FEATURES),
        "n_train": len(train),
        "n_test": len(test),
    }


def load_and_run_ablation(
    horizon: int = 21, test_size: float = 0.2, tipo: str = "venda", query: str | None = None
) -> dict:
    """Le o PTAX e o tom do GDELT ja coletados (data/ptax.py, data/gdelt_news.py)
    e roda a ablacao HAR-RV baseline vs com noticia.
    """
    if not PTAX_PROCESSED_PATH.exists():
        raise FileNotFoundError(f"{PTAX_PROCESSED_PATH} nao encontrado -- rode data.ptax primeiro.")
    if not TONE_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{TONE_PROCESSED_PATH} nao encontrado -- rode data.gdelt_news primeiro."
        )

    ptax_df = pd.read_parquet(PTAX_PROCESSED_PATH)
    prices = ptax_df[ptax_df["tipo"] == tipo].set_index("date")["value"].sort_index()

    tone_df = pd.read_parquet(TONE_PROCESSED_PATH)
    if query is not None:
        tone_df = tone_df[tone_df["query"] == query]
    news = tone_df.set_index("date")["tone"].sort_index()

    return run_ablation(prices, news, horizon=horizon, test_size=test_size)
