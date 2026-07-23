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


def har_features_from_variance(daily_variance: pd.Series) -> pd.DataFrame:
    """Features HAR-RV (diaria, semanal=5d, mensal=22d) a partir de uma serie
    generica de variancia realizada diaria (retorno ao quadrado, Parkinson,
    Garman-Klass, etc.) -- ver `har_features` para o caso especifico de so
    ter preco de fechamento. Cada linha usa somente dados ate aquele dia
    (sem look-ahead).
    """
    df = pd.DataFrame(index=daily_variance.index)
    df["rv_d"] = daily_variance
    df["rv_w"] = daily_variance.rolling(5).mean()
    df["rv_m"] = daily_variance.rolling(22).mean()
    return df


def har_features(prices: pd.Series) -> pd.DataFrame:
    """Features HAR-RV a partir so do preco de fechamento (variancia diaria
    aproximada pelo retorno log ao quadrado -- ver docstring do modulo)."""
    return har_features_from_variance(log_returns(prices) ** 2)


def forward_target_from_variance(daily_variance: pd.Series, horizon: int) -> pd.Series:
    """RV anualizada realizada nos `horizon` dias APOS cada data, a partir de
    uma serie generica de variancia realizada diaria -- ver `forward_target`.
    """
    fwd_var = daily_variance.rolling(horizon).mean().shift(-horizon)
    return np.sqrt(fwd_var * TRADING_DAYS_PER_YEAR) * 100


def forward_target(prices: pd.Series, horizon: int) -> pd.Series:
    """RV anualizada realizada nos `horizon` dias APOS cada data -- o alvo a
    prever. Olha para frente por construcao (shift negativo): usado somente
    para montar o dataset supervisionado, nunca como feature de entrada.
    """
    return forward_target_from_variance(log_returns(prices) ** 2, horizon)


def build_dataset(
    prices: pd.Series,
    news: pd.Series | None = None,
    horizon: int = 21,
    news_smooth_window: int | None = None,
    daily_variance: pd.Series | None = None,
) -> pd.DataFrame:
    """Monta o dataset supervisionado: features HAR (+ noticia, se fornecida)
    e o alvo (RV futura de `horizon` dias). A serie de noticia e alinhada por
    ultimo valor conhecido (ffill) -- ela pode nao ter uma observacao exata em
    todo dia util, mas nunca usa informacao futura.

    `news_smooth_window`: se informado, suaviza a noticia com uma media movel
    dessa janela ANTES do ffill/alinhamento (ex.: 21, para casar a cadencia
    com o componente mensal do HAR). O nivel bruto e ruidoso demais para
    prever RV com 21 dias de horizonte; a media suavizada e mais estavel.

    `daily_variance`: se informado, usa essa serie como a variancia diaria
    (em vez de recalcular do retorno de `prices`) -- permite plugar um
    estimador de RV melhor (ex.: Parkinson via High/Low, ver vol/realized.py)
    no mesmo pipeline HAR sem duplicar a logica de features/target/split.
    """
    variance = daily_variance if daily_variance is not None else log_returns(prices) ** 2
    df = har_features_from_variance(variance)
    df["target"] = forward_target_from_variance(variance, horizon)
    if news is not None:
        if news_smooth_window is not None:
            news = news.rolling(news_smooth_window, min_periods=1).mean()
        df["news"] = news.reindex(df.index, method="ffill")
    return df.dropna()


def chronological_split(dataset: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cronologico simples (sem embaralhar, treino sempre antes do teste
    no tempo). Avaliacao preliminar -- o walk-forward purgado fica em backtest/.
    """
    n_test = max(1, int(len(dataset) * test_size))
    return dataset.iloc[:-n_test], dataset.iloc[-n_test:]


def fit_har(train: pd.DataFrame, feature_cols: list[str], log_target: bool = False):
    """Ajusta HAR-RV por OLS. `log_target=True` ajusta em log(RV) -- mais
    proximo da pratica padrao (RV e aproximadamente log-normal, evita previsao
    negativa e estabiliza a variancia dos residuos) e costuma generalizar
    melhor fora da amostra do que RV em nivel.
    """
    # has_constant="add" e obrigatorio aqui: o default ("skip") detecta
    # qualquer feature constante DENTRO do fold de treino (pode acontecer
    # com series de baixa variancia, ex.: dispersion/theta_baseline em
    # janelas curtas) e pula a adicao do intercepto -- af entao o numero de
    # colunas do treino fica menor que o do teste (que sempre forca "add" em
    # evaluate()), quebrando o predict() por incompatibilidade de shape.
    X = sm.add_constant(train[feature_cols], has_constant="add")
    y = np.log(train["target"]) if log_target else train["target"]
    return sm.OLS(y, X).fit()


def evaluate(model, test: pd.DataFrame, feature_cols: list[str], log_target: bool = False) -> dict:
    """Avalia RMSE/MAE/R2 fora da amostra, sempre na escala original de RV
    (se `log_target=True`, desfaz o log da previsao com exp antes de comparar)
    -- assim os resultados sao comparaveis entre as duas parametrizacoes.
    """
    X_test = sm.add_constant(test[feature_cols], has_constant="add")
    pred = model.predict(X_test)
    if log_target:
        pred = np.exp(pred)
    err = test["target"].to_numpy() - pred.to_numpy()
    ss_res = float((err**2).sum())
    ss_tot = float(((test["target"] - test["target"].mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2_oos": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def persistence_forecast(dataset: pd.DataFrame) -> pd.Series:
    """Previsao INGENUA de RV futura: usa rv_m (variancia media dos ultimos
    22 dias, ja em `dataset`) anualizada e convertida pra vol em pontos
    percentuais -- equivale a dizer "a RV daqui a `horizon` dias vai ser
    igual a de agora". Nao ajusta nenhum modelo.

    Baseline minimo que o HAR-RV ajustado por OLS precisa bater pra
    justificar usar regressao em vez de so projetar a RV recente pra
    frente. Se HAR-RV perder pra isso, o problema esta na estimacao/
    especificacao do modelo, nao na falta de sinal na feature.
    """
    return np.sqrt(dataset["rv_m"] * TRADING_DAYS_PER_YEAR) * 100


def evaluate_persistence(dataset: pd.DataFrame) -> dict:
    """RMSE/MAE/R2 do baseline de persistencia (sem ajuste de modelo) no
    mesmo `dataset` (tipicamente um fold de teste)."""
    pred = persistence_forecast(dataset)
    err = dataset["target"].to_numpy() - pred.to_numpy()
    ss_res = float((err**2).sum())
    ss_tot = float(((dataset["target"] - dataset["target"].mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2_oos": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def run_ablation(
    prices: pd.Series,
    news: pd.Series,
    horizon: int = 21,
    test_size: float = 0.2,
    log_target: bool = False,
    news_smooth_window: int | None = None,
) -> dict:
    """Ajusta o HAR-RV baseline e a versao com noticia (mesmo split de
    treino/teste para os dois) e devolve as metricas fora da amostra lado a
    lado -- a comparacao com noticia vs sem noticia exigida pelo CLAUDE.md.
    """
    dataset = build_dataset(prices, news=news, horizon=horizon, news_smooth_window=news_smooth_window)
    train, test = chronological_split(dataset, test_size)

    baseline_model = fit_har(train, BASELINE_FEATURES, log_target=log_target)
    news_model = fit_har(train, NEWS_FEATURES, log_target=log_target)

    return {
        "baseline": evaluate(baseline_model, test, BASELINE_FEATURES, log_target=log_target),
        "com_noticia": evaluate(news_model, test, NEWS_FEATURES, log_target=log_target),
        "n_train": len(train),
        "n_test": len(test),
    }


def expanding_window_splits(dataset: pd.DataFrame, n_splits: int = 5) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Divide o dataset em `n_splits` folds cronologicos de janela expansiva:
    o fold i treina com tudo antes do bloco de teste i e testa num bloco
    contiguo subsequente (sem embaralhar, sem sobreposicao entre os blocos de
    teste). Reduz a chance de o resultado da ablacao depender de um unico
    corte "sortudo"/"azarado" -- o walk-forward purgado com embargo completo
    ainda fica para backtest/.
    """
    n = len(dataset)
    fold_size = n // (n_splits + 1)
    if fold_size < 1:
        raise ValueError(f"dataset com {n} linhas e pequeno demais para {n_splits} folds")

    splits = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = min(n, fold_size * (i + 1))
        splits.append((dataset.iloc[:train_end], dataset.iloc[train_end:test_end]))
    return splits


def run_ablation_cv(
    prices: pd.Series,
    news: pd.Series,
    horizon: int = 21,
    n_splits: int = 5,
    log_target: bool = True,
    news_smooth_window: int | None = 21,
) -> dict:
    """Versao com validacao cruzada expansiva do run_ablation: ajusta baseline
    e com-noticia em cada fold e agrega (media) as metricas fora da amostra,
    alem de devolver o detalhe por fold para inspecao.
    """
    dataset = build_dataset(prices, news=news, horizon=horizon, news_smooth_window=news_smooth_window)
    folds = expanding_window_splits(dataset, n_splits=n_splits)

    per_fold = {"baseline": [], "com_noticia": []}
    for train, test in folds:
        baseline_model = fit_har(train, BASELINE_FEATURES, log_target=log_target)
        news_model = fit_har(train, NEWS_FEATURES, log_target=log_target)
        per_fold["baseline"].append(evaluate(baseline_model, test, BASELINE_FEATURES, log_target=log_target))
        per_fold["com_noticia"].append(evaluate(news_model, test, NEWS_FEATURES, log_target=log_target))

    def _mean_metrics(results: list[dict]) -> dict:
        return {k: float(np.mean([r[k] for r in results])) for k in results[0]}

    return {
        "baseline": _mean_metrics(per_fold["baseline"]),
        "com_noticia": _mean_metrics(per_fold["com_noticia"]),
        "per_fold": per_fold,
        "n_splits": n_splits,
    }


def load_prices_and_news(tipo: str, query: str | None) -> tuple[pd.Series, pd.Series]:
    """I/O comum aos wrappers load_and_run_ablation*: le o PTAX e o tom do
    GDELT ja coletados (data/ptax.py, data/gdelt_news.py)."""
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

    return prices, news


def load_and_run_ablation(
    horizon: int = 21, test_size: float = 0.2, tipo: str = "venda", query: str | None = None
) -> dict:
    """Le o PTAX e o tom do GDELT ja coletados e roda a ablacao HAR-RV
    baseline vs com noticia (split unico treino/teste)."""
    prices, news = load_prices_and_news(tipo, query)
    return run_ablation(prices, news, horizon=horizon, test_size=test_size)


def load_and_run_ablation_cv(
    horizon: int = 21,
    n_splits: int = 5,
    tipo: str = "venda",
    query: str | None = None,
    log_target: bool = True,
    news_smooth_window: int | None = 21,
) -> dict:
    """Equivalente a load_and_run_ablation, mas usando run_ablation_cv
    (validacao cruzada expansiva) em vez de um unico split treino/teste.
    """
    prices, news = load_prices_and_news(tipo, query)
    return run_ablation_cv(
        prices,
        news,
        horizon=horizon,
        n_splits=n_splits,
        log_target=log_target,
        news_smooth_window=news_smooth_window,
    )
