"""Avaliacao FINAL (rigorosa) da ablacao RV baseline vs com noticia: usa o
walk-forward PURGADO com embargo (backtest/walk_forward.py) em vez do CV
expansivo simples de vol/forecast.py (que nao purga rotulos sobrepostos).

Esta e a comparacao que de fato sustenta a decisao de usar (ou nao) a
noticia na estrategia -- vol/forecast.run_ablation_cv fica como a versao
preliminar/mais rapida usada na Fase 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.metrics import pooled_oos_metrics
from backtest.walk_forward import purged_walk_forward_splits
from data.gdelt_news import fiscal_risk_surprise, load_fiscal_risk_series
from data.ptax import PROCESSED_PATH as PTAX_PROCESSED_PATH
from vol.forecast import (
    BASELINE_FEATURES,
    NEWS_FEATURES,
    build_dataset,
    evaluate,
    fit_har,
    forward_target_from_variance,
    har_features_from_variance,
    load_prices_and_news,
    predict,
)
from vol.realized import log_returns

FISCAL_RISK_FEATURES = BASELINE_FEATURES + ["fiscal_surprise", "fiscal_sentiment"]


def run_purged_ablation(
    prices: pd.Series,
    news: pd.Series,
    horizon: int = 21,
    n_splits: int = 5,
    embargo_days: int = 5,
    log_target: bool = True,
    news_smooth_window: int | None = 21,
    daily_variance: pd.Series | None = None,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> dict:
    """Ajusta o HAR-RV baseline e a versao com noticia em cada fold do
    walk-forward purgado e agrega (media) as metricas fora da amostra.

    `daily_variance`: ver vol.forecast.build_dataset -- permite usar um
    estimador de RV melhor (ex.: Parkinson) em vez do proxy de retorno de
    fechamento.

    `folds`: se informado, usa esses folds prontos em vez de gerar via
    `purged_walk_forward_splits(n_splits=...)` -- permite plugar um esquema
    de reestimacao diferente (ex.: `purged_walk_forward_splits_by_step`,
    reestimando todo mes em vez de so 5 vezes) sem duplicar a logica de
    ajuste/avaliacao por fold.

    O resultado traz DUAS versoes de R2/RMSE/MAE: `baseline`/`com_noticia`
    (media simples do metrica por fold -- fica instavel com folds pequenos,
    ver backtest.metrics.pooled_oos_metrics) e `baseline_pooled`/
    `com_noticia_pooled` (concatena as previsoes de todos os folds antes de
    calcular -- mais robusto, e a metrica que deve ser lida como "resultado
    principal").
    """
    dataset = build_dataset(
        prices,
        news=news,
        horizon=horizon,
        news_smooth_window=news_smooth_window,
        daily_variance=daily_variance,
    )
    if folds is None:
        folds = purged_walk_forward_splits(
            dataset, n_splits=n_splits, horizon=horizon, embargo_days=embargo_days
        )
    if not folds:
        raise ValueError("nenhum fold valido -- dataset pequeno demais para esses parametros")

    per_fold = {"baseline": [], "com_noticia": []}
    baseline_preds, news_preds = [], []
    for train, test in folds:
        baseline_model = fit_har(train, BASELINE_FEATURES, log_target=log_target)
        news_model = fit_har(train, NEWS_FEATURES, log_target=log_target)
        per_fold["baseline"].append(
            evaluate(baseline_model, test, BASELINE_FEATURES, log_target=log_target)
        )
        per_fold["com_noticia"].append(
            evaluate(news_model, test, NEWS_FEATURES, log_target=log_target)
        )
        baseline_preds.append(predict(baseline_model, test, BASELINE_FEATURES, log_target=log_target))
        news_preds.append(predict(news_model, test, NEWS_FEATURES, log_target=log_target))

    def _mean_metrics(results: list[dict]) -> dict:
        return {k: float(np.mean([r[k] for r in results])) for k in results[0]}

    baseline_forecast = pd.concat(baseline_preds).sort_index()
    news_forecast = pd.concat(news_preds).sort_index()

    return {
        "baseline": _mean_metrics(per_fold["baseline"]),
        "com_noticia": _mean_metrics(per_fold["com_noticia"]),
        "baseline_pooled": pooled_oos_metrics(dataset["target"], baseline_forecast),
        "com_noticia_pooled": pooled_oos_metrics(dataset["target"], news_forecast),
        "per_fold": per_fold,
        "n_splits": len(folds),
    }


def load_and_run_purged_ablation(
    horizon: int = 21,
    n_splits: int = 5,
    embargo_days: int = 5,
    tipo: str = "venda",
    query: str | None = None,
    log_target: bool = True,
    news_smooth_window: int | None = 21,
    use_parkinson: bool = False,
) -> dict:
    """Le o PTAX e o tom do GDELT ja coletados (data/ptax.py, data/gdelt_news.py)
    e roda a avaliacao final (walk-forward purgado) da ablacao.

    `use_parkinson=True`: usa o estimador de Parkinson via fx_spot (OHLC) em
    vez do proxy de retorno de fechamento do PTAX -- baseline preferencial
    (RMSE menor em todos os folds testados no diagnostico do backtest).
    """
    ptax_prices, news = load_prices_and_news(tipo, query)
    if use_parkinson:
        from vol.realized import load_parkinson_prices_and_variance

        prices, daily_variance = load_parkinson_prices_and_variance()
    else:
        prices, daily_variance = ptax_prices, None

    return run_purged_ablation(
        prices,
        news,
        horizon=horizon,
        n_splits=n_splits,
        embargo_days=embargo_days,
        log_target=log_target,
        news_smooth_window=news_smooth_window,
        daily_variance=daily_variance,
    )


def load_and_run_fiscal_risk_ablation(
    horizon: int = 21,
    n_splits: int = 5,
    embargo_days: int = 5,
    tipo: str = "venda",
    log_target: bool = True,
    news_smooth_window: int | None = 21,
    use_parkinson: bool = False,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    use_surprise: bool = False,
    surprise_window: int = 63,
) -> dict:
    """Mesma avaliacao final (walk-forward purgado) de load_and_run_purged_ablation,
    mas usando ATENCAO da imprensa a risco fiscal (GDELT, share_pct de
    timelinevolraw -- data.gdelt_news.load_fiscal_risk_series) como camada de
    "noticia" em vez do tom (timelinetone). Chave `com_noticia` no resultado
    passa a significar "com risco fiscal" aqui.

    `use_surprise=True`: em vez do NIVEL de atencao (media suavizada em
    `news_smooth_window`), usa a SURPRESA (data.gdelt_news.fiscal_risk_surprise
    -- desvio em unidades de desvio-padrao vs a media movel de `surprise_window`
    dias). Motivacao: nivel medio de cobertura fiscal e persistente e pouco
    informativo; um pico acima do normal recente e o tipo de evento que
    deveria se relacionar a RV futura. Quando True, `news_smooth_window` fica
    forcado a None por padrao aqui pra nao re-suavizar a surpresa e apagar o
    pico -- passe explicitamente se quiser suavizar mesmo assim.

    `folds`: ver run_purged_ablation -- permite reestimar com mais frequencia
    (ex.: purged_walk_forward_splits_by_step) em vez de so n_splits fixos.
    """
    news = load_fiscal_risk_series()
    if use_surprise:
        news = fiscal_risk_surprise(news, window=surprise_window)
        if news_smooth_window == 21:  # default nao alterado explicitamente pelo chamador
            news_smooth_window = None

    if use_parkinson:
        from vol.realized import load_parkinson_prices_and_variance

        prices, daily_variance = load_parkinson_prices_and_variance()
    else:
        if not PTAX_PROCESSED_PATH.exists():
            raise FileNotFoundError(f"{PTAX_PROCESSED_PATH} nao encontrado -- rode data.ptax primeiro.")
        ptax_df = pd.read_parquet(PTAX_PROCESSED_PATH)
        prices = ptax_df[ptax_df["tipo"] == tipo].set_index("date")["value"].sort_index()
        daily_variance = None

    return run_purged_ablation(
        prices,
        news,
        horizon=horizon,
        n_splits=n_splits,
        embargo_days=embargo_days,
        log_target=log_target,
        news_smooth_window=news_smooth_window,
        daily_variance=daily_variance,
        folds=folds,
    )


def _align_daily_series(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Normaliza o indice de `series` para o mesmo tz de `target_index` antes
    do reindex/ffill em build_dataset_with_fiscal_risk. Necessario porque
    sentiment.daily_index.daily_sentiment_index guarda `date` como
    datetime.date PURO (sem tz) -- um reindex direto contra um indice
    tz-aware (America/Sao_Paulo, como o de prices/fiscal_surprise) casaria
    zero labels e viraria tudo NaN silenciosamente.
    """
    idx = pd.to_datetime(series.index)
    if target_index.tz is not None:
        idx = idx.tz_localize(target_index.tz) if idx.tz is None else idx.tz_convert(target_index.tz)
    elif idx.tz is not None:
        idx = idx.tz_localize(None)
    aligned = series.copy()
    aligned.index = idx
    return aligned.sort_index()


def build_dataset_with_fiscal_risk(
    prices: pd.Series,
    fiscal_surprise_series: pd.Series,
    fiscal_sentiment_series: pd.Series,
    horizon: int = 21,
    daily_variance: pd.Series | None = None,
    sentiment_smooth_window: int | None = 5,
) -> pd.DataFrame:
    """Monta o dataset supervisionado com as DUAS camadas refinadas de risco
    fiscal (versao 2, depois do diagnostico de que o nivel bruto de atencao
    nao ajudava -- ver load_and_run_fiscal_risk_ablation):

    - `fiscal_surprise_series`: data.gdelt_news.fiscal_risk_surprise (atencao
      em desvios-padrao vs a media movel recente, ja causal).
    - `fiscal_sentiment_series`: media diaria do FinBERT-PT-BR nas manchetes
      fiscais em PORTUGUES (sentiment.daily_index.load_fiscal_risk_sentiment_index).
      `sentiment_smooth_window`: suaviza com media movel antes do ffill --
      poucas manchetes/dia tornam a media diaria crua ruidosa.

    Ambas alinhadas por ultimo valor conhecido (ffill), sem look-ahead --
    mesmo padrao de vol.forecast.build_dataset / credibility.ablation.
    """
    variance = daily_variance if daily_variance is not None else log_returns(prices) ** 2
    df = har_features_from_variance(variance)
    df["target"] = forward_target_from_variance(variance, horizon)

    surprise = _align_daily_series(fiscal_surprise_series, df.index)
    df["fiscal_surprise"] = surprise.reindex(df.index, method="ffill")

    sentiment = _align_daily_series(fiscal_sentiment_series, df.index)
    if sentiment_smooth_window is not None:
        sentiment = sentiment.rolling(sentiment_smooth_window, min_periods=1).mean()
    df["fiscal_sentiment"] = sentiment.reindex(df.index, method="ffill")

    return df.dropna()


def run_fiscal_risk_ablation_v2(
    prices: pd.Series,
    fiscal_surprise_series: pd.Series,
    fiscal_sentiment_series: pd.Series,
    horizon: int = 21,
    n_splits: int = 5,
    embargo_days: int = 5,
    log_target: bool = True,
    daily_variance: pd.Series | None = None,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    sentiment_smooth_window: int | None = 5,
) -> dict:
    """Ajusta o HAR-RV baseline e a versao refinada de risco fiscal
    (fiscal_surprise + fiscal_sentiment) em cada fold do walk-forward purgado
    -- ablacao formal da versao 2, apos o diagnostico com o nivel bruto
    (load_and_run_fiscal_risk_ablation) nao ter ajudado.
    """
    dataset = build_dataset_with_fiscal_risk(
        prices,
        fiscal_surprise_series,
        fiscal_sentiment_series,
        horizon=horizon,
        daily_variance=daily_variance,
        sentiment_smooth_window=sentiment_smooth_window,
    )
    if folds is None:
        folds = purged_walk_forward_splits(
            dataset, n_splits=n_splits, horizon=horizon, embargo_days=embargo_days
        )
    if not folds:
        raise ValueError("nenhum fold valido -- dataset pequeno demais para esses parametros")

    per_fold = {"baseline": [], "com_risco_fiscal_v2": []}
    baseline_preds, fiscal_preds = [], []
    for train, test in folds:
        baseline_model = fit_har(train, BASELINE_FEATURES, log_target=log_target)
        fiscal_model = fit_har(train, FISCAL_RISK_FEATURES, log_target=log_target)
        per_fold["baseline"].append(
            evaluate(baseline_model, test, BASELINE_FEATURES, log_target=log_target)
        )
        per_fold["com_risco_fiscal_v2"].append(
            evaluate(fiscal_model, test, FISCAL_RISK_FEATURES, log_target=log_target)
        )
        baseline_preds.append(predict(baseline_model, test, BASELINE_FEATURES, log_target=log_target))
        fiscal_preds.append(predict(fiscal_model, test, FISCAL_RISK_FEATURES, log_target=log_target))

    def _mean_metrics(results: list[dict]) -> dict:
        return {k: float(np.mean([r[k] for r in results])) for k in results[0]}

    baseline_forecast = pd.concat(baseline_preds).sort_index()
    fiscal_forecast = pd.concat(fiscal_preds).sort_index()

    return {
        "baseline": _mean_metrics(per_fold["baseline"]),
        "com_risco_fiscal_v2": _mean_metrics(per_fold["com_risco_fiscal_v2"]),
        "baseline_pooled": pooled_oos_metrics(dataset["target"], baseline_forecast),
        "com_risco_fiscal_v2_pooled": pooled_oos_metrics(dataset["target"], fiscal_forecast),
        "per_fold": per_fold,
        "n_splits": len(folds),
    }


def load_and_run_fiscal_risk_ablation_v2(
    horizon: int = 21,
    n_splits: int = 5,
    embargo_days: int = 5,
    tipo: str = "venda",
    log_target: bool = True,
    use_parkinson: bool = False,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    surprise_window: int = 63,
    sentiment_smooth_window: int | None = 5,
) -> dict:
    """Le o risco fiscal ja processado (data.gdelt_news.load_fiscal_risk_series,
    transformado em surpresa) e o sentimento fiscal ja processado
    (sentiment.daily_index.FISCAL_SENTIMENT_PROCESSED_PATH -- rode
    sentiment.daily_index.load_fiscal_risk_sentiment_index() antes, so uma
    vez, pra nao recarregar o FinBERT a cada chamada) e roda a ablacao v2.
    """
    from sentiment.daily_index import FISCAL_SENTIMENT_PROCESSED_PATH

    if not FISCAL_SENTIMENT_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{FISCAL_SENTIMENT_PROCESSED_PATH} nao encontrado -- rode "
            "sentiment.daily_index.load_fiscal_risk_sentiment_index() primeiro."
        )
    sentiment_df = pd.read_parquet(FISCAL_SENTIMENT_PROCESSED_PATH)
    fiscal_sentiment_series = sentiment_df.set_index("date")["sentiment_mean"].sort_index()
    fiscal_surprise_series = fiscal_risk_surprise(load_fiscal_risk_series(), window=surprise_window)

    if use_parkinson:
        from vol.realized import load_parkinson_prices_and_variance

        prices, daily_variance = load_parkinson_prices_and_variance()
    else:
        if not PTAX_PROCESSED_PATH.exists():
            raise FileNotFoundError(f"{PTAX_PROCESSED_PATH} nao encontrado -- rode data.ptax primeiro.")
        ptax_df = pd.read_parquet(PTAX_PROCESSED_PATH)
        prices = ptax_df[ptax_df["tipo"] == tipo].set_index("date")["value"].sort_index()
        daily_variance = None

    return run_fiscal_risk_ablation_v2(
        prices,
        fiscal_surprise_series,
        fiscal_sentiment_series,
        horizon=horizon,
        n_splits=n_splits,
        embargo_days=embargo_days,
        log_target=log_target,
        daily_variance=daily_variance,
        folds=folds,
        sentiment_smooth_window=sentiment_smooth_window,
    )
