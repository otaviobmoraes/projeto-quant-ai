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

from backtest.walk_forward import purged_walk_forward_splits
from vol.forecast import (
    BASELINE_FEATURES,
    NEWS_FEATURES,
    build_dataset,
    evaluate,
    fit_har,
    load_prices_and_news,
)


def run_purged_ablation(
    prices: pd.Series,
    news: pd.Series,
    horizon: int = 21,
    n_splits: int = 5,
    embargo_days: int = 5,
    log_target: bool = True,
    news_smooth_window: int | None = 21,
    daily_variance: pd.Series | None = None,
) -> dict:
    """Ajusta o HAR-RV baseline e a versao com noticia em cada fold do
    walk-forward purgado e agrega (media) as metricas fora da amostra.

    `daily_variance`: ver vol.forecast.build_dataset -- permite usar um
    estimador de RV melhor (ex.: Parkinson) em vez do proxy de retorno de
    fechamento.
    """
    dataset = build_dataset(
        prices,
        news=news,
        horizon=horizon,
        news_smooth_window=news_smooth_window,
        daily_variance=daily_variance,
    )
    folds = purged_walk_forward_splits(
        dataset, n_splits=n_splits, horizon=horizon, embargo_days=embargo_days
    )
    if not folds:
        raise ValueError("nenhum fold valido -- dataset pequeno demais para esses parametros")

    per_fold = {"baseline": [], "com_noticia": []}
    for train, test in folds:
        baseline_model = fit_har(train, BASELINE_FEATURES, log_target=log_target)
        news_model = fit_har(train, NEWS_FEATURES, log_target=log_target)
        per_fold["baseline"].append(
            evaluate(baseline_model, test, BASELINE_FEATURES, log_target=log_target)
        )
        per_fold["com_noticia"].append(
            evaluate(news_model, test, NEWS_FEATURES, log_target=log_target)
        )

    def _mean_metrics(results: list[dict]) -> dict:
        return {k: float(np.mean([r[k] for r in results])) for k in results[0]}

    return {
        "baseline": _mean_metrics(per_fold["baseline"]),
        "com_noticia": _mean_metrics(per_fold["com_noticia"]),
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
