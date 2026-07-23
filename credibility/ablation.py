"""Ablacao Tier 1 (credibility/CLAUDE.md): previsao de RV com vs. sem
credibilidade (theta_baseline + dispersao), usando o mesmo walk-forward
PURGADO de backtest/walk_forward.py -- guardrail da raiz: toda avaliacao de
RV compara com vs. sem a camada nova, walk-forward purgado, nunca split
aleatorio.

Reaproveita as pecas puras de vol/forecast.py (har_features, forward_target,
fit_har, evaluate). Monta seu proprio dataset em vez de reusar
vol.forecast.build_dataset porque aqui entram DUAS features novas
(theta_baseline, dispersion), nao uma so ("news").
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from backtest.walk_forward import purged_walk_forward_splits
from credibility.credibility import PROCESSED_PATH as CREDIBILITY_PROCESSED_PATH
from data.ptax import PROCESSED_PATH as PTAX_PROCESSED_PATH
from vol.forecast import evaluate, fit_har, forward_target, har_features

BASELINE_FEATURES = ["rv_d", "rv_w", "rv_m"]
CREDIBILITY_FEATURES = BASELINE_FEATURES + ["theta_baseline", "dispersion"]


def build_dataset_with_credibility(
    prices: pd.Series, credibility_df: pd.DataFrame, horizon: int = 21
) -> pd.DataFrame:
    """Monta o dataset supervisionado: features HAR + theta_baseline +
    dispersion (alinhadas por ultimo valor conhecido -- ffill, sem
    look-ahead) + alvo (RV futura de `horizon` dias).

    `credibility_df`: saida de credibility.credibility.load_credibility_processed
    (colunas date, gap, dispersion, theta_baseline).
    """
    df = har_features(prices)
    df["target"] = forward_target(prices, horizon)

    cred_indexed = credibility_df.set_index("date")[["theta_baseline", "dispersion"]]
    aligned = cred_indexed.reindex(df.index, method="ffill")
    df["theta_baseline"] = aligned["theta_baseline"]
    df["dispersion"] = aligned["dispersion"]

    return df.dropna()


def run_credibility_ablation(
    prices: pd.Series,
    credibility_df: pd.DataFrame,
    horizon: int = 21,
    n_splits: int = 5,
    embargo_days: int = 5,
    log_target: bool = True,
) -> dict:
    """Ajusta o HAR-RV baseline e a versao com credibilidade em cada fold do
    walk-forward purgado e agrega (media) as metricas fora da amostra.
    """
    dataset = build_dataset_with_credibility(prices, credibility_df, horizon=horizon)
    folds = purged_walk_forward_splits(
        dataset, n_splits=n_splits, horizon=horizon, embargo_days=embargo_days
    )
    if not folds:
        raise ValueError("nenhum fold valido -- dataset pequeno demais para esses parametros")

    per_fold = {"baseline": [], "com_credibilidade": []}
    for train, test in folds:
        baseline_model = fit_har(train, BASELINE_FEATURES, log_target=log_target)
        cred_model = fit_har(train, CREDIBILITY_FEATURES, log_target=log_target)
        per_fold["baseline"].append(
            evaluate(baseline_model, test, BASELINE_FEATURES, log_target=log_target)
        )
        per_fold["com_credibilidade"].append(
            evaluate(cred_model, test, CREDIBILITY_FEATURES, log_target=log_target)
        )

    def _mean_metrics(results: list[dict]) -> dict:
        return {k: float(np.mean([r[k] for r in results])) for k in results[0]}

    return {
        "baseline": _mean_metrics(per_fold["baseline"]),
        "com_credibilidade": _mean_metrics(per_fold["com_credibilidade"]),
        "per_fold": per_fold,
        "n_splits": len(folds),
    }


def load_and_run_credibility_ablation(
    horizon: int = 21,
    n_splits: int = 5,
    embargo_days: int = 5,
    tipo: str = "venda",
    log_target: bool = True,
) -> dict:
    """Le o PTAX e a credibilidade ja processados (data.ptax,
    credibility.credibility) e roda a ablacao."""
    if not PTAX_PROCESSED_PATH.exists():
        raise FileNotFoundError(f"{PTAX_PROCESSED_PATH} nao encontrado -- rode data.ptax primeiro.")
    if not CREDIBILITY_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{CREDIBILITY_PROCESSED_PATH} nao encontrado -- rode "
            "credibility.credibility.load_credibility_processed primeiro."
        )

    ptax_df = pd.read_parquet(PTAX_PROCESSED_PATH)
    prices = ptax_df[ptax_df["tipo"] == tipo].set_index("date")["value"].sort_index()
    credibility_df = pd.read_parquet(CREDIBILITY_PROCESSED_PATH)

    return run_credibility_ablation(
        prices,
        credibility_df,
        horizon=horizon,
        n_splits=n_splits,
        embargo_days=embargo_days,
        log_target=log_target,
    )
