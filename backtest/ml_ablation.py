"""Ablacao formal HAR-RV (OLS) vs gradient boosting (XGBoost), no MESMO
walk-forward purgado com embargo usado por todas as demais ablacoes do projeto.

DESENHO (pre-registrado antes de rodar):

- MESMAS features (rv_d, rv_w, rv_m) e MESMO alvo para todos os modelos, nos
  MESMOS folds. A unica coisa que varia e a forma funcional -- ver
  vol/ml_forecast.py para a fundamentacao na literatura.

- MESMA transformacao log do alvo em todos, para o R2 ser comparavel.

- Benchmark de persistencia incluido sempre: e o modelo de ZERO parametros que
  ja derrubou varias camadas neste projeto. Um ML que nao bate a persistencia
  nao merece discussao.

- VARREDURA DE HORIZONTE (1, 3, 5, 10, 15, 21). E a medida central: em h=1 o
  alvo nao se sobrepoe e ha ~2.135 observacoes independentes; em h=21 ha ~101.
  Se a capacidade extra do ML se pagar apenas no horizonte curto, isso
  QUANTIFICA o diagnostico de descasamento de horizonte do projeto usando um
  modelo que encontraria estrutura nao-linear se ela existisse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.metrics import pooled_oos_metrics
from backtest.walk_forward import purged_walk_forward_splits
from vol.forecast import (
    BASELINE_FEATURES,
    fit_har,
    forward_target_from_variance,
    har_features_from_variance,
    predict,
)
from vol.ml_forecast import PARAM_SETS, fit_xgb, predict_xgb
from vol.realized import TRADING_DAYS_PER_YEAR


def build_dataset(daily_variance: pd.Series, horizon: int) -> pd.DataFrame:
    """Dataset supervisionado HAR padrao (features + alvo), identico ao que o
    baseline do projeto usa -- reexportado aqui so por conveniencia."""
    df = har_features_from_variance(daily_variance)
    df["target"] = forward_target_from_variance(daily_variance, horizon)
    return df.dropna()


def run_ml_ablation(
    daily_variance: pd.Series,
    horizon: int = 21,
    n_splits: int = 5,
    embargo_days: int = 5,
    log_target: bool = True,
) -> dict:
    """HAR (OLS) vs XGBoost (arvore e linear) vs persistencia, nos mesmos folds
    purgados. Devolve R2/RMSE/MAE pooled de cada um e o detalhe por fold."""
    dataset = build_dataset(daily_variance, horizon)
    folds = purged_walk_forward_splits(dataset, n_splits, horizon, embargo_days)

    nomes = ["har", *PARAM_SETS, "persistencia"]
    acumulado: dict[str, list[pd.Series]] = {n: [] for n in nomes}
    por_fold: dict[str, list[float]] = {n: [] for n in nomes}
    alvos: list[pd.Series] = []

    for train, test in folds:
        alvos.append(test["target"])

        modelo_har = fit_har(train, BASELINE_FEATURES, log_target=log_target)
        previsoes = {"har": predict(modelo_har, test, BASELINE_FEATURES, log_target=log_target)}

        for nome, params in PARAM_SETS.items():
            modelo = fit_xgb(train, BASELINE_FEATURES, params=params, log_target=log_target)
            previsoes[nome] = predict_xgb(modelo, test, BASELINE_FEATURES, log_target=log_target)

        # persistencia: rv_m anualizada, sem ajustar nada
        previsoes["persistencia"] = np.sqrt(test["rv_m"] * TRADING_DAYS_PER_YEAR) * 100

        for nome, pred in previsoes.items():
            acumulado[nome].append(pred)
            por_fold[nome].append(pooled_oos_metrics(test["target"], pred)["r2_oos"])

    alvo = pd.concat(alvos)
    resultado: dict = {
        "horizon": horizon,
        "n_folds": len(folds),
        "n_obs": int(len(alvo)),
        "por_fold": por_fold,
    }
    for nome in nomes:
        resultado[nome] = pooled_oos_metrics(alvo, pd.concat(acumulado[nome]))
    return resultado


def run_horizon_sweep(
    horizons: tuple[int, ...] = (1, 3, 5, 10, 15, 21),
    n_splits: int = 5,
    embargo_days: int = 5,
    source: str = "b3",
) -> dict[int, dict]:
    """Varredura de horizonte da ablacao de ML -- ver docstring do modulo."""
    from vol.realized import load_prices_and_variance

    _, variance = load_prices_and_variance(source=source)
    return {
        h: run_ml_ablation(variance, horizon=h, n_splits=n_splits, embargo_days=embargo_days)
        for h in horizons
    }
