"""GARCH(1,1) como estimador/previsor alternativo de RV, pra comparar com o
HAR-RV linear (vol/forecast.py) no mesmo walk-forward purgado -- o CLAUDE.md
da raiz lista HAR-RV/GARCH como os baselines a considerar antes de modelos
mais complexos.

Design pareado com o HAR-RV pra comparacao justa: os PARAMETROS do
GARCH(1,1) sao ajustados so no periodo de treino de cada fold (congelados
dentro do fold, igual aos coeficientes OLS do HAR-RV), mas a previsao
horizon-a-frente pra cada dia do periodo de teste usa os retornos reais ate
aquele dia (dados atualizados, parametros fixos) -- exatamente como o
HAR-RV recalcula rv_d/rv_w/rv_m a cada dia com coeficientes fixos.

Os retornos sao reescalados por 100 antes do fit (pratica padrao do pacote
arch: evita warnings de convergencia com retornos em escala decimal
minuscula) e a variancia prevista e desescalada de volta antes de anualizar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from vol.realized import TRADING_DAYS_PER_YEAR, log_returns

RESCALE = 100.0


def fit_garch(train_returns: pd.Series):
    """Ajusta GARCH(1,1) (Bollerslev, 1986), media zero, erros normais, nos
    retornos (decimal) de treino -- reescala por RESCALE internamente."""
    scaled = train_returns.dropna() * RESCALE
    return arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", dist="normal").fit(disp="off")


def forecast_garch_daily_variances(
    all_returns: pd.Series, fitted_params: pd.Series, start: pd.Timestamp, horizon: int
) -> pd.DataFrame:
    """Previsao horizon-a-frente (colunas h.1..h.horizon) da variancia
    diaria, uma linha por data a partir de `start`, usando os parametros
    FIXOS `fitted_params` (ajustados so no treino) sobre a serie completa de
    retornos -- os parametros nao mudam, so o retorno mais recente conhecido
    em cada linha. Retorna a variancia diaria ja desescalada (nao anualizada).
    """
    scaled = all_returns.dropna() * RESCALE
    model = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
    fc = model.forecast(params=fitted_params, horizon=horizon, start=start, reindex=False)
    return fc.variance / (RESCALE**2)


def garch_forward_target_forecast(
    prices: pd.Series, train_end: pd.Timestamp, test_index: pd.DatetimeIndex, horizon: int
) -> pd.Series:
    """Previsao de RV anualizada (pontos percentuais) pros proximos
    `horizon` dias, uma linha por data em `test_index`: media das variancias
    diarias previstas (h=1..horizon), anualizada -- mesma definicao de
    vol.forecast.forward_target_from_variance, pra comparacao direta.
    """
    returns = log_returns(prices)
    train_returns = returns.loc[returns.index <= train_end]
    fitted = fit_garch(train_returns)

    daily_var_forecasts = forecast_garch_daily_variances(
        returns, fitted.params, start=test_index.min(), horizon=horizon
    )
    avg_daily_var = daily_var_forecasts.reindex(test_index).mean(axis=1)
    return np.sqrt(avg_daily_var * TRADING_DAYS_PER_YEAR) * 100


def evaluate_garch_fold(prices: pd.Series, train: pd.DataFrame, test: pd.DataFrame, horizon: int) -> dict:
    """RMSE/MAE/R2 do GARCH(1,1) num fold no mesmo formato de
    backtest.walk_forward.purged_walk_forward_splits (train/test sao
    DataFrames indexados por data, com coluna "target").
    """
    pred = garch_forward_target_forecast(prices, train.index.max(), test.index, horizon)
    err = test["target"].to_numpy() - pred.to_numpy()
    ss_res = float((err**2).sum())
    ss_tot = float(((test["target"] - test["target"].mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2_oos": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }
