"""Metricas robustas de avaliacao: Sharpe, Probabilistic Sharpe Ratio (PSR) e
Deflated Sharpe Ratio (DSR).

Bailey, D. H. & Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality".

Guardrail do CLAUDE.md: "Registrar o numero de configuracoes testadas e usar
Deflated Sharpe Ratio. Nao repetir o grid search cego da referencia de
2023." O DSR ajusta o Sharpe observado pelo numero de configuracoes
testadas (`n_trials`) -- sem isso, escolher a melhor entre N variantes e so
reportar essa superestima sistematicamente a performance esperada fora da
amostra (overfitting de selecao).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns, annualization_factor: int = 252) -> float:
    """Sharpe anualizado (assume taxa livre de risco ja descontada dos
    retornos, i.e. `returns` sao retornos em excesso)."""
    returns = np.asarray(returns, dtype=float)
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float(returns.mean() / std * math.sqrt(annualization_factor))


def probabilistic_sharpe_ratio(
    sr_hat: float, benchmark_sr: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """P(SR verdadeiro > benchmark_sr | Sharpe observado = sr_hat), ajustando
    pela nao-normalidade dos retornos. `kurtosis` na convencao normal=3 (nao
    curtose em excesso).
    """
    denom = math.sqrt(1 - skew * sr_hat + ((kurtosis - 1) / 4) * sr_hat**2)
    z = (sr_hat - benchmark_sr) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe_ratio(sr_trials_std: float, n_trials: int) -> float:
    """Sharpe maximo esperado por puro acaso ao testar `n_trials`
    configuracoes independentes cujos Sharpes tem dispersao `sr_trials_std`
    (aproximacao de teoria de valores extremos; requer n_trials >= 2).
    """
    if n_trials < 2:
        raise ValueError("n_trials precisa ser >= 2 para a aproximacao de valor extremo")
    return float(
        sr_trials_std
        * (
            (1 - EULER_MASCHERONI) * norm.ppf(1 - 1 / n_trials)
            + EULER_MASCHERONI * norm.ppf(1 - 1 / (n_trials * math.e))
        )
    )


def deflated_sharpe_ratio(
    sr_hat: float,
    sr_trials_std: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR: PSR avaliado contra o Sharpe maximo esperado por acaso dado
    `n_trials` configuracoes testadas -- quanto mais configuracoes testadas
    (maior `n_trials`) ou mais dispersos os resultados entre elas (maior
    `sr_trials_std`), maior a barra que o Sharpe observado precisa vencer
    para nao ser so sorte de selecao.
    """
    benchmark = expected_max_sharpe_ratio(sr_trials_std, n_trials)
    return probabilistic_sharpe_ratio(sr_hat, benchmark, n_obs, skew, kurtosis)
