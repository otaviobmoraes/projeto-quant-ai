"""Ablacao formal da correcao de ciclo de rolagem (vol/roll.py) no walk-forward
PURGADO com embargo -- mesmo protocolo das demais ablacoes do projeto
(backtest/ablation.py, credibility/ablation.py).

DESENHO (pre-registrado antes de rodar):

- O ALVO e sempre a RV futura construida da variancia ORIGINAL, nunca da
  dessazonalizada. Sem isso o R2 nao seria comparavel ao baseline de -0,255
  ja publicado no relatorio -- estariamos medindo acerto contra um alvo mais
  facil, que e uma das formas classicas de inflar R2 sem ganho real.

- O fator sazonal e estimado SOMENTE no periodo de treino de cada fold e
  aplicado ao teste. Estimar na amostra completa vazaria o nivel de vol do
  periodo de teste para dentro da feature.

- Duas variantes testadas contra o mesmo baseline, nos mesmos folds:
  (a) "dessazonalizada": features HAR construidas sobre a variancia corrigida.
  (b) "com_dte": features HAR originais + a posicao no ciclo do contrato como
      feature explicita (fator sazonal do dia, estimado no treino), deixando o
      OLS decidir o peso em vez de impor a correcao.

- Varredura de HORIZONTE (1, 5, 10, 21): a hipotese mecanica e que a correcao
  so pode ajudar onde o alvo NAO cobre um ciclo inteiro de contrato (~20,2
  pregoes). Em h=21 o dente de serra se cancela dentro do proprio alvo.
"""

from __future__ import annotations

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
from vol.roll import (
    DEFAULT_BUCKET_EDGES,
    bucket_days_to_expiry,
    deseasonalize,
    load_b3_futures_with_dte,
    seasonal_factor,
)

DTE_FEATURES = BASELINE_FEATURES + ["roll_factor"]


def _assemble(
    variance_for_features: pd.Series,
    variance_for_target: pd.Series,
    horizon: int,
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Monta o dataset supervisionado com features de uma serie de variancia e
    ALVO de outra -- o que permite corrigir as features sem tocar no alvo."""
    df = har_features_from_variance(variance_for_features)
    df["target"] = forward_target_from_variance(variance_for_target, horizon)
    if extra is not None:
        for col in extra.columns:
            df[col] = extra[col]
    return df.dropna()


def run_roll_ablation(
    variance: pd.Series,
    dte: pd.Series,
    horizon: int = 21,
    n_splits: int = 5,
    embargo_days: int = 5,
    log_target: bool = True,
    edges: tuple[int, ...] = DEFAULT_BUCKET_EDGES,
) -> dict:
    """Roda baseline vs as duas variantes de correcao de rolagem nos MESMOS
    folds purgados, e devolve o R2 pooled de cada uma.

    O fator sazonal e reestimado dentro de cada fold, so com dados de treino.
    """
    base = _assemble(variance, variance, horizon)
    folds = purged_walk_forward_splits(base, n_splits, horizon, embargo_days)

    collected: dict[str, dict[str, list]] = {
        name: {"actual": [], "pred": []} for name in ("baseline", "dessazonalizada", "com_dte")
    }
    per_fold: dict[str, list[float]] = {name: [] for name in collected}

    for train, test in folds:
        train_idx, test_idx = train.index, test.index

        # --- fator sazonal estimado SO no treino -------------------------
        factor = seasonal_factor(variance.loc[train_idx], dte.loc[train_idx], edges)
        var_adj = deseasonalize(variance, dte, factor, edges)
        roll_factor = bucket_days_to_expiry(dte, edges).map(factor).fillna(1.0).astype(float)

        datasets = {
            "baseline": base,
            "dessazonalizada": _assemble(var_adj, variance, horizon),
            "com_dte": _assemble(
                variance, variance, horizon, extra=pd.DataFrame({"roll_factor": roll_factor})
            ),
        }
        features = {
            "baseline": BASELINE_FEATURES,
            "dessazonalizada": BASELINE_FEATURES,
            "com_dte": DTE_FEATURES,
        }

        for name, dataset in datasets.items():
            tr = dataset.loc[dataset.index.intersection(train_idx)]
            te = dataset.loc[dataset.index.intersection(test_idx)]
            if tr.empty or te.empty:
                continue
            model = fit_har(tr, features[name], log_target=log_target)
            pred = predict(model, te, features[name], log_target=log_target)
            collected[name]["actual"].append(te["target"])
            collected[name]["pred"].append(pred)
            per_fold[name].append(pooled_oos_metrics(te["target"], pred)["r2_oos"])

    result: dict = {"horizon": horizon, "n_folds": len(folds), "per_fold": per_fold}
    for name, acc in collected.items():
        actual = pd.concat(acc["actual"])
        pred = pd.concat(acc["pred"])
        result[f"{name}_pooled"] = pooled_oos_metrics(actual, pred)
    return result


def run_horizon_sweep(
    horizons: tuple[int, ...] = (1, 5, 10, 21),
    n_splits: int = 5,
    embargo_days: int = 5,
) -> dict[int, dict]:
    """Varredura de horizonte da ablacao de rolagem -- ver docstring do modulo
    para a hipotese que ela testa."""
    _, variance, dte = load_b3_futures_with_dte()
    return {h: run_roll_ablation(variance, dte, horizon=h, n_splits=n_splits,
                                 embargo_days=embargo_days) for h in horizons}
