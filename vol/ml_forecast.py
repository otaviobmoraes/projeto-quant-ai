"""Previsao de RV com gradient boosting (XGBoost), como alternativa ao HAR-RV
linear -- avaliada no MESMO walk-forward purgado das demais ablacoes.

FUNDAMENTACAO NA LITERATURA (as tres decisoes de desenho abaixo nao sao
arbitrarias):

1. MESMAS FEATURES DO HAR (rv_d, rv_w, rv_m), sem features novas.
   Christensen, Siggaard & Veliyev (2023), "A Machine Learning Approach to
   Volatility Forecasting", Journal of Financial Econometrics 21(5):1680-1727,
   mostram que o ML supera a linhagem HAR *mesmo quando os unicos preditores
   sao as defasagens diaria, semanal e mensal da variancia realizada*. Manter
   as features identicas isola a FORMA FUNCIONAL como unica variavel -- se o
   XGBoost ganhar, o ganho e da nao-linearidade, nao de informacao extra.

2. AJUSTE MINIMO DE HIPERPARAMETROS, pre-registrados e fixos.
   Christensen et al. obtem seus resultados "in spite of minimal hyperparameter
   tuning". Aqui isso e ainda mais necessario: o CLAUDE.md exige registrar o
   numero de configuracoes testadas e aplicar Deflated Sharpe Ratio, e uma
   busca em grade inflaria esse contador ate destruir a significancia. Os
   valores abaixo foram fixados ANTES de rodar qualquer avaliacao.

3. DOIS BASE LEARNERS (arvore e linear), comparados por horizonte.
   Teller, Pigorsch & Pigorsch, "Short- to Long-Term Realized Volatility
   Forecasting using Extreme Gradient Boosting" (SSRN 4267541): XGBoost supera
   HAR e LSTM em um passo a frente, mas em horizontes LONGOS os base learners
   LINEARES superam as especificacoes nao-lineares. Testar so arvore
   responderia metade da pergunta.

DIFERENCA DE CONTEXTO, declarada: os dois trabalhos acima usam variancia
realizada INTRADIARIA e, no caso de Christensen et al., um PAINEL de dezenas
de ativos -- efetivamente dezenas de milhares de observacoes. Aqui ha um unico
ativo, variancia de Parkinson sobre OHLC diario, 2.135 pregoes. Em h=1 o alvo
nao se sobrepoe e ha ~2.135 observacoes independentes (regime confortavel);
em h=21 ha ~101 janelas independentes (regime apertado para modelo de alta
capacidade). Essa assimetria e justamente o que a varredura de horizonte mede.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Hiperparametros PRE-REGISTRADOS. Nao ajustar sem registrar a mudanca como
# configuracao nova em report/run_report.py:CONFIGS_TESTED.
TREE_PARAMS: dict = {
    "booster": "gbtree",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,          # raso: a amostra e pequena para o padrao de ML
    "min_child_weight": 10,  # alto: exige suporte real antes de abrir um no
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": 0,
}

LINEAR_PARAMS: dict = {
    "booster": "gblinear",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": 0,
}

PARAM_SETS: dict[str, dict] = {"xgb_arvore": TREE_PARAMS, "xgb_linear": LINEAR_PARAMS}


def fit_xgb(
    train: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
    log_target: bool = True,
):
    """Ajusta um XGBoost no fold de treino.

    `log_target=True` por padrao, igual ao `fit_har` do projeto: a RV e
    aproximadamente log-normal, e prever em log evita previsao negativa e
    estabiliza a variancia dos residuos. Manter a MESMA transformacao do
    baseline e o que torna a comparacao justa.

    PADRONIZACAO CONDICIONAL, e o motivo importa: o base learner `gblinear`
    ajusta por descida de gradiente, que e SENSIVEL A ESCALA. As features aqui
    sao variancias diarias na ordem de 1e-5 a 1e-3, enquanto o alvo em log fica
    perto de 2,6 -- sem padronizar, os coeficientes nao conseguem crescer o
    bastante em 200 rodadas e o modelo sai gravemente subajustado (medido: R2
    negativo em TODOS os horizontes, artefato puro de escala). O OLS do
    baseline nao sofre disso porque resolve as equacoes normais exatamente.
    Arvores sao invariantes a escala monotona, entao NAO sao padronizadas --
    escalar ali nao mudaria nada e so obscureceria a comparacao.

    O scaler e ajustado SO no treino e aplicado ao teste (dentro do Pipeline),
    sem vazamento entre folds.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    params = params or TREE_PARAMS
    regressor = XGBRegressor(**params)
    model = (
        make_pipeline(StandardScaler(), regressor)
        if params.get("booster") == "gblinear"
        else regressor
    )
    y = np.log(train["target"]) if log_target else train["target"]
    model.fit(train[feature_cols], y)
    return model


def predict_xgb(
    model, data: pd.DataFrame, feature_cols: list[str], log_target: bool = True
) -> pd.Series:
    """Previsao na escala ORIGINAL de RV (desfaz o log), para ser comparavel
    ponto a ponto com `vol.forecast.predict`."""
    pred = np.asarray(model.predict(data[feature_cols]), dtype=float)
    if log_target:
        pred = np.exp(pred)
    return pd.Series(pred, index=data.index, name="rv_forecast")
