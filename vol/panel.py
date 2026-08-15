"""Modelo GLOBAL de volatilidade estimado em PAINEL de moedas emergentes.

A IDEIA. Um modelo "local" ajusta coeficientes usando so a historia do
USD/BRL; um modelo "global" ajusta UM conjunto de coeficientes sobre a
historia EMPILHADA de varias moedas e o aplica ao USD/BRL. A aposta e de
vies contra variancia: o global e mais enviesado (as moedas nao sao
identicas) mas estima seus parametros com ~8x mais dados, entao erra menos
por ruido de estimacao. Bollerslev, Hood, Huss & Pedersen (2018, RFS 31(7))
mostram que essa troca compensa para volatilidade; Montero-Manso & Hyndman
(2021, IJF 37(4):1632-1653) mostram que modelos globais batem locais mesmo
quando as series NAO sao relacionadas, justamente pelo ganho de estimacao.

O QUE TORNA O EMPILHAMENTO POSSIVEL. Nao da para empilhar variancia em nivel:
a vol de Parkinson media da amostra vai de 7,98% (INR) a 16,18% (TRY), e uma
regressao em nivel aprenderia sobretudo a diferenca de escala entre moedas. As
features LIVRES DE ESCALA (rv_d/rv_m, rv_w/rv_m) e o alvo em razao
(RV_futura/RV_corrente) sao adimensionais e diretamente comparaveis entre
moedas -- e por isso que o painel so ficou viavel depois daquele trabalho.
Nota: o alvo em razao REPROVOU como melhora de acuracia no modelo local (ver
CONFIGS_TESTED); aqui ele nao entra como melhoria, e como pre-requisito
tecnico do empilhamento.

PURGA NO PAINEL -- o ponto de vazamento que este desenho poderia ter. Nao
basta purgar a historia do BRL: uma observacao de MXN em t cujo alvo cobre
[t, t+21] carrega informacao sobre o MESMO regime global de vol que o bloco de
teste do BRL. Por isso `purge_panel_by_date` corta TODAS as moedas na mesma
data de corte, e nao so a moeda alvo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from vol.forecast import (
    SCALE_FREE_FEATURES,
    forward_target_from_variance,
    har_features_from_variance,
    persistence_forecast,
    scale_free_features,
)
from vol.realized import parkinson_daily_variance

PANEL_COLUMNS = [
    "date", "currency", "ratio_d", "ratio_w", "rv_trailing", "target_ratio", "target",
]


def build_currency_frame(
    high: pd.Series, low: pd.Series, horizon: int, currency: str
) -> pd.DataFrame:
    """Monta o bloco de painel de UMA moeda a partir de high/low.

    Usa PARKINSON de proposito: depende so de high/low, que sao os campos do
    yfinance validados como corretamente datados (ver data/em_fx.py). Nao usar
    `open`/`close` do yfinance para nada aqui.
    """
    var = parkinson_daily_variance(high, low)
    df = har_features_from_variance(var)
    df["target"] = forward_target_from_variance(var, horizon)
    df = df.dropna()

    out = pd.DataFrame(index=df.index)
    out[SCALE_FREE_FEATURES] = scale_free_features(df)
    out["rv_trailing"] = persistence_forecast(df)
    out["target_ratio"] = df["target"] / out["rv_trailing"]
    out["target"] = df["target"]
    out["currency"] = currency
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    # Ver vol.forecast.build_scale_free_dataset: target_ratio == 0 e finito mas
    # quebra o log. No painel isso NAO e hipotetico -- CLP tem 0,40% dos
    # pregoes com high == low (e COP 0,13%, TRY 0,04%), o que zera a variancia
    # de Parkinson do dia. Em h=21 a media de 21 dias esconde; em h=1 o alvo e
    # o proprio dia e vira log(0) = -inf, contaminando TODO o modelo global.
    return out[out["target_ratio"] > 0].reset_index(names="date")


def build_panel(em_fx: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Empilha todas as moedas de `em_fx` (formato longo de data/em_fx.py) num
    unico painel adimensional pronto para regressao."""
    blocos = []
    for currency, g in em_fx.groupby("currency"):
        g = g.sort_values("date").set_index("date")
        blocos.append(build_currency_frame(g["high"], g["low"], horizon, currency))
    if not blocos:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    return pd.concat(blocos, ignore_index=True).sort_values(["date", "currency"])


def purge_panel_by_date(
    panel: pd.DataFrame, train_end: pd.Timestamp, horizon: int, embargo_days: int = 0
) -> pd.DataFrame:
    """Corta o painel em `train_end`, removendo tambem as ultimas
    `horizon + embargo_days` observacoes de TODAS as moedas.

    A purga por DATA (e nao por moeda) e o que impede o vazamento descrito na
    docstring do modulo: sem ela, o alvo de uma moeda vizinha se sobreporia ao
    bloco de teste do BRL e o modelo global "veria" o regime de vol do periodo
    que deveria prever.
    """
    datas = pd.DatetimeIndex(sorted(panel["date"].unique()))
    corte_pos = datas.searchsorted(train_end, side="right") - (horizon + embargo_days)
    if corte_pos <= 0:
        return panel.iloc[0:0]
    corte = datas[corte_pos - 1]
    return panel[panel["date"] <= corte]


def fit_global_model(panel_train: pd.DataFrame, feature_cols: list[str] | None = None):
    """Ajusta UM conjunto de coeficientes sobre o painel empilhado, em
    log(RV_futura / RV_corrente).

    Sem efeito fixo por moeda de proposito: um intercepto por moeda seria um
    parametro local de volta, e a razao de ser do modelo global e ter os MESMOS
    parametros para todos (Montero-Manso & Hyndman: series tratadas como
    permutaveis). Como as features e o alvo ja sao adimensionais, o intercepto
    comum tem interpretacao economica -- o quanto a vol tipicamente reverte no
    horizonte, em qualquer moeda emergente.
    """
    feature_cols = feature_cols or SCALE_FREE_FEATURES
    X = sm.add_constant(panel_train[feature_cols], has_constant="add")
    return sm.OLS(np.log(panel_train["target_ratio"]), X).fit()


def predict_with_global_model(
    model,
    local_data: pd.DataFrame,
    feature_cols: list[str] | None = None,
    jensen_correction: bool = True,
) -> pd.Series:
    """Aplica os coeficientes globais as features LOCAIS (as do futuro da B3) e
    devolve a previsao de RV em NIVEL, comparavel ponto a ponto com
    `vol.forecast.predict`.

    `local_data` precisa ter as colunas de `feature_cols` e `rv_trailing` --
    isto e, o formato de `vol.forecast.build_scale_free_dataset`. O nivel vem
    SEMPRE da RV corrente do proprio BRL medida no futuro da B3; do painel vem
    apenas a FORMA da reversao.
    """
    feature_cols = feature_cols or SCALE_FREE_FEATURES
    X = sm.add_constant(local_data[feature_cols], has_constant="add")
    log_ratio = model.predict(X)
    fator = np.exp(model.mse_resid / 2.0) if jensen_correction else 1.0
    return (np.exp(log_ratio) * fator * local_data["rv_trailing"]).rename("rv_forecast")