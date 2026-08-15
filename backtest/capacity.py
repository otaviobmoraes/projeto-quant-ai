"""Diagnostico do LIMITE: e o modelo que e fraco, ou nao ha sinal a extrair?

Este modulo existe para responder com MEDICAO, e nao com afirmacao, a
pergunta "nao da para treinar mais o modelo para melhorar?". Sao tres
diagnosticos, cada um fechando uma porta diferente:

1. CURVA DE APRENDIZADO (`learning_curve`) -- desempenho fora da amostra em
   funcao do TAMANHO DO TREINO, com o periodo de TESTE FIXO. E o instrumento
   padrao para decidir se mais dados ajudariam: curva ainda subindo => faltam
   dados; curva plana => o limite nao e amostra.

2. TETO EM AMOSTRA (`in_sample_ceiling`) -- R2 do modelo ajustado E avaliado
   nos MESMOS dados, isto e, com permissao para colar. E um limite SUPERIOR do
   que aquela classe de modelo consegue extrair daquelas features. Se nem
   colando o modelo explica o alvo, nenhuma variante honesta vai.

3. VARREDURA DE CAPACIDADE (`capacity_sweep`) -- R2 fora da amostra em funcao
   da capacidade do modelo. Se aumentar capacidade nao melhora (ou piora), o
   gargalo nao e poder de representacao.

O CONTRASTE QUE DECIDE e entre (2) e (3): um modelo de alta capacidade que
atinge R2 em amostra perto de 1,0 e R2 fora da amostra perto de 0 nao esta
"mal treinado" -- ele memorizou ruido. Essa e a definicao operacional de
"nao ha sinal", e e o que separa um problema de MODELAGEM de um problema de
DADOS.

GUARDRAIL: todas as comparacoes usam o MESMO bloco de teste (armadilha 4 do
CLAUDE.md -- R2 nao e comparavel entre amostras de composicao diferente), e o
treino e sempre PURGADO do bloco de teste por horizon + embargo.
"""

from __future__ import annotations

import pandas as pd

from vol.forecast import fit_har, forecast_metrics, predict

# Capacidades pre-registradas para a varredura. Nao sao ajustadas: sao uma
# escada de "cada vez mais poder de representacao", justamente para mostrar
# que o R2 fora da amostra NAO acompanha.
CAPACITY_GRID: list[dict] = [
    {"rotulo": "arvore rasa",   "max_depth": 2,  "n_estimators": 100},
    {"rotulo": "arvore media",  "max_depth": 4,  "n_estimators": 300},
    {"rotulo": "arvore funda",  "max_depth": 8,  "n_estimators": 600},
    {"rotulo": "arvore enorme", "max_depth": 12, "n_estimators": 1500},
]


def _split_fixo(
    dataset: pd.DataFrame, test_size: int, train_size: int | None, horizon: int, embargo_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bloco de teste SEMPRE o mesmo (as ultimas `test_size` linhas); o treino
    e a fatia imediatamente anterior, purgada por horizon + embargo.

    `train_size=None` usa tudo o que houver antes da purga.
    """
    n = len(dataset)
    test = dataset.iloc[n - test_size :]
    fim_treino = n - test_size - (horizon + embargo_days)
    if fim_treino <= 0:
        raise ValueError("dataset pequeno demais para o bloco de teste pedido")
    inicio = 0 if train_size is None else max(0, fim_treino - train_size)
    return dataset.iloc[inicio:fim_treino], test


def learning_curve(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    train_sizes: list[int],
    test_size: int = 400,
    embargo_days: int = 5,
    log_target: bool = True,
) -> pd.DataFrame:
    """R2/RMSE fora da amostra em funcao do tamanho do treino, com bloco de
    teste FIXO. Curva plana => mais dados nao resolvem."""
    linhas = []
    for tam in train_sizes:
        try:
            train, test = _split_fixo(dataset, test_size, tam, horizon, embargo_days)
        except ValueError:
            continue
        if len(train) < 50:
            continue
        modelo = fit_har(train, feature_cols, log_target=log_target)
        pred = predict(modelo, test, feature_cols, log_target=log_target)
        m = forecast_metrics(pred, test["target"])
        linhas.append({"n_treino": len(train), "n_teste": len(test), **m})
    return pd.DataFrame(linhas)


def in_sample_ceiling(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    test_size: int = 400,
    embargo_days: int = 5,
    log_target: bool = True,
) -> dict:
    """TETO: ajusta e avalia o HAR no MESMO bloco (com permissao para colar).

    O R2 daqui e o maximo que a especificacao linear consegue extrair daquelas
    features naquele periodo. Comparar com o R2 fora da amostra separa duas
    causas muito diferentes de desempenho ruim: teto BAIXO significa que nao ha
    relacao a aprender; teto ALTO com OOS baixo significa que a relacao existe
    mas nao e estavel no tempo.
    """
    _, test = _split_fixo(dataset, test_size, None, horizon, embargo_days)
    modelo = fit_har(test, feature_cols, log_target=log_target)
    pred = predict(modelo, test, feature_cols, log_target=log_target)
    dentro = forecast_metrics(pred, test["target"])
    return {"n": len(test), **dentro}


def capacity_sweep(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    test_size: int = 400,
    embargo_days: int = 5,
    grid: list[dict] | None = None,
) -> pd.DataFrame:
    """R2 DENTRO e FORA da amostra para capacidades crescentes de XGBoost.

    Reporta os dois de proposito: e a divergencia entre eles -- dentro subindo
    para perto de 1,0 enquanto fora nao sai do lugar -- que demonstra que o
    modelo esta memorizando ruido, e nao que esta "pouco treinado".
    """
    from vol.ml_forecast import fit_xgb, predict_xgb

    grid = grid or CAPACITY_GRID
    train, test = _split_fixo(dataset, test_size, None, horizon, embargo_days)

    linhas = []
    for cfg in grid:
        params = {
            "booster": "gbtree",
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": 0.05,
            "min_child_weight": 1,   # deliberadamente permissivo: queremos ver
            "subsample": 1.0,        # a capacidade MAXIMA, nao a regularizada
            "colsample_bytree": 1.0,
            "reg_lambda": 0.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": 1,
            "verbosity": 0,
        }
        modelo = fit_xgb(train, feature_cols, params=params, log_target=True)
        dentro = forecast_metrics(
            predict_xgb(modelo, train, feature_cols, log_target=True), train["target"]
        )
        fora = forecast_metrics(
            predict_xgb(modelo, test, feature_cols, log_target=True), test["target"]
        )
        linhas.append(
            {
                "capacidade": cfg["rotulo"],
                "max_depth": cfg["max_depth"],
                "n_estimators": cfg["n_estimators"],
                "r2_dentro": dentro["r2_oos"],
                "r2_fora": fora["r2_oos"],
                "rmse_dentro": dentro["rmse"],
                "rmse_fora": fora["rmse"],
            }
        )
    return pd.DataFrame(linhas)


def target_autocorrelation(daily_variance: pd.Series, lags: list[int]) -> pd.DataFrame:
    """Autocorrelacao da variancia diaria nos `lags` pedidos.

    E o numero que fecha o argumento: se a variancia de hoje nao tem relacao
    com a de daqui a 21 dias, o alvo de 21 dias nao e previsivel a partir do
    passado, por nenhum modelo. Reportado ao lado dos diagnosticos acima
    porque explica o PORQUE deles.
    """
    return pd.DataFrame(
        [{"lag": lag, "autocorrelacao": float(daily_variance.autocorr(lag))} for lag in lags]
    )