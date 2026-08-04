"""Walk-forward purgado com embargo (Lopez de Prado) para datasets com alvo
de horizonte fixo (o rotulo em t depende dos dados ate t+horizon -- ver
vol/forecast.py:forward_target).

Guardrail do CLAUDE.md: NUNCA usar split aleatorio em serie temporal. Aqui o
split e sempre cronologico (treino antes do teste) e, alem disso, PURGA do
treino qualquer linha cujo rotulo se sobreponha ao periodo de teste -- sem
isso, o modelo "veria" no treino informacao que so existe depois do corte
(vazamento por rotulo com janela futura), inflando a performance fora da
amostra de forma artificial. `embargo_days` adiciona uma margem extra de
seguranca alem da purga estritamente necessaria.
"""

from __future__ import annotations

import pandas as pd


def purged_walk_forward_splits(
    dataset: pd.DataFrame, n_splits: int, horizon: int, embargo_days: int = 0
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Divide `dataset` (uma linha por dia util, em ordem cronologica) em
    `n_splits` folds de janela expansiva. Cada fold treina com tudo antes do
    bloco de teste, MENOS as ultimas `horizon + embargo_days` linhas
    (purgadas por causa da sobreposicao de rotulo), e testa num bloco
    contiguo subsequente.
    """
    n = len(dataset)
    fold_size = n // (n_splits + 1)
    if fold_size < 1:
        raise ValueError(f"dataset com {n} linhas e pequeno demais para {n_splits} folds")

    purge = horizon + embargo_days
    splits = []
    for i in range(1, n_splits + 1):
        test_start = fold_size * i
        test_end = min(n, fold_size * (i + 1))
        train_end = max(0, test_start - purge)

        train = dataset.iloc[:train_end]
        test = dataset.iloc[test_start:test_end]
        if len(train) == 0 or len(test) == 0:
            continue
        splits.append((train, test))
    return splits


def purged_walk_forward_splits_by_step(
    dataset: pd.DataFrame, min_train_size: int, step_size: int, horizon: int, embargo_days: int = 0
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Walk-forward purgado com PASSO FIXO (`step_size` linhas de teste por
    fold, ex.: ~21 = 1 mes de pregao) em vez de numero de folds fixo -- gera
    muito mais folds que `purged_walk_forward_splits` (5 grandes), reajustando
    os coeficientes com mais frequencia. Motivacao: uma feature que so importa
    durante um regime especifico (ex.: risco fiscal durante estresse) fica
    diluida por um coeficiente unico congelado por ~130 dias; reajustar todo
    mes deixa o peso da feature reagir mais rapido a mudanca de regime.

    `min_train_size`: tamanho minimo do treino ANTES do primeiro fold de
    teste (ex.: ~252 = 1 ano). Sem isso, comecar a reestimar desde o
    principio geraria folds iniciais com treino minusculo e instavel (poucas
    dezenas de linhas pra ajustar uma regressao com varios parametros).
    """
    n = len(dataset)
    purge = horizon + embargo_days
    splits = []
    test_start = min_train_size
    while test_start < n:
        test_end = min(n, test_start + step_size)
        train_end = max(0, test_start - purge)

        train = dataset.iloc[:train_end]
        test = dataset.iloc[test_start:test_end]
        if len(train) > 0 and len(test) > 0:
            splits.append((train, test))
        test_start = test_end
    return splits
