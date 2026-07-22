"""Wrapper fino sobre o FinBERT-PT-BR (transfer learning, sem treinar nada do
zero -- ver CLAUDE.md: ML e o ponto mais fraco do time).

https://huggingface.co/lucas-leme/FinBERT-PT-BR
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

MODEL_ID = "lucas-leme/FinBERT-PT-BR"

_SIGN = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}

_pipeline = None


def _get_pipeline():
    """Carrega o pipeline da HuggingFace sob demanda (lazy), para nao pagar o
    custo de carregar o modelo (~ligacao de rede/disco) em todo import."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline

        pipe = pipeline(task="text-classification", model=MODEL_ID)
        # truncation precisa ser passado na chamada (nao na construcao) para
        # ser de fato aplicado pelo tokenizer nesta versao do transformers --
        # sem isso, atas longas do Copom estouram o limite de 512 tokens do BERT.
        _pipeline = lambda texts: pipe(texts, truncation=True, max_length=512)
    return _pipeline


def score_texts(
    texts: list[str], pipeline_fn: Callable[[list[str]], list[dict]] | None = None
) -> pd.DataFrame:
    """Classifica uma lista de textos com o FinBERT-PT-BR.

    `pipeline_fn` e injetavel para testes (evita carregar o modelo real); por
    padrao usa o pipeline da HuggingFace.

    Colunas: text, label (POSITIVE|NEGATIVE|NEUTRAL), score (confianca do
    label), signed_score (score assinado pelo label: + positivo, - negativo,
    0 neutro -- pronto para agregar num indice diario).
    """
    scorer = pipeline_fn or _get_pipeline()
    results = scorer(texts)

    df = pd.DataFrame(results)
    df.insert(0, "text", texts)
    df["signed_score"] = df["label"].map(_SIGN) * df["score"]
    return df
