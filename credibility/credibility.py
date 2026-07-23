"""Indice de credibilidade BASELINE do Banco Central (Tier 1 -- sem NLP,
sem jogo ainda, ver credibility/CLAUDE.md).

theta_t in (0,1]: 1 = alta credibilidade (expectativas ancoradas na meta),
perto de 0 = baixa credibilidade (expectativas bem distantes da meta).

theta_t = exp(-|gap_t| / scale), onde gap_t = piᵉ_t - meta_t (ja calculado
em data_focus.py) e `scale` e um parametro FIXO (default 2.0 p.p.), nao uma
normalizacao estatistica pelo desvio-padrao historico do proprio gap.

Motivo: uma normalizacao "auto-calibrada" pelo desvio-padrao expansivo de
|gap| tem um problema serio quando o gap fica PERSISTENTEMENTE alto e pouco
volatil (exatamente o caso do Brasil, com expectativas cronicamente ~1-1.5
p.p. acima da meta) -- o desvio-padrao fica pequeno so porque o gap nao
oscila muito EM TORNO do proprio nivel elevado, fazendo |gap|/scale explodir
e theta colapsar perto de zero sempre, mesmo quando o desvio absoluto e
moderado. Isso mede "quao incomum e o gap vs. seu proprio historico", nao
"quao perto da meta estao as expectativas" -- nao e o que credibilidade
deveria capturar. Uma escala fixa (p.ex., 2.0 p.p. como o desvio "grande"
de referencia) e mais simples, mais interpretavel e evita essa degeneracao.

A dispersao do Focus (data_focus.py: `dispersion`) entra como uma SEGUNDA
feature separada (nao combinada em theta) -- "alta dispersao = baixa
credibilidade" e um sinal relacionado mas distinto (dispersao mede
DESACORDO entre analistas; theta mede DISTANCIA da meta), conforme
credibility/CLAUDE.md.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from credibility.data_focus import load_focus_processed

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "credibility.parquet"


def theta_baseline(gap: pd.Series, scale: float = 2.0) -> pd.Series:
    """theta_t = exp(-|gap_t| / scale). Pura funcao de dados (sem I/O),
    testavel isoladamente. `scale` e o desvio absoluto (em p.p. de inflacao)
    que conta como "grande" -- default 2.0 p.p. (documentado no modulo).
    """
    return np.exp(-gap.abs() / scale).rename("theta_baseline")


def load_credibility_processed(start: date, end: date) -> pd.DataFrame:
    """Garante o Focus em cache (data_focus.load_focus_processed), monta
    theta_baseline e organiza as features de credibilidade.

    Colunas: date (tz-aware America/Sao_Paulo), gap, dispersion, theta_baseline.
    """
    focus_df = load_focus_processed(start, end)
    df = focus_df[["date", "gap", "dispersion"]].copy()
    df["theta_baseline"] = theta_baseline(df["gap"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    return df
