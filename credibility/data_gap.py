"""Hiato do produto (output gap) a partir do IBC-Br (BCB), via filtro HP com
JANELA EXPANSIVA -- guardrail central desta fase (credibility/CLAUDE.md):
"nunca rodar HP no full sample e usar como se fosse informacao disponivel
em tempo real. Recalcular expanding-window."

IBC-Br: SGS serie 24363, mensal, desde 01/2003. Tem defasagem REAL de
divulgacao (diferente do Focus, ver data_focus.py): o valor do mes de
referencia M so e publicado por volta de 60 dias depois. Por isso esta
serie tem duas datas: `reference_date` (mes que o indice mede) e
`release_date` (aproximacao de quando o dado ficou publico) -- SEMPRE
alinhar outras series por `release_date`, nunca por `reference_date`, senao
e look-ahead bias disfarcado.

O calculo do hiato busca historico desde o inicio da serie (2003) mesmo que
o chamador so precise de uma janela recente -- isso garante que a janela
expansiva do HP ja tenha bastante historico "de aquecimento" ao chegar no
periodo de interesse, em vez de comecar do zero nele.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.filters.hp_filter import hpfilter

IBCBR_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.24363/dados"
IBCBR_SERIES_START = date(2003, 1, 1)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "ibcbr"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "output_gap.parquet"

TIMEZONE = "America/Sao_Paulo"
HP_LAMBDA_MONTHLY = 14400  # Ravn-Uhlig (2002), padrao pra dados mensais
PUBLICATION_LAG_DAYS = 60  # aproximacao conservadora do atraso de divulgacao do IBC-Br
MIN_WINDOW = 24  # minimo de observacoes (meses) pra comecar a calcular o HP expansivo

MAX_WINDOW_DAYS = 365 * 10 - 30  # limite da API SGS: no maximo ~10 anos por chamada


def _date_windows(start: date, end: date) -> list[tuple[date, date]]:
    windows = []
    window_start = start
    while window_start <= end:
        window_end = min(end, window_start + pd.Timedelta(days=MAX_WINDOW_DAYS))
        windows.append((window_start, window_end))
        window_start = window_end + pd.Timedelta(days=1)
    return windows


def _raw_path(start: date, end: date) -> Path:
    return RAW_DIR / f"ibcbr_{start.isoformat()}_{end.isoformat()}.json"


def fetch_ibcbr_raw(start: date, end: date) -> list[Path]:
    """Baixa o IBC-Br (SGS 24363) em janelas de ate 10 anos (limite da API),
    com cache idempotente por janela de datas.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for window_start, window_end in _date_windows(start, end):
        path = _raw_path(window_start, window_end)
        if not path.exists():
            params = {
                "formato": "json",
                "dataInicial": window_start.strftime("%d/%m/%Y"),
                "dataFinal": window_end.strftime("%d/%m/%Y"),
            }
            response = requests.get(IBCBR_SGS_URL, params=params, timeout=30)
            response.raise_for_status()
            path.write_text(response.text, encoding="utf-8")
        paths.append(path)
    return paths


def _parse_ibcbr_raw(paths: list[Path]) -> pd.DataFrame:
    """Parse puro do IBC-Br: data de referencia (mes) + valor, mais a data de
    divulgacao ESTIMADA (referencia + PUBLICATION_LAG_DAYS)."""
    frames = []
    for path in paths:
        records = json.loads(path.read_text(encoding="utf-8"))
        frames.append(pd.DataFrame(records))
    df = pd.concat(frames, ignore_index=True)

    df["reference_date"] = pd.to_datetime(df["data"], format="%d/%m/%Y").dt.tz_localize(TIMEZONE)
    df["value"] = df["valor"].astype(float)
    df["release_date"] = df["reference_date"] + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    df = df[["reference_date", "release_date", "value"]]
    return df.sort_values("reference_date").drop_duplicates("reference_date").reset_index(drop=True)


def expanding_hp_gap(
    log_value: pd.Series, lam: float = HP_LAMBDA_MONTHLY, min_window: int = MIN_WINDOW
) -> pd.Series:
    """Hiato (ciclo HP) calculado em janela expansiva: pra cada t >= min_window,
    roda o filtro HP em log_value[:t+1] e guarda so o ULTIMO ponto do ciclo
    estimado (o mais recente, que so usa dados ate t). Evita o vazamento de
    informacao futura de rodar HP no full sample de uma vez so.
    """
    values = log_value.to_numpy()
    gaps = np.full(len(values), np.nan)
    for i in range(min_window, len(values)):
        cycle, _trend = hpfilter(values[: i + 1], lamb=lam)
        gaps[i] = cycle[-1]
    return pd.Series(gaps, index=log_value.index)


def load_output_gap_processed(start: date, end: date) -> pd.DataFrame:
    """Garante o IBC-Br em cache (desde o inicio da serie, pra dar historico
    de aquecimento ao HP expansivo), calcula o hiato e devolve so a fatia
    entre `start` e `end`. Salva o parquet processado com o historico
    completo calculado (nao so a fatia pedida).

    Colunas: reference_date, release_date (usar esta pra alinhar com outras
    series -- nunca reference_date), value (IBC-Br), gap (ciclo HP
    expansivo, em log).
    """
    paths = fetch_ibcbr_raw(IBCBR_SERIES_START, end)
    df = _parse_ibcbr_raw(paths)
    df["gap"] = expanding_hp_gap(np.log(df["value"]))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)

    start_ts = pd.Timestamp(start, tz=TIMEZONE)
    end_ts = pd.Timestamp(end, tz=TIMEZONE)
    return df[(df["reference_date"] >= start_ts) & (df["reference_date"] <= end_ts)].reset_index(
        drop=True
    )
