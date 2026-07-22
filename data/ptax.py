"""Coletor de PTAX (USD/BRL) via API SGS do Banco Central do Brasil.

Serie 1 = dolar americano (venda), serie 10813 = dolar americano (compra).
Endpoint: https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

SeriesName = Literal["venda", "compra"]

SERIES_CODES: dict[SeriesName, int] = {"venda": 1, "compra": 10813}

# A API SGS limita o intervalo entre dataInicial e dataFinal a 10 anos.
MAX_WINDOW_DAYS = 365 * 10 - 30

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "ptax"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "ptax.parquet"

TIMEZONE = "America/Sao_Paulo"


def _sgs_base_url() -> str:
    return os.environ.get("BCB_SGS_BASE_URL", "https://api.bcb.gov.br/dados/serie")


def _date_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Quebra [start, end] em janelas de no maximo MAX_WINDOW_DAYS dias."""
    windows = []
    window_start = start
    while window_start <= end:
        window_end = min(end, window_start + pd.Timedelta(days=MAX_WINDOW_DAYS))
        windows.append((window_start, window_end))
        window_start = window_end + pd.Timedelta(days=1)
    return windows


def _raw_path(series: SeriesName, start: date, end: date) -> Path:
    return RAW_DIR / f"ptax_{series}_{start.isoformat()}_{end.isoformat()}.json"


def fetch_ptax_raw(start: date, end: date, series: SeriesName = "venda") -> list[Path]:
    """Baixa o JSON cru da API SGS para o intervalo pedido, com cache idempotente em disco.

    Retorna a lista de arquivos raw (um por janela de ate 10 anos) cobrindo o intervalo.
    Se o arquivo de uma janela ja existe, nao refaz a chamada de rede.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    codigo = SERIES_CODES[series]
    paths: list[Path] = []

    for window_start, window_end in _date_windows(start, end):
        path = _raw_path(series, window_start, window_end)
        if path.exists():
            paths.append(path)
            continue

        url = f"{_sgs_base_url()}/bcdata.sgs.{codigo}/dados"
        params = {
            "formato": "json",
            "dataInicial": window_start.strftime("%d/%m/%Y"),
            "dataFinal": window_end.strftime("%d/%m/%Y"),
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        path.write_text(json.dumps(response.json()), encoding="utf-8")
        paths.append(path)

    return paths


def _parse_raw_file(path: Path, series: SeriesName) -> pd.DataFrame:
    """Parse puro de um JSON cru da SGS para DataFrame (sem I/O de rede)."""
    records = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["date", "value", "tipo"])

    df["date"] = pd.to_datetime(df["data"], format="%d/%m/%Y").dt.tz_localize(TIMEZONE)
    df["value"] = df["valor"].astype(float)
    df["tipo"] = series
    return df[["date", "value", "tipo"]]


def load_ptax_processed(
    start: date,
    end: date,
    series: tuple[SeriesName, ...] = ("venda", "compra"),
) -> pd.DataFrame:
    """Garante os dados raw em cache, monta o DataFrame limpo e salva parquet processado.

    Colunas: date (tz-aware America/Sao_Paulo), value, tipo (venda|compra).
    """
    frames = []
    for s in series:
        for path in fetch_ptax_raw(start, end, s):
            frames.append(_parse_raw_file(path, s))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["tipo", "date"]).drop_duplicates(["tipo", "date"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    return df
