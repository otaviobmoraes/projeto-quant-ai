"""Coletor de cambio USD/BRL via yfinance (ticker BRL=X).

Conforme CLAUDE.md, o futuro de dolar (DOL/WDO) da B3 e a fonte primaria; este
modulo cobre o "apoio" citado no edital: uma serie de cambio publica e estavel,
util como fallback/cross-check enquanto o coletor de futuro B3 nao existe.

yfinance nao fornece o horario exato de fechamento da B3, entao a data de
referencia e tratada como dia de pregao e localizada em America/Sao_Paulo por
convencao do projeto (ver PTAX), nao como o instante exato do fechamento.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKER = "BRL=X"

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "fx_spot"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "fx_spot.parquet"

TIMEZONE = "America/Sao_Paulo"


def _raw_path(start: date, end: date) -> Path:
    return RAW_DIR / f"fx_spot_{start.isoformat()}_{end.isoformat()}.parquet"


def fetch_fx_spot_raw(start: date, end: date) -> Path:
    """Baixa o historico diario de BRL=X via yfinance, com cache idempotente em disco.

    yfinance trata `end` como exclusivo, entao pedimos end+1 dia para incluir o
    dia final. Se o arquivo de cache ja existe, nao refaz o download.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_path(start, end)
    if path.exists():
        return path

    raw = yf.download(
        TICKER,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    raw.to_parquet(path)
    return path


def _parse_raw_file(path: Path) -> pd.DataFrame:
    """Parse puro do parquet cru do yfinance para o formato limpo do projeto."""
    raw = pd.read_parquet(path)
    if raw.empty:
        return pd.DataFrame(columns=["date", "close"])

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].rename(columns={"Close": "close"}).reset_index()
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(TIMEZONE)
    return df[["date", "close"]]


def load_fx_spot_processed(start: date, end: date) -> pd.DataFrame:
    """Garante os dados raw em cache, monta o DataFrame limpo e salva parquet processado."""
    path = fetch_fx_spot_raw(start, end)
    df = _parse_raw_file(path)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    return df
