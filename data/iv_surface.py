"""Coletor da superficie de volatilidade de dolar (IV baseline) publicada pela B3.

Fonte: pagina "Precos referenciais > Superficie de volatilidade de dolar", construida
pela B3 com base em informacoes coletadas por um pool de Informantes as 18h.
O arquivo e um .zip com uma planilha .xlsx, sempre disponivel na mesma URL (a B3
sobrescreve o conteudo diariamente) -- por isso o coletor precisa rodar uma vez por
dia e arquivar cada snapshot por data para construir historico.

Layout da planilha (celula A1 = data de referencia da superficie; B1:L1 = percentis
de delta; demais linhas = vencimento x vol implicita, em pontos percentuais ao ano):

           A           B      C     ...    L
    1  2026-07-21      1      5     ...    99
    2  2026-08-03  13.44  12.49     ...  9.09
    3  2026-09-01  14.35  13.90     ...  9.04
    ...
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime
from pathlib import Path

import openpyxl
import pandas as pd
import requests

URL = (
    "https://www.b3.com.br/data/files/16/35/6A/F9/623589100A29E189AC094EA8/"
    "Superficie-de-volatilidade-de-dolar.zip"
)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "iv_surface"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "iv_surface.parquet"

TIMEZONE = "America/Sao_Paulo"


def _download_zip_bytes() -> bytes:
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    return response.content


def _extract_xlsx_bytes(zip_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        return zf.read(names[0])


def _parse_workbook(xlsx_bytes: bytes) -> tuple[date, pd.DataFrame]:
    """Parse puro do xlsx (sem I/O de rede) para (refdate, DataFrame long)."""
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    refdate = rows[0][0]
    if isinstance(refdate, datetime):
        refdate = refdate.date()

    deltas = [float(d) for d in rows[0][1:] if d is not None]

    records = []
    for row in rows[1:]:
        maturity = row[0]
        if not isinstance(maturity, datetime):
            continue
        maturity = maturity.date()
        for delta_pct, iv in zip(deltas, row[1:]):
            if not isinstance(iv, (int, float)):
                continue
            records.append((maturity, delta_pct, float(iv)))

    df = pd.DataFrame(records, columns=["maturity_date", "delta_pct", "iv_pct"])
    return refdate, df


def _raw_path(refdate: date) -> Path:
    return RAW_DIR / f"iv_surface_{refdate.isoformat()}.xlsx"


def fetch_iv_surface_raw() -> tuple[date, Path]:
    """Baixa a superficie de vol publicada (URL sempre retorna o snapshot mais
    recente) e arquiva em disco por data de referencia, de forma idempotente:
    se ja existe um raw file para a data encontrada, nao sobrescreve.
    """
    zip_bytes = _download_zip_bytes()
    xlsx_bytes = _extract_xlsx_bytes(zip_bytes)
    refdate, _ = _parse_workbook(xlsx_bytes)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_path(refdate)
    if not path.exists():
        path.write_bytes(xlsx_bytes)
    return refdate, path


def load_iv_surface_processed() -> pd.DataFrame:
    """Garante o snapshot do dia em cache, monta o DataFrame limpo (long format)
    e faz upsert no parquet processado (mantendo o historico acumulado de outras
    datas ja coletadas).

    Colunas: refdate (tz-aware America/Sao_Paulo), maturity_date, delta_pct, iv_pct.
    """
    refdate, path = fetch_iv_surface_raw()
    _, day_df = _parse_workbook(path.read_bytes())
    day_df.insert(0, "refdate", pd.Timestamp(refdate).tz_localize(TIMEZONE))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if PROCESSED_PATH.exists():
        history = pd.read_parquet(PROCESSED_PATH)
        combined = pd.concat([history, day_df], ignore_index=True)
    else:
        combined = day_df

    combined = combined.sort_values(["refdate", "maturity_date", "delta_pct"])
    combined = combined.drop_duplicates(["refdate", "maturity_date", "delta_pct"]).reset_index(
        drop=True
    )
    combined.to_parquet(PROCESSED_PATH, index=False)
    return combined
