"""Coletor de noticias globais via GDELT DOC 2.0 API.

Cobre o lado "notas globais" do CLAUDE.md: GDELT ja calcula um score de tom por
artigo (nao exposto artigo a artigo, mas agregado por dia via mode=timelinetone)
e tambem devolve a lista de manchetes que compoem a busca (mode=artlist), util
como evidencia/auditoria por tras do indice agregado.

Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
A API pede no maximo 1 request a cada 5s (throttle e responsabilidade de quem
chama o coletor, ex.: rodar uma vez por dia).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Cobertura de risco/cambio do Brasil na imprensa global -- ajustavel por chamada.
DEFAULT_QUERY = "(Brazil OR Brazilian) (real OR currency OR economy)"

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "gdelt"
PROCESSED_DIR = BASE_DIR / "processed"
TONE_PROCESSED_PATH = PROCESSED_DIR / "gdelt_tone.parquet"
HEADLINES_PROCESSED_PATH = PROCESSED_DIR / "gdelt_headlines.parquet"

TIMEZONE = "America/Sao_Paulo"


def _query_slug(query: str) -> str:
    return hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]


def _gdelt_datetime(d: date, end_of_day: bool = False) -> str:
    t = time(23, 59, 59) if end_of_day else time(0, 0, 0)
    return datetime.combine(d, t).strftime("%Y%m%d%H%M%S")


def _raw_path(mode: str, query: str, start: date, end: date) -> Path:
    return RAW_DIR / f"{mode}_{_query_slug(query)}_{start.isoformat()}_{end.isoformat()}.json"


def _fetch_raw(mode: str, start: date, end: date, query: str, extra_params: dict) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_path(mode, query, start, end)
    if path.exists():
        return path

    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": _gdelt_datetime(start),
        "enddatetime": _gdelt_datetime(end, end_of_day=True),
        **extra_params,
    }
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return path


def fetch_gdelt_tone_raw(start: date, end: date, query: str = DEFAULT_QUERY) -> Path:
    """Baixa a serie diaria de tom medio (mode=timelinetone), com cache idempotente."""
    return _fetch_raw("timelinetone", start, end, query, extra_params={})


def fetch_gdelt_headlines_raw(
    start: date, end: date, query: str = DEFAULT_QUERY, maxrecords: int = 250
) -> Path:
    """Baixa a lista de manchetes que casam com a busca (mode=artlist), com cache."""
    return _fetch_raw(
        "artlist", start, end, query, extra_params={"maxrecords": maxrecords, "sort": "datedesc"}
    )


def _parse_tone_raw(path: Path, query: str) -> pd.DataFrame:
    """Parse puro do JSON de timelinetone para DataFrame (sem I/O de rede)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    timeline = data.get("timeline", [])
    if not timeline:
        return pd.DataFrame(columns=["date", "tone", "query"])

    series = timeline[0]["data"]
    df = pd.DataFrame(series)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%SZ", utc=True).dt.tz_convert(
        TIMEZONE
    )
    df = df.rename(columns={"value": "tone"})
    df["query"] = query
    return df[["date", "tone", "query"]]


def _parse_headlines_raw(path: Path, query: str) -> pd.DataFrame:
    """Parse puro do JSON de artlist para DataFrame (sem I/O de rede)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    articles = data.get("articles", [])
    if not articles:
        return pd.DataFrame(
            columns=["date", "title", "url", "domain", "language", "sourcecountry", "query"]
        )

    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", utc=True).dt.tz_convert(
        TIMEZONE
    )
    df["query"] = query
    return df[["date", "title", "url", "domain", "language", "sourcecountry", "query"]]


def load_gdelt_tone_processed(start: date, end: date, query: str = DEFAULT_QUERY) -> pd.DataFrame:
    """Garante o raw em cache, monta o DataFrame limpo e faz upsert no parquet
    processado (mantendo historico de outras janelas/queries ja coletadas)."""
    path = fetch_gdelt_tone_raw(start, end, query)
    day_df = _parse_tone_raw(path, query)
    return _upsert_processed(day_df, TONE_PROCESSED_PATH, key_cols=["query", "date"])


def load_gdelt_headlines_processed(
    start: date, end: date, query: str = DEFAULT_QUERY
) -> pd.DataFrame:
    """Garante o raw em cache, monta o DataFrame limpo e faz upsert no parquet
    processado (mantendo historico de outras janelas/queries ja coletadas)."""
    path = fetch_gdelt_headlines_raw(start, end, query)
    day_df = _parse_headlines_raw(path, query)
    return _upsert_processed(day_df, HEADLINES_PROCESSED_PATH, key_cols=["query", "url"])


def _upsert_processed(new_df: pd.DataFrame, path: Path, key_cols: list[str]) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history = pd.read_parquet(path)
        combined = pd.concat([history, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values("date").drop_duplicates(key_cols, keep="last")
    combined = combined.reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return combined
