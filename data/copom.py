"""Coletor de atas e comunicados do Copom via API publica do Banco Central.

Endpoints (mesma API usada pelo site do BCB, sem chave):
- GET /copom/atas?quantidade=N              -> lista as N atas mais recentes
- GET /copom/atas_detalhes?nro_reuniao=N     -> texto completo de uma ata
- GET /copom/comunicados?quantidade=N        -> lista os N comunicados mais recentes
- GET /copom/comunicados_detalhes?nro_reuniao=N -> texto completo de um comunicado

As listas mudam a cada reuniao nova, entao sao cacheadas por dia (o padrao de
uso e rodar o coletor uma vez por dia). Os detalhes de uma reuniao especifica
sao imutaveis uma vez publicados, entao sao cacheados para sempre por
nro_reuniao (nunca refeitos).
"""

from __future__ import annotations

import html
import json
import re
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

BASE_URL = "https://www.bcb.gov.br/api/servico/sitebcb/copom"

Kind = Literal["atas", "comunicados"]

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "copom"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "copom.parquet"

TIMEZONE = "America/Sao_Paulo"

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(raw_html: str) -> str:
    text = _TAG_RE.sub(" ", raw_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _list_raw_path(kind: Kind, quantidade: int, as_of: date) -> Path:
    return RAW_DIR / f"{kind}_list_{quantidade}_{as_of.isoformat()}.json"


def _detail_raw_path(kind: Kind, nro_reuniao: int) -> Path:
    return RAW_DIR / f"{kind}_detalhe_{nro_reuniao}.json"


def fetch_copom_list_raw(kind: Kind, quantidade: int = 50, as_of: date | None = None) -> Path:
    """Baixa a lista das reunioes mais recentes, cacheada por dia (as_of)."""
    as_of = as_of or date.today()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _list_raw_path(kind, quantidade, as_of)
    if path.exists():
        return path

    response = requests.get(f"{BASE_URL}/{kind}", params={"quantidade": quantidade}, timeout=30)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return path


def fetch_copom_detail_raw(kind: Kind, nro_reuniao: int) -> Path:
    """Baixa o texto completo de uma reuniao, cacheado para sempre (imutavel)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _detail_raw_path(kind, nro_reuniao)
    if path.exists():
        return path

    response = requests.get(
        f"{BASE_URL}/{kind}_detalhes", params={"nro_reuniao": nro_reuniao}, timeout=30
    )
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return path


def _parse_list_raw(path: Path) -> list[int]:
    """Retorna os numeros de reuniao presentes numa resposta de lista."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item.get("nroReuniao", item.get("nro_reuniao")) for item in data.get("conteudo", [])]


def _parse_detail_raw(path: Path, kind: Kind) -> dict:
    """Parse puro do detalhe de uma reuniao para um registro plano."""
    data = json.loads(path.read_text(encoding="utf-8"))
    item = data["conteudo"][0]

    if kind == "atas":
        return {
            "nro_reuniao": item["nroReuniao"],
            "tipo": "ata",
            "data_referencia": item["dataReferencia"],
            "titulo": item["titulo"],
            "texto": _strip_html(item["textoAta"]),
            "url_pdf": item.get("urlPdfAta"),
        }
    return {
        "nro_reuniao": item["nro_reuniao"],
        "tipo": "comunicado",
        "data_referencia": item["dataReferencia"],
        "titulo": item["titulo"],
        "texto": _strip_html(item["textoComunicado"]),
        "url_pdf": None,
    }


def load_copom_processed(quantidade: int = 50) -> pd.DataFrame:
    """Garante listas + detalhes em cache e monta o corpus limpo (long format),
    fazendo upsert no parquet processado por (nro_reuniao, tipo).

    Colunas: nro_reuniao, tipo (ata|comunicado), data_referencia (tz-aware
    America/Sao_Paulo), titulo, texto (HTML removido), url_pdf.
    """
    records = []
    for kind in ("atas", "comunicados"):
        list_path = fetch_copom_list_raw(kind, quantidade)
        for nro_reuniao in _parse_list_raw(list_path):
            detail_path = fetch_copom_detail_raw(kind, nro_reuniao)
            records.append(_parse_detail_raw(detail_path, kind))

    new_df = pd.DataFrame(records)
    new_df["data_referencia"] = pd.to_datetime(new_df["data_referencia"]).dt.tz_localize(TIMEZONE)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if PROCESSED_PATH.exists():
        history = pd.read_parquet(PROCESSED_PATH)
        combined = pd.concat([history, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values("data_referencia").drop_duplicates(
        ["nro_reuniao", "tipo"], keep="last"
    )
    combined = combined.reset_index(drop=True)
    combined.to_parquet(PROCESSED_PATH, index=False)
    return combined
