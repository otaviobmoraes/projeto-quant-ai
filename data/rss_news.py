"""Coletor de manchetes de veiculos financeiros brasileiros via RSS.

Cobre o lado "noticias BR" do CLAUDE.md (ao lado do Copom): fontes gratuitas,
sem chave, em RSS 2.0 padrao. O feed sempre traz so os itens mais recentes,
entao e cacheado por dia (rodar uma vez por dia) e o historico e acumulado
via upsert no parquet processado.
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests

FEEDS = {
    "infomoney": "https://www.infomoney.com.br/feed/",
    "g1_economia": "https://g1.globo.com/rss/g1/economia/",
    "moneytimes": "https://www.moneytimes.com.br/feed/",
}

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "rss_news"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "rss_news.parquet"

TIMEZONE = "America/Sao_Paulo"

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(raw_html: str) -> str:
    text = _TAG_RE.sub(" ", raw_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _raw_path(source: str, as_of: date) -> Path:
    return RAW_DIR / f"{source}_{as_of.isoformat()}.xml"


def fetch_rss_raw(source: str, as_of: date | None = None) -> Path:
    """Baixa o feed RSS de uma fonte, cacheado por dia (as_of)."""
    as_of = as_of or date.today()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_path(source, as_of)
    if path.exists():
        return path

    response = requests.get(FEEDS[source], headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def _parse_rss_raw(path: Path, source: str) -> pd.DataFrame:
    """Parse puro de um feed RSS 2.0 (sem I/O de rede)."""
    tree = ElementTree.parse(path)
    records = []
    for item in tree.getroot().iter("item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub_date_raw = item.findtext("pubDate")
        description_raw = item.findtext("description") or ""

        pub_date = parsedate_to_datetime(pub_date_raw) if pub_date_raw else None
        records.append(
            {
                "date": pub_date,
                "title": title.strip(),
                "link": link.strip(),
                "description": _strip_html(description_raw),
                "source": source,
            }
        )

    df = pd.DataFrame(records, columns=["date", "title", "link", "description", "source"])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(TIMEZONE)
    return df


def load_rss_processed(
    sources: tuple[str, ...] = tuple(FEEDS), as_of: date | None = None
) -> pd.DataFrame:
    """Garante os feeds do dia em cache, monta o DataFrame limpo e faz upsert
    no parquet processado (mantendo o historico acumulado de outros dias).

    Colunas: date (tz-aware America/Sao_Paulo), title, link, description, source.
    """
    frames = []
    for source in sources:
        path = fetch_rss_raw(source, as_of)
        frames.append(_parse_rss_raw(path, source))

    new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if PROCESSED_PATH.exists():
        history = pd.read_parquet(PROCESSED_PATH)
        combined = pd.concat([history, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values("date").drop_duplicates(["source", "link"], keep="last")
    combined = combined.reset_index(drop=True)
    combined.to_parquet(PROCESSED_PATH, index=False)
    return combined
