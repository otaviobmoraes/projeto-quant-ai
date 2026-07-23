"""Coletor do Boletim Focus (BCB): expectativa de inflacao (piᵉ, rolante 12
meses a frente) e meta de inflacao (CMN) -- ver credibility/CLAUDE.md.

piᵉ: API Olinda (OData), recurso ExpectativasMercadoInflacao12Meses --
expectativa ROLANTE de 12 meses a frente (nao presa a ano-calendario). Mais
adequada como estado em tempo real pra alimentar uma curva de Phillips do
que a expectativa de fim de ano (essa fica artificialmente mais "ancorada"
conforme o ano avanca, so por causa do calendario, nao por credibilidade).

Convencao adotada (documentada, nao a unica possivel): baseCalculo=0 (usa
todas as respostas do mes) e Suavizada='N' (serie bruta, sem suavizacao
feita pelo BCB).

meta: SGS serie 13521 (meta de inflacao fixada pelo CMN, uma vez por ano).
Como piᵉ e rolante (cobre pedacos de dois anos-calendario), a meta usada
pra calcular o gap e uma MEDIA PONDERADA das metas dos dois anos que o
horizonte de 12 meses cobre, pela fracao de dias em cada um -- ver
`_blended_meta`.

O campo `Data` do Focus JA E a data de divulgacao (o BCB publica novo
consolidado a cada dia util) -- diferente do IBC-Br (ver data_gap.py), que
tem defasagem real entre mes de referencia e data de divulgacao. Por isso
aqui nao e preciso nenhum deslocamento adicional.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.parse import quote, urlencode

import pandas as pd
import requests

FOCUS_BASE_URL = (
    "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
    "ExpectativasMercadoInflacao12Meses"
)
META_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.13521/dados"

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "focus"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "focus.parquet"

TIMEZONE = "America/Sao_Paulo"


def _focus_raw_path(start: date, end: date) -> Path:
    return RAW_DIR / f"focus_ipca12m_{start.isoformat()}_{end.isoformat()}.json"


def fetch_focus_raw(start: date, end: date) -> Path:
    """Baixa expectativas de IPCA 12 meses a frente (Olinda OData), com cache
    idempotente por janela de datas. A API nao pagina nesse volume (testado
    ate ~3.5 anos numa chamada so), entao nao ha necessidade de quebrar em
    janelas menores como no coletor de PTAX.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _focus_raw_path(start, end)
    if path.exists():
        return path

    filter_expr = (
        f"Indicador eq 'IPCA' and Data ge '{start.isoformat()}' and Data le '{end.isoformat()}'"
    )
    params = {
        "$filter": filter_expr,
        "$format": "json",
        "$select": "Data,Suavizada,Media,Mediana,DesvioPadrao,numeroRespondentes,baseCalculo",
        "$top": 20000,
    }
    # requests codifica espaco como '+' por padrao, mas esse endpoint OData
    # so aceita '%20' (com '+' ele devolve um 400 de "types not compatible"
    # sem relacao nenhuma com o problema real) -- por isso montamos a query
    # string na mao em vez de deixar `params=` fazer a codificacao.
    query_string = urlencode(params, quote_via=quote)
    response = requests.get(f"{FOCUS_BASE_URL}?{query_string}", timeout=30)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return path


def _meta_raw_path() -> Path:
    return RAW_DIR / "meta_inflacao.json"


def fetch_meta_raw() -> Path:
    """Baixa a meta de inflacao (SGS 13521). A serie muda no maximo 1x/ano,
    entao o cache so refaz o download se o arquivo ainda nao existir --
    apague o arquivo manualmente se precisar forcar atualizacao.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _meta_raw_path()
    if path.exists():
        return path
    response = requests.get(META_SGS_URL, params={"formato": "json"}, timeout=30)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return path


def _parse_focus_raw(path: Path) -> pd.DataFrame:
    """Parse puro do JSON de expectativas (sem I/O de rede)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("value", [])
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(
            columns=["date", "pi_expectation", "pi_median", "dispersion", "n_respondentes"]
        )

    df = df[(df["Suavizada"] == "N") & (df["baseCalculo"] == 0)].copy()
    df["date"] = pd.to_datetime(df["Data"]).dt.tz_localize(TIMEZONE)
    df = df.rename(
        columns={
            "Media": "pi_expectation",
            "Mediana": "pi_median",
            "DesvioPadrao": "dispersion",
            "numeroRespondentes": "n_respondentes",
        }
    )
    cols = ["date", "pi_expectation", "pi_median", "dispersion", "n_respondentes"]
    return df[cols].sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _parse_meta_raw(path: Path) -> dict[int, float]:
    """Parse puro da meta de inflacao (SGS) -- {ano: meta}."""
    records = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    df["year"] = pd.to_datetime(df["data"], format="%d/%m/%Y").dt.year
    df["meta"] = df["valor"].astype(float)
    return dict(zip(df["year"], df["meta"]))


def _blended_meta(d: pd.Timestamp, meta_by_year: dict[int, float]) -> float:
    """Meta 'blend' pro horizonte rolante de 12 meses a partir de `d`: media
    das metas do ano corrente e do ano seguinte, ponderada pela fracao de
    dias do horizonte [d, d+365d] que cai em cada ano-calendario.
    """
    horizon_end = d + pd.Timedelta(days=365)
    year0, year1 = d.year, horizon_end.year
    if year0 == year1:
        return meta_by_year.get(year0, float("nan"))

    year_end = pd.Timestamp(year=year0 + 1, month=1, day=1, tz=d.tz)
    days_in_year0 = (year_end - d).days
    total_days = (horizon_end - d).days
    w0 = days_in_year0 / total_days

    m0 = meta_by_year.get(year0, float("nan"))
    # Se a meta do ano seguinte ainda nao foi fixada pelo CMN, usa a corrente
    # como proxy (metas raramente mudam de um ano pro outro).
    m1 = meta_by_year.get(year1, m0)
    return w0 * m0 + (1 - w0) * m1


def load_focus_processed(start: date, end: date) -> pd.DataFrame:
    """Garante Focus + meta em cache, monta o DataFrame limpo com a meta
    'blended' pro horizonte rolante e o gap (pi_expectation - meta).

    Colunas: date (tz-aware America/Sao_Paulo), pi_expectation, pi_median,
    dispersion, n_respondentes, meta, gap.
    """
    focus_path = fetch_focus_raw(start, end)
    focus_df = _parse_focus_raw(focus_path)

    meta_path = fetch_meta_raw()
    meta_by_year = _parse_meta_raw(meta_path)

    focus_df["meta"] = focus_df["date"].apply(lambda d: _blended_meta(d, meta_by_year))
    focus_df["gap"] = focus_df["pi_expectation"] - focus_df["meta"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    focus_df.to_parquet(PROCESSED_PATH, index=False)
    return focus_df
