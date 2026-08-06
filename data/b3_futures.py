"""Coletor de precos de AJUSTE do futuro de dolar (DOL) direto dos arquivos
diarios da B3 (BVBG-086, "Arquivo de Precos de Mercado").

Motivacao: o resto do projeto usa o spot USD/BRL (yfinance BRL=X, ver
data/fx_spot.py) como proxy do forward -- simplificacao explicita
documentada em strategy/sizing.py. Mas a estrategia negocia opcao SOBRE
FUTURO (Black-76), entao o instrumento correto e o futuro de dolar, e a B3
publica o preco de AJUSTE oficial dele. Este modulo serve pra validar o
proxy (comparar RV do futuro vs RV do spot) e, se compensar, substituir a
fonte de preco.

URL (verificada empiricamente, ~7-12 MB por dia):
    https://www.b3.com.br/pesquisapregao/download?filelist=PR{aammdd}.zip

Estrutura: zip -> zip interno -> 4 XMLs (BVBG.086.01). Os 4 sao snapshots
progressivos do mesmo pregao; o ULTIMO e o mais completo, e o unico lido
aqui.

ATENCAO ao tamanho: cada XML descompactado tem ~140 MB e o cache raw guarda
o zip de cada dia (~10 MB/dia). Uma janela de 1 ano (~250 dias uteis)
ocupa ~2.5 GB em data/raw/b3_futures/.

LIMITACAO IMPORTANTE (verificada): as OPCOES de dolar neste mesmo arquivo
NAO tem preco de ajuste publicado (0 de 2042 series numa data testada) e
praticamente nao negociam (10 de 2042, com 1-2 negocios cada). Por isso NAO
e possivel reconstruir historico de IV invertendo Black-76 a partir daqui,
como a Fase 2 do CLAUDE.md previa via BD_Arbit -- ver relatorio.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

URL_TEMPLATE = "https://www.b3.com.br/pesquisapregao/download?filelist=PR{stamp}.zip"

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "b3_futures"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "b3_dol_futures.parquet"

TIMEZONE = "America/Sao_Paulo"

def _has_content(payload: bytes) -> bool:
    """Em dia sem pregao (fim de semana/feriado) a B3 devolve 200 com um zip
    VAZIO (~22 bytes, so o registro de fim de arquivo) em vez de 404 -- o
    que conta como zip valido. Por isso a checagem e "zip valido E com pelo
    menos uma entrada", nao um limiar de bytes.
    """
    try:
        return len(zipfile.ZipFile(io.BytesIO(payload)).namelist()) > 0
    except zipfile.BadZipFile:
        return False

# Futuro de dolar: DOL + codigo de mes (1 letra) + ano (2 digitos), ex.
# DOLU26. As OPCOES tem sufixo adicional (DOLU26C005600), entao o `>` no
# fim do padrao garante que so o futuro casa.
_FUTURE_BLOCK_RE = re.compile(
    rb"<PricRpt>(?:(?!</PricRpt>).)*?<TckrSymb>(DOL[A-Z]\d{2})</TckrSymb>"
    rb"((?:(?!</PricRpt>).)*?)</PricRpt>",
    re.S,
)


def _field(block: bytes, tag: str) -> float | None:
    match = re.search(rf"<{tag}[^>]*>([^<>]+)</{tag}>".encode(), block)
    return float(match.group(1)) if match else None


def _stamp(d: date) -> str:
    return d.strftime("%y%m%d")


def _raw_path(d: date) -> Path:
    return RAW_DIR / f"PR{_stamp(d)}.zip"


def fetch_pr_bytes(d: date, max_retries: int = 4) -> bytes | None:
    """Baixa o pacote de precos do pregao de `d` e devolve os bytes (usa o
    cache em disco se existir). Devolve None se nao houver pregao nessa data
    (fim de semana/feriado).

    Faz retry com espera crescente em erro de rede: numa coleta de varios
    anos sao centenas de requisicoes, e uma falha transitoria no meio nao
    deve derrubar a coleta inteira.
    """
    path = _raw_path(d)
    if path.exists():
        return path.read_bytes()

    for attempt in range(max_retries):
        try:
            response = requests.get(URL_TEMPLATE.format(stamp=_stamp(d)), timeout=180)
            response.raise_for_status()
            return response.content if _has_content(response.content) else None
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    return None


def fetch_pr_raw(d: date) -> Path | None:
    """Versao que PERSISTE o pacote em data/raw/ (cache idempotente), para
    uso pontual. Em coletas longas prefira `fetch_pr_bytes` com
    `keep_raw=False` -- ver load_dol_futures_processed."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_path(d)
    if path.exists():
        return path

    payload = fetch_pr_bytes(d)
    if payload is None:
        return None
    path.write_bytes(payload)
    return path


def parse_dol_futures(zip_bytes: bytes, refdate: date) -> pd.DataFrame:
    """Parse PURO (sem I/O de rede) do pacote diario -> um registro por
    vencimento de futuro de dolar negociado naquele pregao.

    Colunas: date, ticker, settlement (preco de ajuste), low, high,
    open_interest, trades.
    """
    outer = zipfile.ZipFile(io.BytesIO(zip_bytes))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(outer.namelist()[0])))
    # Os 4 XMLs sao snapshots progressivos; o ultimo e o mais completo.
    payload = inner.read(sorted(inner.namelist())[-1])

    records = []
    for match in _FUTURE_BLOCK_RE.finditer(payload):
        ticker = match.group(1).decode()
        block = match.group(2)
        settlement = _field(block, "AdjstdQt")
        if settlement is None:
            continue  # vencimento sem ajuste publicado nesse dia
        records.append(
            {
                "date": pd.Timestamp(refdate, tz=TIMEZONE),
                "ticker": ticker,
                "settlement": settlement,
                # OHLC de verdade: FrstPric/LastPric sao o primeiro e o ultimo
                # NEGOCIO do pregao (nao snapshots no limite do dia, como no
                # yfinance) -- e o que torna possivel uma feature de gap
                # overnight legitima (abertura de hoje vs ajuste de ontem).
                "open": _field(block, "FrstPric"),
                "last": _field(block, "LastPric"),
                "low": _field(block, "MinPric"),
                "high": _field(block, "MaxPric"),
                "open_interest": _field(block, "OpnIntrst"),
                "trades": _field(block, "TradQty"),
            }
        )
    return pd.DataFrame(
        records,
        columns=[
            "date", "ticker", "settlement", "open", "last", "low", "high",
            "open_interest", "trades",
        ],
    )


def front_month_series(df: pd.DataFrame) -> pd.DataFrame:
    """Escolhe, em cada dia, o contrato MAIS LIQUIDO (maior numero de
    negocios; empate desfeito por posicoes em aberto) -- proxy robusto do
    "contrato ativo" sem precisar codificar as regras de vencimento da B3.

    ATENCAO: a serie resultante TROCA de contrato ao longo do tempo (rolagem).
    Comparar precos de dias em contratos diferentes gera saltos artificiais
    -- por isso `contract_changed` marca esses dias, pra quem for calcular
    retorno poder descartar a emenda.
    """
    cols = ["date", "ticker", "settlement", "open", "last", "low", "high", "contract_changed"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    ranked = df.sort_values(
        ["date", "trades", "open_interest"], ascending=[True, False, False], na_position="last"
    )
    front = ranked.groupby("date", as_index=False).first()
    front["contract_changed"] = front["ticker"] != front["ticker"].shift(1)
    front.loc[front.index[0], "contract_changed"] = False
    return front[cols]


def load_dol_futures_processed(
    start: date, end: date, keep_raw: bool = True, progress_every: int = 0
) -> pd.DataFrame:
    """Baixa o intervalo de pregoes, monta a serie do contrato mais liquido
    por dia e salva o parquet processado. Dias sem pregao sao pulados.

    `keep_raw=False`: NAO persiste os pacotes em data/raw/ -- so parseia em
    memoria e descarta. Cada pacote tem ~10 MB, entao uma coleta de 3 anos
    (~800 pregoes) ocuparia ~8 GB; como este repositorio fica dentro de uma
    pasta sincronizada (OneDrive), guardar isso dispararia sincronizacao
    pesada sem beneficio -- o parquet processado (poucos KB) e o que
    importa. Use keep_raw=True so em janelas curtas.

    `progress_every`: se > 0, imprime progresso a cada N pregoes coletados
    (util pra acompanhar coletas longas rodando em background).
    """
    frames = []
    day = start
    collected = 0
    while day <= end:
        if day.weekday() < 5:  # pula fim de semana sem nem tentar a rede
            if keep_raw:
                path = fetch_pr_raw(day)
                payload = path.read_bytes() if path is not None else None
            else:
                payload = fetch_pr_bytes(day)
            if payload is not None:
                frames.append(parse_dol_futures(payload, day))
                collected += 1
                if progress_every and collected % progress_every == 0:
                    print(f"  {collected} pregoes coletados (ultimo: {day.isoformat()})", flush=True)
        day += timedelta(days=1)

    all_contracts = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["date", "ticker", "settlement", "low", "high", "open_interest", "trades"]
        )
    )
    front = front_month_series(all_contracts)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    front.to_parquet(PROCESSED_PATH, index=False)
    return front
