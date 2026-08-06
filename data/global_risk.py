"""Coletor de indicadores de risco global via yfinance: VIX (^VIX) e indice
dolar DXY (DX-Y.NYB).

Hipotese testada (pesquisa de literatura, ver report/run_report.py
CONFIGS_TESTED): volatilidade de cambio de mercado emergente (USD/BRL) reage
a choques de risco GLOBAL, nao so ao proprio historico de preco -- VIX alto
costuma vir acompanhado de fuga de capital de mercados emergentes; DXY forte
(dolar se apreciando globalmente) costuma coincidir com estresse em moedas
EM. Diferente de tudo testado ate agora no projeto (overnight, leverage,
noticia, credibilidade), que e derivado do proprio preco/imprensa do
USD/BRL -- aqui o dado e EXOGENO.

Mesmo padrao idempotente/cacheado de data/fx_spot.py.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = {"vix": "^VIX", "dxy": "DX-Y.NYB"}

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "global_risk"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "global_risk.parquet"

TIMEZONE = "America/Sao_Paulo"


def _raw_path(start: date, end: date) -> Path:
    return RAW_DIR / f"global_risk_{start.isoformat()}_{end.isoformat()}.parquet"


def fetch_global_risk_raw(start: date, end: date) -> Path:
    """Baixa VIX + DXY via yfinance (uma chamada, os 2 tickers juntos), com
    cache idempotente em disco. yfinance trata `end` como exclusivo, entao
    pedimos end+1 dia para incluir o dia final.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_path(start, end)
    if path.exists():
        return path

    raw = yf.download(
        list(TICKERS.values()),
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    raw.to_parquet(path)
    return path


def _parse_raw_file(path: Path) -> pd.DataFrame:
    """Parse puro do parquet cru do yfinance (colunas MultiIndex Close x
    ticker) para o formato limpo do projeto -- sem I/O de rede.
    """
    raw = pd.read_parquet(path)
    if raw.empty:
        return pd.DataFrame(columns=["date", "vix_close", "dxy_close"])

    close = raw["Close"]
    df = pd.DataFrame(
        {
            "date": close.index,
            "vix_close": close[TICKERS["vix"]].to_numpy(),
            "dxy_close": close[TICKERS["dxy"]].to_numpy(),
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(TIMEZONE)
    return df.dropna(how="all", subset=["vix_close", "dxy_close"])


def load_global_risk_processed(start: date, end: date) -> pd.DataFrame:
    """Garante os dados raw em cache, monta o DataFrame limpo e salva parquet
    processado. Colunas: date (tz-aware America/Sao_Paulo), vix_close, dxy_close.
    """
    path = fetch_global_risk_raw(start, end)
    df = _parse_raw_file(path)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    return df
