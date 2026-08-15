"""Painel de cambio de MERCADO EMERGENTE via yfinance (OHLC diario).

POR QUE EXISTE: o diagnostico do projeto para o ML perder do HAR e TAMANHO DE
AMOSTRA -- a distancia do XGBoost para o HAR cresce monotonicamente com o
horizonte, acompanhando a queda de observacoes independentes de 1.760 (h=1)
para ~101 (h=21). Christensen, Siggaard & Veliyev (2023) obtem ganho de ML
sobre a linhagem HAR usando um PAINEL de constituintes do Dow com RV
intradiaria (dezenas de milhares de observacoes); aqui havia um ativo so.
A dimensao TEMPO ja esta esgotada (a extensao 830 -> 2.135 pregoes nao
melhorou nada, ver CONFIGS_TESTED); a dimensao SECAO CRUZADA esta intocada.

Referencia central: Bollerslev, Hood, Huss & Pedersen (2018), "Risk
Everywhere: Modeling and Managing Volatility", Review of Financial Studies
31(7):2729-2773 -- estimacao em PAINEL com fatores globais de vol produz
ganho fora da amostra estatisticamente significante sobre modelos ajustados
ativo a ativo, justamente porque reduz erro de estimacao.

USO DELIBERADO DE yfinance, apesar do defeito documentado no CLAUDE.md: o
defeito e no `close` (snapshot no limite do dia, correlacao 0,99998 com a
abertura), e a validacao contra o PTAX mostrou que `high`/`low` estao
corretamente datados (o PTAX cai dentro do range do mesmo dia em 98,9% dos
pregoes). Este modulo expoe OHLC completo mas o painel usa PARKINSON, que
depende so de high/low. Nao usar `open`/`close` daqui para construir feature.

LIMITE DE COMPARABILIDADE, declarado: as demais moedas sao SPOT 24h, enquanto
o alvo do projeto e o FUTURO da B3. O proprio projeto mediu que a
autocorrelacao da variancia difere entre os dois (lag 5/21: 0,190/0,074 no
spot contra 0,096/0,025 no futuro). Um modelo global treinado em spot pode
portanto aprender persistencia maior do que o futuro comporta. E uma razao
para o painel FALHAR, e esta registrada antes do teste, nao depois.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Todos na convencao USD/XXX (igual a BRL=X), entao "sobe" significa a mesma
# coisa em todas: dolar se fortalecendo contra a moeda emergente.
TICKERS: dict[str, str] = {
    "mxn": "MXN=X",
    "zar": "ZAR=X",
    "try": "TRY=X",
    "clp": "CLP=X",
    "cop": "COP=X",
    "inr": "INR=X",
    "pln": "PLN=X",
    "huf": "HUF=X",
}

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw" / "em_fx"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "em_fx.parquet"

TIMEZONE = "America/Sao_Paulo"

# FILTRO DE SANIDADE PRE-REGISTRADO (declarado ANTES de rodar o experimento do
# painel, e aplicado UNIFORMEMENTE a todas as moedas -- excluir uma moeda
# especifica depois de ver o resultado seria selecao pos-hoc, a armadilha 3
# deste projeto).
#
# Regra: um range intradiario de log(high/low) > 0,20 num par de cambio
# liquido nao e mercado, e tick sujo. Verificado na coleta: remove 4 dias em
# ~18 mil observacoes -- 2 do COP (2019-05-15 com low=32,0 contra nivel real
# ~3.300, erro de casa decimal de 100x; 2019-10-22 com low=343 contra ~3.405,
# erro de 10x), 1 do ZAR e 1 do TRY.
#
# RESSALVA HONESTA: o dia removido do TRY (2018, crise cambial turca) pode ser
# movimento GENUINO -- 23% de range intradiario e implausivel mas nao
# impossivel naquele episodio. Mantida a regra uniforme mesmo assim: 1 dia em
# 2.236 de uma moeda entre oito nao move um modelo empilhado, e um limiar
# unico e defensavel enquanto uma excecao caso a caso nao e.
MAX_LOG_RANGE = 0.20


def _raw_path(start: date, end: date) -> Path:
    return RAW_DIR / f"em_fx_{start.isoformat()}_{end.isoformat()}.parquet"


def fetch_em_fx_raw(start: date, end: date) -> Path:
    """Baixa o OHLC diario de todas as moedas do painel (uma chamada), com
    cache idempotente em disco. yfinance trata `end` como exclusivo, entao
    pedimos end+1 dia para incluir o dia final."""
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


def apply_sanity_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mascara (vira NaN em high/low) os dias com range implausivel -- ver
    MAX_LOG_RANGE. Devolve (df_limpo, df_dos_dias_removidos) para que a
    remocao seja auditavel em vez de silenciosa.

    Mascara em vez de apagar a linha: o resto do pipeline alinha as moedas por
    data, e sumir com a linha desalinharia o painel.
    """
    valido = (df["high"] > 0) & (df["low"] > 0)
    log_range = pd.Series(np.nan, index=df.index)
    log_range[valido] = np.log(df.loc[valido, "high"] / df.loc[valido, "low"])

    ruim = log_range > MAX_LOG_RANGE
    removidos = df[ruim].copy()
    removidos["log_range"] = log_range[ruim]

    limpo = df.copy()
    limpo.loc[ruim, ["high", "low"]] = np.nan
    return limpo, removidos


def _parse_raw_file(path: Path) -> pd.DataFrame:
    """Parse do parquet cru do yfinance (colunas MultiIndex campo x ticker)
    para formato LONGO: date, currency, open, high, low, close."""
    raw = pd.read_parquet(path)
    if raw.empty:
        return pd.DataFrame(columns=["date", "currency", "open", "high", "low", "close"])

    blocos = []
    for code, ticker in TICKERS.items():
        bloco = pd.DataFrame(
            {
                "date": raw.index,
                "currency": code,
                "open": raw["Open"][ticker].to_numpy(),
                "high": raw["High"][ticker].to_numpy(),
                "low": raw["Low"][ticker].to_numpy(),
                "close": raw["Close"][ticker].to_numpy(),
            }
        )
        blocos.append(bloco)

    df = pd.concat(blocos, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(TIMEZONE)
    return df.dropna(subset=["high", "low"], how="all")


def load_em_fx_processed(start: date, end: date) -> pd.DataFrame:
    """Garante o raw em cache, aplica o filtro de sanidade e salva o parquet
    processado. Colunas: date (tz-aware), currency, open, high, low, close."""
    df = _parse_raw_file(fetch_em_fx_raw(start, end))
    df, removidos = apply_sanity_filter(df)
    if not removidos.empty:
        resumo = removidos.groupby("currency").size().to_dict()
        print(f"filtro de sanidade mascarou {len(removidos)} dia(s): {resumo}")

    df = df.sort_values(["currency", "date"]).reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    return df


def load_em_fx() -> pd.DataFrame:
    """Le o painel ja coletado (sem rede). Levanta se ainda nao foi coletado."""
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} nao encontrado -- rode "
            "`python -m data.em_fx` para coletar o painel primeiro."
        )
    return pd.read_parquet(PROCESSED_PATH)


if __name__ == "__main__":
    out = load_em_fx_processed(date(2018, 1, 1), date.today())
    print(
        out.groupby("currency").agg(
            n=("date", "size"), inicio=("date", "min"), fim=("date", "max")
        )
    )