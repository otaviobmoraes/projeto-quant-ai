"""Negocios de OPCAO de dolar nos boletins diarios da B3 (BVBG-086) e a vol
implicita invertida deles via Black-76.

POR QUE EXISTE: a Fase 2 do CLAUDE.md previa reconstruir IV propria invertendo
Black-76 a partir dos precos de AJUSTE das opcoes. Esse caminho esta FECHADO --
verificado em 2019, 2022 e 2026: a B3 lista milhares de series de opcao de
dolar neste arquivo e publica ajuste (`AdjstdQt`) para ZERO delas, em todos os
periodos. Nao e mudanca recente, e pratica estrutural.

O caminho que RESTA, e que este modulo implementa: inverter Black-76 sobre o
preco EFETIVAMENTE NEGOCIADO. Nao e a superficie limpa da B3, mas e IV real --
contra a alternativa atual do backtest, que e `RV_trailing x premio constante`
calibrado num unico dia.

LIMITACOES, para nao superestimar o que isso entrega:

- ESPARSO e em queda. Series com negocio por pregao: ~45 em 2019, ~30 em 2022,
  ~3 em 2026. A cobertura e boa justamente nos anos antigos e pessima nos
  recentes -- ou seja, cobre bem o periodo que a extensao da amostra de futuros
  esta trazendo, e mal o periodo da amostra atual (2023-2026).
- RUIDOSO. Um unico negocio carrega bid-ask bounce; `TradAvrgPric` (preco medio
  do pregao naquela serie) e preferido a `FrstPric` justamente por mediar isso,
  mas com 1-5 negocios a media ainda e fragil.
- SEM CURVA DE JUROS. r=0 por padrao, mesma convencao de vol/black76.py.
- Negocios fundo dentro do dinheiro NAO produzem IV (premio vira intrinseco
  puro) -- `vol.black76.implied_vol` devolve None e a linha e descartada.

Convencao de escala: strike, premio e ajuste do futuro estao todos na mesma
unidade do arquivo da B3 (R$ por 1.000 USD), entao entram direto no Black-76.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from vol.black76 import implied_vol
from vol.roll import contract_expiry

TIMEZONE = "America/Sao_Paulo"

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_PATH = PROCESSED_DIR / "b3_option_trades.parquet"

# <PricRpt> de uma serie de OPCAO de dolar: DOL + mes + ano + C/P + strike.
_OPTION_BLOCK_RE = re.compile(
    rb"<PricRpt>(?:(?!</PricRpt>).)*?<TckrSymb>(DOL([A-Z]\d{2})([CP])(\d+))</TckrSymb>"
    rb"((?:(?!</PricRpt>).)*?)</PricRpt>",
    re.S,
)
# <PricRpt> de um FUTURO de dolar -- necessario como F do Black-76.
_FUTURE_BLOCK_RE = re.compile(
    rb"<PricRpt>(?:(?!</PricRpt>).)*?<TckrSymb>DOL([A-Z]\d{2})</TckrSymb>"
    rb"((?:(?!</PricRpt>).)*?)</PricRpt>",
    re.S,
)

COLUMNS = [
    "date", "ticker", "maturity", "option_type", "strike",
    "price", "trades", "open_interest", "future_settlement",
]


def _field(block: bytes, tag: str) -> float | None:
    match = re.search(rf"<{tag}[^>]*>([^<>]+)</{tag}>".encode(), block)
    return float(match.group(1)) if match else None


def _last_snapshot(zip_bytes: bytes) -> bytes:
    """Os 4 XMLs do pacote diario sao snapshots progressivos; o ultimo e o
    mais completo (mesma convencao de data/b3_futures.py)."""
    outer = zipfile.ZipFile(io.BytesIO(zip_bytes))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(outer.namelist()[0])))
    return inner.read(sorted(inner.namelist())[-1])


def parse_dol_options(zip_bytes: bytes, refdate: date) -> pd.DataFrame:
    """Parse PURO (sem I/O de rede) -> uma linha por serie de opcao de dolar
    que EFETIVAMENTE NEGOCIOU no pregao, com o ajuste do futuro de mesmo
    vencimento anexado (o F do Black-76).

    Series sem negocio sao descartadas: sem preco nao ha o que inverter, e
    elas sao a esmagadora maioria (2.159 de 2.162 num pregao de 2026).
    """
    payload = _last_snapshot(zip_bytes)

    futures: dict[str, float] = {}
    for match in _FUTURE_BLOCK_RE.finditer(payload):
        settlement = _field(match.group(2), "AdjstdQt")
        if settlement is not None:
            futures[match.group(1).decode()] = settlement

    records = []
    for match in _OPTION_BLOCK_RE.finditer(payload):
        ticker, maturity, kind, strike_raw = (g.decode() for g in match.group(1, 2, 3, 4))
        block = match.group(5)

        # preco medio do pregao naquela serie; FrstPric so como ultimo recurso
        price = _field(block, "TradAvrgPric") or _field(block, "FrstPric")
        if price is None:
            continue

        records.append(
            {
                "date": pd.Timestamp(refdate, tz=TIMEZONE),
                "ticker": ticker,
                "maturity": maturity,
                "option_type": kind,
                "strike": float(strike_raw),
                "price": price,
                "trades": _field(block, "TradQty"),
                "open_interest": _field(block, "OpnIntrst"),
                "future_settlement": futures.get(maturity),
            }
        )

    return pd.DataFrame(records, columns=COLUMNS)


def save_option_trades(trades: pd.DataFrame) -> Path:
    """Persiste os negocios em data/processed/, fazendo UPSERT por
    (date, ticker) -- a coleta e incremental e pode ser retomada, entao
    reescrever o arquivo do zero perderia o que ja foi baixado."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if PROCESSED_PATH.exists():
        trades = pd.concat([pd.read_parquet(PROCESSED_PATH), trades], ignore_index=True)
    trades.drop_duplicates(subset=["date", "ticker"], keep="last").sort_values(
        ["date", "ticker"]
    ).to_parquet(PROCESSED_PATH, index=False)
    return PROCESSED_PATH


def load_option_trades() -> pd.DataFrame:
    """Le os negocios de opcao ja coletados (sem I/O de rede)."""
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} nao encontrado -- rode a coleta de negocios de opcao primeiro."
        )
    return pd.read_parquet(PROCESSED_PATH)


def add_implied_vols(
    options: pd.DataFrame, r: float = 0.0, max_moneyness: float = 0.10
) -> pd.DataFrame:
    """Acrescenta `days_to_expiry`, `moneyness` e `iv_pct` (vol implicita em
    PONTOS PERCENTUAIS, convencao do resto do projeto) a cada negocio.

    `max_moneyness`: descarta negocios a mais de X% do dinheiro ANTES de
    tentar inverter. Nao e cosmetico -- fora dessa faixa o premio tende ao
    intrinseco puro, a inversao fica mal-posta e o pouco que voltasse seria
    ruido de arredondamento. A faixa observada nos negocios reais e ~+-2%.

    Linhas sem F (vencimento de futuro sem ajuste no dia) ou cuja inversao
    falha saem com `iv_pct` = NaN, para o chamador filtrar.
    """
    df = options.copy()
    if df.empty:
        return df.assign(days_to_expiry=[], moneyness=[], iv_pct=[])

    expiries = df["ticker"].str[:6].map(contract_expiry)
    df["days_to_expiry"] = (
        pd.to_datetime(expiries.to_numpy()) - pd.to_datetime(df["date"].dt.date.to_numpy())
    ).days
    df["moneyness"] = df["strike"] / df["future_settlement"] - 1.0

    def _iv(row) -> float | None:
        if pd.isna(row["future_settlement"]) or row["days_to_expiry"] <= 0:
            return None
        if abs(row["moneyness"]) > max_moneyness:
            return None
        sigma = implied_vol(
            price=row["price"],
            F=row["future_settlement"],
            K=row["strike"],
            T=row["days_to_expiry"] / 365.0,
            option_type=row["option_type"],
            r=r,
        )
        return sigma * 100 if sigma is not None else None

    df["iv_pct"] = df.apply(_iv, axis=1).astype(float)
    return df


def atm_iv_by_date(
    options_with_iv: pd.DataFrame, max_moneyness: float = 0.02, min_trades: float = 1.0
) -> pd.DataFrame:
    """Agrega os negocios em uma IV ATM por (data, vencimento): media das vols
    implicitas dos negocios dentro de `max_moneyness`, PONDERADA pelo numero
    de negocios -- uma serie com 5 negocios e menos ruidosa que uma com 1.

    Devolve tambem `n_trades` e `n_series`, que o consumidor deve usar como
    medida de confianca: uma IV vinda de 1 negocio unico nao merece o mesmo
    peso de uma vinda de 8.
    """
    df = options_with_iv.dropna(subset=["iv_pct"])
    df = df[(df["moneyness"].abs() <= max_moneyness) & (df["trades"] >= min_trades)]
    if df.empty:
        return pd.DataFrame(columns=["date", "maturity", "iv_pct", "n_trades", "n_series"])

    def _agg(g: pd.DataFrame) -> pd.Series:
        w = g["trades"].to_numpy()
        return pd.Series(
            {
                "iv_pct": float((g["iv_pct"] * w).sum() / w.sum()),
                "n_trades": float(w.sum()),
                "n_series": float(len(g)),
            }
        )

    return (
        df.groupby(["date", "maturity"], as_index=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
    )
