"""Spread bid-ask EFETIVO das opcoes de dolar, MEDIDO dos negocios da B3.

POR QUE EXISTE. O custo de transacao do backtest e `spread_pct=0.05` -- 5% do
premio, descrito em backtest/costs.py como "simplificacao explicita e
deliberadamente conservadora". Nunca foi medido. E o parametro que hoje DECIDE
o resultado: o Sharpe bruto da estrategia fica em +0.137/+0.262 e o liquido em
-0.521/-0.374, uma queda de -0.667 atribuida inteiramente a esse numero
suposto. Trocar a suposicao por medicao e a acao de maior retorno que resta.

O DADO QUE PERMITE MEDIR. O bloco <PricRpt> de cada serie no BVBG-086 publica
`MinPric` e `MaxPric` -- o menor e o maior preco NEGOCIADO na serie naquele
pregao -- alem de `TradQty` (numero de negocios) e `TradAvrgPric`. Numa serie
com poucos negocios, a amplitude entre eles e dominada pelo salto entre a
ponta de compra e a de venda (bid-ask bounce), que e exatamente o que se quer
medir.

POUCOS NEGOCIOS: CONDICIONAR, NAO CORRIGIR. Com N negocios caindo na ponta de
compra ou de venda, a amplitude observada e ZERO se todos cairem do mesmo lado
e o SPREAD INTEIRO caso contrario -- nao ha valor intermediario no modelo de
salto entre duas pontas. A tentacao e usar todas as series e dividir pelo
fator E[amplitude]/spread = 1 - 2^(1-N); mas como as series de amplitude zero
sao descartadas de qualquer forma (amplitude nula e "nao medido", nao "spread
zero"), aplicar TAMBEM o fator contaria o mesmo efeito duas vezes e inflaria o
spread. Verificado por simulacao em tests/test_option_spread.py: com N=2 e
spread verdadeiro de 5%, o estimador com fator devolvia 10%.
O tratamento correto e CONDICIONAL: dado que a amplitude foi observada > 0,
ela E o spread. O filtro seleciona series por numero de negocios, nao pelo
tamanho do spread, entao nao enviesa a grandeza medida -- so restringe a
populacao as series que de fato negociaram dos dois lados, que sao exatamente
as operaveis.

O PRECO DO ATIVO SE MOVE DURANTE O PREGAO, e isso INFLA a amplitude. Parte do
intervalo entre o menor e o maior negocio nao e spread, e o premio mudando
porque o futuro andou. Descontamos essa parcela por
|delta| x amplitude_do_futuro, a variacao esperada do premio dado o movimento
observado no subjacente. E aproximacao de primeira ordem (ignora gama),
conservadora no sentido de descontar de menos e portanto SUPERestimar o
spread residual.

CONVENCAO DE SAIDA: `spread_pct` na mesma unidade que backtest/costs.py
espera -- fracao do premio cobrada por transacao, isto e META-amplitude
(metade do spread total) dividida pelo preco medio. E o custo de atravessar o
book uma vez.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Numero minimo de negocios na serie/dia para a amplitude ser informativa.
# Com 1 negocio, MinPric == MaxPric por construcao e a observacao nao carrega
# informacao nenhuma sobre o spread (nao e "spread zero", e "nao medido").
MIN_TRADES = 2

# Teto de sanidade: amplitude acima disso e negocio fora de mercado, nao
# spread. Declarado antes de olhar o resultado.
MAX_RANGE_FRAC = 1.0

_OPTION_BLOCK_RE = re.compile(
    rb"<PricRpt>(?:(?!</PricRpt>).)*?<TckrSymb>(DOL([A-Z]\d{2})([CP])(\d+))</TckrSymb>"
    rb"((?:(?!</PricRpt>).)*?)</PricRpt>",
    re.S,
)


def _field(block: bytes, tag: str) -> float | None:
    m = re.search(rf"<{tag}[^>]*>([^<>]+)</{tag}>".encode(), block)
    return float(m.group(1)) if m else None


def parse_option_price_ranges(zip_bytes: bytes, refdate) -> pd.DataFrame:
    """Uma linha por serie de opcao de dolar NEGOCIADA, com min/max/medio e o
    numero de negocios -- os campos que `data.b3_options.parse_dol_options`
    nao captura porque nao precisava deles para inverter a IV.
    """
    from data.b3_options import _last_snapshot

    payload = _last_snapshot(zip_bytes)
    registros = []
    for m in _OPTION_BLOCK_RE.finditer(payload):
        ticker, maturity, kind, strike = (g.decode() for g in m.group(1, 2, 3, 4))
        b = m.group(5)
        medio = _field(b, "TradAvrgPric")
        if medio is None or medio <= 0:
            continue
        registros.append(
            {
                "date": pd.Timestamp(refdate),
                "ticker": ticker,
                "maturity": maturity,
                "option_type": kind,
                "strike": float(strike),
                "price_min": _field(b, "MinPric"),
                "price_max": _field(b, "MaxPric"),
                "price_avg": medio,
                "n_trades": _field(b, "TradQty") or 0.0,
            }
        )
    return pd.DataFrame(registros)


def reveal_probability(n_trades: pd.Series | np.ndarray) -> np.ndarray:
    """P(a serie revela o spread) = 1 - 2^(1-N): a chance de os N negocios NAO
    cairem todos da mesma ponta.

    NAO entra na estimacao (ver docstring do modulo -- usar isso junto com o
    filtro de amplitude > 0 contaria o efeito duas vezes). Serve para
    diagnostico: diz que fracao das series com N negocios se espera perder, e
    portanto o quanto a amostra medida representa o universo negociado.
    N=2 -> 50%;  N=4 -> 88%;  N=10 -> 99,8%.
    """
    n = np.asarray(n_trades, dtype=float)
    return 1.0 - np.power(2.0, 1.0 - np.maximum(n, 2.0))


def effective_spread(
    ranges: pd.DataFrame,
    underlying_range: pd.Series | None = None,
    deltas: pd.Series | None = None,
    min_trades: int = MIN_TRADES,
) -> pd.DataFrame:
    """Spread efetivo por serie/dia, em FRACAO DO PREMIO por transacao.

    `underlying_range` (opcional): amplitude ABSOLUTA do futuro naquele dia
    (high - low), indexada por data. `deltas` (opcional): |delta| da serie.
    Se os dois vierem, aplica-se a correcao 2 (desconto do movimento do
    subjacente). Sem eles, o resultado e um LIMITE SUPERIOR do spread.
    """
    df = ranges[ranges["n_trades"] >= min_trades].copy()
    df = df[df["price_min"].notna() & df["price_max"].notna() & (df["price_avg"] > 0)]
    # amplitude nula = os negocios cairam todos da mesma ponta => NAO MEDIDO.
    # Tratar como "spread zero" puxaria a mediana para baixo artificialmente.
    df = df[df["price_max"] > df["price_min"]]
    if df.empty:
        return df.assign(spread_pct=pd.Series(dtype=float))

    amplitude = df["price_max"] - df["price_min"]

    if underlying_range is not None and deltas is not None:
        mov = deltas.reindex(df.index).abs() * underlying_range.reindex(df["date"]).to_numpy()
        amplitude = (amplitude - pd.Series(mov, index=df.index).fillna(0.0)).clip(lower=0.0)

    # Condicionado a ter sido observada, a amplitude E o spread total. Converte
    # para "atravessar o book uma vez" (metade) como fracao do premio.
    df["spread_pct"] = (amplitude / 2.0) / df["price_avg"]

    df = df[(df["spread_pct"] > 0) & (df["spread_pct"] <= MAX_RANGE_FRAC)]
    return df


def spread_by_regression(
    ranges: pd.DataFrame,
    underlying_range: pd.Series,
    deltas: pd.Series,
    min_trades: int = MIN_TRADES,
) -> dict:
    """Separa spread de movimento do subjacente por REGRESSAO, em vez de
    subtrair uma estimativa do movimento serie a serie.

    POR QUE A SUBTRACAO DIRETA NAO FUNCIONA: a amplitude diaria do futuro e um
    limite SUPERIOR do quanto o preco andou ENTRE os dois negocios extremos da
    opcao -- sem carimbo de hora nao da para saber se eles ocorreram com 10
    minutos ou 6 horas de diferenca. Subtrair a amplitude inteira zera quase
    toda a amostra (medido: sobram 2 de 138 series), o que nao e "spread zero",
    e o desconto ser grande demais.

    O MODELO, sobre a amostra inteira em vez de serie a serie:

        amplitude_opcao / preco  =  a  +  b * (|delta| * amplitude_futuro / preco)

    O intercepto `a` e a parte da amplitude que NAO e explicada por movimento
    do subjacente -- isto e, o spread bid-ask total, em fracao do premio. O
    coeficiente `b` estima que fracao da amplitude diaria do futuro e de fato
    percorrida entre os negocios extremos, e serve de checagem de sanidade:
    tem de ficar entre 0 e 1.

    Devolve o spread POR TRANSACAO (metade do total), na convencao de
    backtest/costs.py.
    """
    import statsmodels.api as sm

    df = ranges[ranges["n_trades"] >= min_trades].copy()
    df = df[df["price_min"].notna() & df["price_max"].notna() & (df["price_avg"] > 0)]
    df = df[df["price_max"] > df["price_min"]]
    if len(df) < 20:
        return {"n": int(len(df))}

    faixa_fut = underlying_range.reindex(df["date"]).to_numpy()
    delta = deltas.reindex(df.index).abs().to_numpy()
    y = ((df["price_max"] - df["price_min"]) / df["price_avg"]).to_numpy()
    x = delta * faixa_fut / df["price_avg"].to_numpy()

    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) < 20:
        return {"n": int(len(y))}

    modelo = sm.OLS(y, sm.add_constant(x)).fit()
    a, b = float(modelo.params[0]), float(modelo.params[1])
    return {
        "n": int(len(y)),
        "spread_total_pct": a,
        "spread_pct": a / 2.0,
        "ic95_spread_pct": tuple(v / 2.0 for v in modelo.conf_int()[0]),
        "beta_movimento": b,
        "p_intercepto": float(modelo.pvalues[0]),
        "r2": float(modelo.rsquared),
    }


def summarize_spread(spreads: pd.DataFrame) -> dict:
    """Resumo robusto. A MEDIANA e a estatistica principal, nao a media: a
    distribuicao de spread relativo tem cauda direita longa (opcao barata,
    fundo fora do dinheiro, com um tick de amplitude, gera percentual enorme),
    e a media seria dominada por essas series que ninguem operaria.
    """
    if spreads.empty:
        return {"n": 0}
    s = spreads["spread_pct"]
    return {
        "n": int(len(s)),
        "n_pregoes": int(spreads["date"].nunique()),
        "mediana": float(s.median()),
        "media": float(s.mean()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "media_negocios": float(spreads["n_trades"].mean()),
    }