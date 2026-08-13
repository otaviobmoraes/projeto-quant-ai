"""Serie historica de IV construida a partir de NEGOCIOS de opcao de dolar, e
avaliacao dela como previsor da RV futura.

Complementa data/b3_options.py (que parseia e inverte negocio a negocio) com a
camada de agregacao e avaliacao: dos negocios brutos ate "IV ATM do dia" e ate
o veredito "a IV do mercado preve melhor que os modelos do projeto?".

POR QUE ISSO EXISTE: sem historico de IV, o backtest de P&L do projeto usa
`IV_proxy = RV_trailing x premio constante` -- o que faz os dois lados do
spread (RV prevista vs IV) virem da mesma serie, com correlacao ~0,994, e
esvazia o sinal. Esta e a primeira IV historica REAL do projeto.

FILTROS DE QUALIDADE (padroes abaixo), todos com motivo:

- `min_trades=2`: uma serie com um unico negocio carrega bid-ask bounce cheio.
- `iv_range=(4, 60)`: fora disso e negocio fora de mercado, nao vol de USD/BRL.
- `dte_range=(15, 60)`: o alvo do projeto e RV de 21 dias UTEIS (~30 corridos);
  vencimentos muito curtos ou muito longos medem outro horizonte.
- `max_moneyness=0.03`: longe do dinheiro o premio tende ao intrinseco e a
  inversao perde condicionamento (ver vol.black76.implied_vol).

Os filtros foram fixados ANTES de olhar o resultado da avaliacao. Alterar
qualquer um deles conta como configuracao nova e precisa entrar no
CONFIGS_TESTED de report/run_report.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.metrics import pooled_oos_metrics
from data.b3_options import add_implied_vols, atm_iv_by_date
from vol.forecast import forward_target_from_variance
from vol.realized import TRADING_DAYS_PER_YEAR

MIN_TRADES = 2.0
IV_RANGE = (4.0, 60.0)
DTE_RANGE = (15, 60)
MAX_MONEYNESS = 0.03


def filter_quality(
    trades_with_iv: pd.DataFrame,
    min_trades: float = MIN_TRADES,
    iv_range: tuple[float, float] = IV_RANGE,
    dte_range: tuple[int, int] = DTE_RANGE,
) -> pd.DataFrame:
    """Aplica os filtros de qualidade documentados no modulo. Entrada: saida de
    data.b3_options.add_implied_vols."""
    df = trades_with_iv.dropna(subset=["iv_pct"])
    return df[
        (df["trades"] >= min_trades)
        & (df["iv_pct"].between(*iv_range))
        & (df["days_to_expiry"].between(*dte_range))
    ]


def daily_atm_iv(trades: pd.DataFrame, max_moneyness: float = MAX_MONEYNESS) -> pd.Series:
    """Dos negocios brutos ate uma IV ATM por DIA (media entre vencimentos
    elegiveis, cada um ja ponderado por numero de negocios em atm_iv_by_date).

    Indice normalizado para data sem fuso -- a serie e diaria e vai ser pareada
    com a variancia do futuro, que vem tz-aware; comparar as duas com fusos
    diferentes e uma fonte silenciosa de desalinhamento.
    """
    com_iv = filter_quality(add_implied_vols(trades))
    atm = atm_iv_by_date(com_iv, max_moneyness=max_moneyness)
    if atm.empty:
        return pd.Series(dtype=float, name="iv_pct")
    serie = atm.groupby("date")["iv_pct"].mean()
    serie.index = pd.DatetimeIndex([pd.Timestamp(d).date() for d in serie.index])
    return serie.rename("iv_pct")


def pair_with_forward_rv(
    iv: pd.Series, daily_variance: pd.Series, horizon: int = 21
) -> pd.DataFrame:
    """Pareia a IV de cada dia com a RV EFETIVAMENTE REALIZADA nos `horizon`
    dias seguintes, e com a persistencia (RV dos 21 dias anteriores) como
    benchmark de zero parametros.

    Colunas: iv, target (RV futura), persist.
    """
    idx = pd.DatetimeIndex([pd.Timestamp(d).date() for d in daily_variance.index])
    fwd = pd.Series(
        forward_target_from_variance(daily_variance, horizon).to_numpy(), index=idx
    )
    trail = pd.Series(
        (np.sqrt(daily_variance.rolling(horizon).mean() * TRADING_DAYS_PER_YEAR) * 100).to_numpy(),
        index=idx,
    )
    return (
        pd.DataFrame({"iv": iv})
        .join(pd.DataFrame({"target": fwd, "persist": trail}), how="inner")
        .dropna()
    )


def independent_windows(paired: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Seleciona observacoes cujas janelas de `horizon` dias NAO se sobrepoem,
    espacando por DATA (guloso, do mais antigo para o mais novo).

    Por que nao `iloc[::horizon]`: aquilo espaca por POSICAO NA TABELA, o que
    so equivale a espacar por data se as linhas forem dias consecutivos. A
    serie de IV deste projeto mistura um bloco denso (dias seguidos de 2018)
    com dias ja coletados de 21 em 21 -- no trecho esparso, pular 21 linhas
    pularia 21 * 21 dias e descartaria quase tudo, por um criterio errado.

    Usa dias CORRIDOS equivalentes (horizon dias uteis ~ horizon * 7/5) para
    nao depender de calendario de feriados.
    """
    if paired.empty:
        return paired
    gap = pd.Timedelta(days=int(round(horizon * 7 / 5)))
    escolhidos: list = []
    ultimo: pd.Timestamp | None = None
    for data in paired.index:
        if ultimo is None or data - ultimo >= gap:
            escolhidos.append(data)
            ultimo = data
    return paired.loc[escolhidos]


def evaluate_forecasts(paired: pd.DataFrame, horizon: int = 21) -> dict:
    """Compara a IV do mercado contra a persistencia como previsor da RV
    futura, nas DUAS versoes: todos os dias (sobrepostos) e janelas
    INDEPENDENTES (ver `independent_windows`).

    A versao independente e a que vale estatisticamente -- janelas sobrepostas
    ja produziram p-valor inflado neste projeto e o registro dessa armadilha
    esta no CONFIGS_TESTED. As duas sao devolvidas de proposito, para o
    relatorio poder mostrar a diferenca entre elas.
    """
    out: dict = {}
    for rotulo, sub in (
        ("sobreposto", paired),
        ("independente", independent_windows(paired, horizon)),
    ):
        out[rotulo] = {
            "n": int(len(sub)),
            "iv": pooled_oos_metrics(sub["target"], sub["iv"]),
            "persistencia": pooled_oos_metrics(sub["target"], sub["persist"]),
            "corr_iv_rv": float(sub["iv"].corr(sub["target"])) if len(sub) > 2 else float("nan"),
        }
    return out


def load_and_evaluate(horizon: int = 21, source: str = "b3") -> dict:
    """Caminho completo a partir dos dados ja coletados: negocios de opcao ->
    IV ATM diaria -> pareamento com a RV futura -> veredito.

    Existe para que nenhum numero desta linha de investigacao precise ser
    digitado a mao no relatorio -- mesma regra que ja vale para
    relatorio/gerar_dados.py.
    """
    from data.b3_options import load_option_trades
    from vol.realized import load_prices_and_variance

    trades = load_option_trades()
    iv = daily_atm_iv(trades)
    _, variance = load_prices_and_variance(source=source)
    paired = pair_with_forward_rv(iv, variance, horizon=horizon)

    return {
        "n_negocios": int(len(trades)),
        "n_pregoes_com_iv": int(len(iv)),
        "periodo": (
            (str(paired.index.min().date()), str(paired.index.max().date()))
            if not paired.empty
            else None
        ),
        "avaliacao": evaluate_forecasts(paired, horizon=horizon),
        "premio": variance_risk_premium(paired) if not paired.empty else None,
    }


def variance_risk_premium(paired: pd.DataFrame) -> dict:
    """Premio de risco de variancia medido: razao entre IV e RV.

    `iv_over_forward` e o premio ECONOMICO (o que a opcao cobrou contra o que
    ela entregou). `iv_over_trailing` e o que o backtest ilustrativo do projeto
    assume constante (calibrado num unico dia) -- devolvido para comparacao
    direta com aquela calibragem.
    """
    return {
        "n": int(len(paired)),
        "iv_medio": float(paired["iv"].mean()),
        "rv_futura_media": float(paired["target"].mean()),
        "iv_over_forward": float((paired["iv"] / paired["target"]).mean()),
        "iv_over_trailing": float((paired["iv"] / paired["persist"]).mean()),
        "share_iv_acima": float((paired["iv"] > paired["target"]).mean()),
    }
