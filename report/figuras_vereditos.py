"""Figuras dos vereditos da fase de delta-hedge e horizonte curto.

POR QUE EXISTE. As figuras em report/output/ eram de 2026-08-06 e nao conheciam
delta-hedge, alvo em razao, painel, bootstrap, diagnostico de limite nem spread
medido. Uma delas -- `backtest_pnl.png` -- era ATIVAMENTE ENGANOSA: mostrava a
curva do backtest SEM hedge, cujo P&L mede direcao mais ruido, e cujos Sharpes
positivos este projeto depois estabeleceu serem sorte direcional.
`pnl_delta_hedge.png` e a substituta honesta.

DESENHO DO MODULO: as funcoes de figura sao PURAS -- recebem os dados prontos e
devolvem a Figure, sem tocar disco nem recalcular nada. Quem calcula e
`computar_dados()`. Isso e o que torna as figuras testaveis com dado sintetico,
sem rodar backtest de minutos dentro da suite.

IDENTIDADE VISUAL: dourado e grafite do relatorio, fundo TRANSPARENTE e sem
grade -- as figuras sao coladas dentro do card branco do documento. Ver o bloco
de constantes abaixo para o porque de cada escolha.

Uso: `python -m report.figuras_vereditos`
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"

# ---------------------------------------------------------------------------
# Identidade visual do RELATORIO (bloco grafite + dourado sobre card branco).
#
# FUNDO TRANSPARENTE, sem grade: as figuras sao coladas dentro do card branco
# do documento, entao pintar um fundo proprio criaria um retangulo visivel de
# tom levemente diferente, e a grade competiria com a diagramacao da pagina.
# A referencia de leitura fica na linha do zero e nos rotulos diretos, que ja
# estao em toda barra.
#
# O par dourado+azul foi validado no script da skill de dataviz (protanopia,
# deuteranopia, tritanopia e contraste sobre superficie clara). O dourado
# sozinho nao serve para series multiplas -- duas tonalidades do mesmo ouro
# reprovam na separacao para daltonicos --, entao o azul entra como segunda
# serie e o dourado fica reservado ao que a figura quer destacar.
# ---------------------------------------------------------------------------
OURO = "#c2a14d"          # destaque, mesmo dourado das faixas do relatorio
AZUL = "#2d6a9f"          # segunda serie
GRAFITE = "#2b2b2d"       # titulos, mesma tinta do bloco de capitulo
CINZA_TEXTO = "#5a5a5c"   # subtitulos e rotulos de eixo
CINZA_FRACO = "#9a9a9c"   # ticks
LINHA_ZERO = "#c9c9cb"    # unica referencia horizontal que sobra

HORIZON_PADRAO, EMBARGO = 21, 5


def _eixos(figsize=(9, 4.8)) -> tuple[Figure, plt.Axes]:
    """Eixos no padrao do relatorio: sem fundo, sem grade, sem molduras."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.grid(False)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(LINHA_ZERO)
    ax.tick_params(colors=CINZA_FRACO, labelsize=9, length=0)
    return fig, ax


def _titulo(ax, titulo: str, subtitulo: str = "") -> None:
    """Titulo e subtitulo como textos empilhados ACIMA dos eixos.

    Nao usa `ax.set_title` porque combinar set_title com um texto de subtitulo
    em coordenadas de eixo faz os dois colidirem -- os offsets sao medidos em
    unidades diferentes (pontos vs fracao do eixo) e nao se coordenam.
    """
    if subtitulo:
        ax.text(0, 1.13, titulo, transform=ax.transAxes, fontsize=12.5,
                color=GRAFITE, va="bottom", ha="left")
        ax.text(0, 1.045, subtitulo, transform=ax.transAxes, fontsize=9.2,
                color=CINZA_FRACO, va="bottom", ha="left")
    else:
        ax.text(0, 1.045, titulo, transform=ax.transAxes, fontsize=12.5,
                color=GRAFITE, va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Figuras (puras)
# ---------------------------------------------------------------------------


def fig_sharpe_por_horizonte(dados: pd.DataFrame) -> Figure:
    """Sharpe por horizonte, sem custo e com o spread medido, com IC95.

    `dados`: colunas h, sr_sem, lo_sem, hi_sem, sr_com, lo_com, hi_com.
    """
    fig, ax = _eixos((9.5, 5.0))
    x = np.arange(len(dados))
    larg = 0.36

    for desloc, suf, cor, rot in (
        (-larg / 2, "sem", AZUL, "Sem custo de execução"),
        (+larg / 2, "com", OURO, "Com spread medido"),
    ):
        vals = dados[f"sr_{suf}"].to_numpy()
        lo, hi = dados[f"lo_{suf}"].to_numpy(), dados[f"hi_{suf}"].to_numpy()
        ax.bar(x + desloc, vals, larg, color=cor, label=rot, zorder=3)
        ax.errorbar(x + desloc, vals, yerr=[vals - lo, hi - vals], fmt="none",
                    ecolor=cor, elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)
        for xi, v, h in zip(x + desloc, vals, hi):
            ax.annotate(f"{v:+.2f}".replace(".", ","), (xi, h), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=9, color=GRAFITE,
                        fontweight="bold", zorder=5)

    ax.axhline(0, color=LINHA_ZERO, linewidth=1.3, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h = {int(h)} dias úteis" for h in dados["h"]], fontsize=10,
                       color=CINZA_TEXTO)
    ax.set_ylabel("Sharpe anualizado", color=CINZA_TEXTO, fontsize=10)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=CINZA_TEXTO, loc="upper right")
    _titulo(ax, "O Sharpe é monotônico no horizonte",
            "Straddle ATM delta-neutro · IC95 por bootstrap de bloco (Ledoit & Wolf, 2008)")
    fig.tight_layout()
    return fig


def fig_capacidade(dados: pd.DataFrame) -> Figure:
    """R2 dentro vs fora da amostra conforme a capacidade do modelo cresce.

    `dados`: colunas capacidade, r2_dentro, r2_fora.
    """
    fig, ax = _eixos((9.5, 4.8))
    x = np.arange(len(dados))
    for col, cor, rot in (("r2_dentro", OURO, "R² dentro da amostra"),
                          ("r2_fora", AZUL, "R² fora da amostra")):
        ax.plot(x, dados[col], color=cor, linewidth=2.0, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.6, label=rot, zorder=3)
        ax.annotate(f"{dados[col].iloc[-1]:+.3f}".replace(".", ","),
                    (x[-1], dados[col].iloc[-1]), textcoords="offset points",
                    xytext=(10, -3), fontsize=9.5, color=cor, fontweight="bold")

    ax.axhline(0, color=LINHA_ZERO, linewidth=1.3, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(dados["capacidade"], fontsize=9.5, color=CINZA_TEXTO)
    ax.set_ylabel("R²", color=CINZA_TEXTO, fontsize=10)
    ax.set_xlim(-0.3, len(dados) - 0.5)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=CINZA_TEXTO, loc="center left")
    _titulo(ax, "Mais capacidade memoriza o treino e piora a previsão",
            "h=21 · XGBoost com regularização desligada · bloco de teste fixo em 400 pregões")
    fig.tight_layout()
    return fig


def fig_spread_por_prazo(dados: pd.DataFrame) -> Figure:
    """Spread efetivo mediano por faixa de prazo ate o vencimento.

    `dados`: colunas faixa, mediana (em fracao), n.
    """
    fig, ax = _eixos((9.0, 4.6))
    x = np.arange(len(dados))
    pct = dados["mediana"].to_numpy() * 100
    ax.bar(x, pct, 0.6, color=OURO, zorder=3)
    for xi, v, n in zip(x, pct, dados["n"]):
        ax.annotate(f"{v:.2f}%".replace(".", ","), (xi, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=9.5, color=GRAFITE,
                    fontweight="bold")
        ax.annotate(f"n={int(n)}", (xi, 0), textcoords="offset points", xytext=(0, -22),
                    ha="center", fontsize=8.5, color=CINZA_FRACO, annotation_clip=False)

    ax.set_xticks(x)
    ax.set_xticklabels(dados["faixa"], fontsize=10, color=CINZA_TEXTO)
    ax.set_ylabel("Spread por transação (% do prêmio)", color=CINZA_TEXTO, fontsize=10)
    ax.set_ylim(0, max(pct) * 1.22)
    _titulo(ax, "O spread relativo explode em prazo curto",
            "Séries ATM (|moneyness| ≤ 3%) com ≥2 negócios · medido de MinPric/MaxPric da B3")
    fig.tight_layout()
    return fig


def fig_hedge_antes_depois(dados: pd.DataFrame) -> Figure:
    """Sharpe por banda morta, antes e depois do delta-hedge.

    `dados`: colunas banda, sem_hedge, com_hedge.
    """
    fig, ax = _eixos((9.0, 4.8))
    x = np.arange(len(dados))
    larg = 0.36
    for desloc, col, cor, rot in ((-larg / 2, "sem_hedge", OURO, "Sem hedge"),
                                  (+larg / 2, "com_hedge", AZUL, "Com delta-hedge")):
        vals = dados[col].to_numpy()
        ax.bar(x + desloc, vals, larg, color=cor, label=rot, zorder=3)
        for xi, v in zip(x + desloc, vals):
            ax.annotate(f"{v:+.2f}".replace(".", ","), (xi, v), textcoords="offset points",
                        xytext=(0, 7 if v >= 0 else -16), ha="center", fontsize=9,
                        color=GRAFITE, fontweight="bold")

    ax.axhline(0, color=LINHA_ZERO, linewidth=1.3, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"banda {b}".replace(".", ",") for b in dados["banda"]],
                       fontsize=10, color=CINZA_TEXTO)
    ax.set_ylabel("Sharpe anualizado", color=CINZA_TEXTO, fontsize=10)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=CINZA_TEXTO, loc="lower right")
    _titulo(ax, "Sem hedge o Sharpe troca de sinal; com hedge, não",
            "HAR em nível · IV real · trocar de sinal entre bandas "
            "é assinatura de resultado frágil")
    fig.tight_layout()
    return fig


def fig_rv_historia(rv: pd.Series) -> Figure:
    """Historia da volatilidade realizada do futuro de dolar.

    E a figura de contexto do relatorio: mostra a amplitude de regimes que a
    amostra cobre (2018-2026) e, por consequencia, por que metrica agregada
    engana neste dataset -- a mistura de regimes infla o denominador de
    qualquer R2 calculado no pool.
    """
    fig, ax = _eixos((9.5, 4.4))
    ax.plot(rv.index, rv.to_numpy(), color=AZUL, linewidth=1.3, zorder=3)
    ax.fill_between(rv.index, 0, rv.to_numpy(), color=AZUL, alpha=0.10, zorder=2)

    media = float(rv.mean())
    ax.axhline(media, color=OURO, linewidth=1.6, linestyle="--", zorder=4,
               label=f"média {media:.1f}%".replace(".", ","))

    i_max = rv.idxmax()
    # o replace so pode alcancar o NUMERO: aplicado a frase inteira, ele
    # transformaria "max." em "max," (bug pego por teste)
    ax.annotate("máx. " + f"{rv.max():.1f}%".replace(".", ","), (i_max, rv.max()),
                textcoords="offset points", xytext=(8, -4), fontsize=9,
                color=GRAFITE, fontweight="bold")

    ax.set_ylabel("Vol. realizada anualizada (%)", color=CINZA_TEXTO, fontsize=10)
    ax.set_ylim(0, rv.max() * 1.12)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=CINZA_TEXTO, loc="upper right")
    _titulo(ax, "Volatilidade realizada do futuro de dólar",
            f"Estimador de Parkinson · janela de 21 pregões · {len(rv)} observações")
    fig.tight_layout()
    return fig


def fig_pnl_e_drawdown(trades: pd.DataFrame, titulo: str) -> Figure:
    """Curva de P&L acumulado com o drawdown logo abaixo, eixo x compartilhado.

    EM UNIDADES ABSOLUTAS, e nao em %, e a razao e substantiva: a estrategia
    opera majoritariamente VENDIDA e a perda de uma venda de straddle nao tem
    teto, entao nao ha base de capital derivavel do premio. O capital exigido e
    a MARGEM, que depende de regras da camara e nao esta nos dados. Duas
    tentativas anteriores de normalizar (premio de cada trade; capital
    constante) produziram retorno abaixo de -100% e drawdown de -193%.
    Declarar uma base arbitraria fabricaria o numero mais visivel do grafico.
    """
    ordenado = trades.sort_values("exit_date")
    acum = ordenado["pnl_net"].cumsum().to_numpy()
    pico = np.maximum.accumulate(acum)
    dd = acum - pico

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.12},
    )
    fig.patch.set_alpha(0.0)
    for ax in (ax1, ax2):
        ax.set_facecolor("none")
        ax.grid(False)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(LINHA_ZERO)
        ax.tick_params(colors=CINZA_FRACO, labelsize=9)

    datas = ordenado["exit_date"]
    ax1.plot(datas, acum, color=AZUL, linewidth=1.8, zorder=3)
    ax1.fill_between(datas, 0, acum, color=AZUL, alpha=0.10, zorder=2)
    ax1.axhline(0, color=LINHA_ZERO, linewidth=1.2, zorder=1)
    ax1.set_ylabel("P&L acumulado (unidades do modelo)", color=CINZA_TEXTO, fontsize=9.5)
    _titulo(ax1, titulo,
            "Unidades absolutas — retorno sobre capital não é calculável "
            "(a margem não está nos dados)")

    ax2.fill_between(datas, 0, dd, color=OURO, alpha=0.30, zorder=2)
    ax2.plot(datas, dd, color=OURO, linewidth=1.4, zorder=3)
    ax2.set_ylabel("Drawdown", color=CINZA_TEXTO, fontsize=9.5)
    i_vale = int(np.argmin(dd))
    ax2.annotate(f"máx. {dd[i_vale]:,.0f}".replace(",", "."),
                 (datas.iloc[i_vale], dd[i_vale]), textcoords="offset points",
                 xytext=(8, 6), fontsize=9, color=OURO, fontweight="bold")

    # subplots_adjust em vez de tight_layout: com sharex + gridspec_kw o
    # tight_layout avisa que pode errar o resultado, e aqui as margens sao
    # conhecidas.
    fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.09, hspace=0.12)
    return fig


def fig_distribuicao_pnl(trades: pd.DataFrame, titulo: str) -> Figure:
    """Histograma do P&L por operacao.

    Mostra o que o Sharpe sozinho esconde: a assimetria. Com poucas operacoes e
    cauda esquerda longa, o resultado medio e refem de um punhado de trades.
    """
    x = trades["pnl_net"].to_numpy()
    fig, ax = _eixos((9.0, 4.4))
    n, bins, _ = ax.hist(x, bins=22, color=LINHA_ZERO, zorder=2)
    for conta, esq, dir_ in zip(n, bins[:-1], bins[1:]):
        if conta:
            ax.bar(esq, conta, width=dir_ - esq, align="edge",
                   color=AZUL if esq >= 0 else OURO, zorder=3)
    ax.axvline(0, color=LINHA_ZERO, linewidth=1.3, zorder=4)
    ax.axvline(float(x.mean()), color=GRAFITE, linewidth=1.4,
               linestyle="--", zorder=5, label=f"média {x.mean():,.0f}".replace(",", "."))
    ax.set_xlabel("P&L por operação (unidades do modelo)", color=CINZA_TEXTO, fontsize=10)
    ax.set_ylabel("operações", color=CINZA_TEXTO, fontsize=10)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=CINZA_TEXTO)
    assimetria = float(pd.Series(x).skew())
    _titulo(ax, titulo,
            f"{len(x)} operações · assimetria {assimetria:+.2f}".replace(".", ",") +
            " — poucas perdas grandes dominam o resultado")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cálculo (I/O + backtests)
# ---------------------------------------------------------------------------


def _previsao_livre_de_escala(v: pd.Series, h: int):
    from backtest.walk_forward import purged_walk_forward_splits_by_step
    from vol.forecast import (
        build_scale_free_dataset,
        fit_har_scale_free,
        forward_target_from_variance,
        har_features_from_variance,
        predict_scale_free,
    )

    ds = har_features_from_variance(v)
    ds["target"] = forward_target_from_variance(v, h)
    sf = build_scale_free_dataset(ds.dropna())
    folds = purged_walk_forward_splits_by_step(sf, 252, h, h, EMBARGO)
    preds = [predict_scale_free(fit_har_scale_free(tr), te) for tr, te in folds]
    return pd.concat(preds).sort_index()


def computar_dados() -> dict:
    """Recalcula tudo que as figuras precisam. Devolve um dict de DataFrames.

    Leva alguns minutos: roda backtests com bootstrap em tres horizontes.
    """
    from backtest.bootstrap import block_bootstrap_sharpe
    from backtest.capacity import capacity_sweep
    from backtest.engine import (
        generate_oos_rv_forecast,
        run_backtest,
        run_backtest_delta_hedged,
    )
    from backtest.walk_forward import purged_walk_forward_splits_by_step
    from data.b3_options import add_implied_vols, atm_iv_by_date, load_option_trades
    from vol.forecast import (
        BASELINE_FEATURES,
        forward_target_from_variance,
        har_features_from_variance,
    )
    from vol.iv_trades import filter_quality
    from vol.realized import load_b3_variance

    precos, var = load_b3_variance(estimator="parkinson")
    idx = pd.DatetimeIndex([d.date() for d in precos.index])
    p = pd.Series(precos.to_numpy(), index=idx)
    v = pd.Series(var.to_numpy(), index=idx)
    com_iv = add_implied_vols(load_option_trades())

    # --- Sharpe por horizonte ------------------------------------------
    linhas = []
    for h, (lo_d, hi_d) in ((5, (3, 12)), (10, (8, 20)), (21, (15, 60))):
        atm = atm_iv_by_date(filter_quality(com_iv, dte_range=(lo_d, hi_d)), 0.03)
        iv = atm.groupby("date")["iv_pct"].mean()
        iv.index = pd.DatetimeIndex([pd.Timestamp(d).date() for d in iv.index])
        fc = _previsao_livre_de_escala(v, h)
        reg = {"h": h}
        for suf, spread in (("sem", 0.0), ("com", {5: 0.1712, 10: 0.0560, 21: 0.0694}[h])):
            tr = run_backtest_delta_hedged(p, fc, iv, horizon=h, band_pct=1.0,
                                           spread_pct=spread, allow_overlap=True)
            b = block_bootstrap_sharpe(tr["pnl_net"].to_numpy(), annualization=252 / h,
                                       n_boot=2000, seed=42)
            reg |= {f"sr_{suf}": b["sharpe"], f"lo_{suf}": b["ic_baixo"], f"hi_{suf}": b["ic_alto"]}
        linhas.append(reg)
    sharpe_h = pd.DataFrame(linhas)

    # --- capacidade ------------------------------------------------------
    ds21 = har_features_from_variance(v)
    ds21["target"] = forward_target_from_variance(v, HORIZON_PADRAO)
    ds21 = ds21.dropna()
    cap = capacity_sweep(ds21, BASELINE_FEATURES, horizon=HORIZON_PADRAO, test_size=400)

    # --- delta-hedge antes/depois + curva de P&L -------------------------
    atm21 = atm_iv_by_date(filter_quality(com_iv, dte_range=(15, 60)), 0.03)
    iv21 = atm21.groupby("date")["iv_pct"].mean()
    iv21.index = pd.DatetimeIndex([pd.Timestamp(d).date() for d in iv21.index])
    folds21 = purged_walk_forward_splits_by_step(ds21, 252, HORIZON_PADRAO, HORIZON_PADRAO, EMBARGO)
    fc_nivel = generate_oos_rv_forecast(ds21, BASELINE_FEATURES, HORIZON_PADRAO, 5, EMBARGO,
                                        log_target=True, folds=folds21)
    hedge_linhas, trades_hedge = [], None
    for banda in (0.5, 1.0, 2.0):
        args = dict(horizon=HORIZON_PADRAO, band_pct=banda, spread_pct=0.05)
        sem = run_backtest(p, fc_nivel, iv21, **args)
        com = run_backtest_delta_hedged(p, fc_nivel, iv21, **args)

        def _sr(t):
            x = t["pnl_net"].to_numpy()
            return float(x.mean() / x.std(ddof=1) * np.sqrt(252 / HORIZON_PADRAO))

        hedge_linhas.append({"banda": banda, "sem_hedge": _sr(sem), "com_hedge": _sr(com)})
        if banda == 1.0:
            trades_hedge = com

    # serie de RV realizada (contexto do relatorio): Parkinson em janela de 21
    rv_hist = (np.sqrt(v.rolling(21).mean() * 252) * 100).dropna()

    return {
        "sharpe_horizonte": sharpe_h,
        "capacidade": cap,
        "hedge": pd.DataFrame(hedge_linhas),
        "trades_hedge": trades_hedge,
        "rv_historia": rv_hist,
    }


def carregar_amplitudes(usar_cache: bool = True) -> pd.DataFrame:
    """Min/max por serie-dia, parseados dos boletins da B3, com CACHE.

    O parse e o gargalo do modulo: cada boletim e um zip de ~10 MB que
    descompacta para um XML de ~140 MB, e sao dezenas deles. Sem cache, mexer
    numa cor de grafico custa a releitura inteira. O parquet e derivado e mora
    em data/processed/ (gitignored), entao apagar e seguro.
    """
    from datetime import datetime

    from data.b3_futures import RAW_DIR
    from data.option_spread import parse_option_price_ranges

    cache = (Path(__file__).resolve().parents[1] / "data" / "processed"
             / "option_price_ranges.parquet")
    if usar_cache and cache.exists():
        return pd.read_parquet(cache)

    blocos = []
    for z in sorted(Path(RAW_DIR).glob("PR*.zip")):
        try:
            d = datetime.strptime(z.stem[2:], "%y%m%d").date()
            df = parse_option_price_ranges(z.read_bytes(), d)
        except Exception:
            continue
        if not df.empty:
            blocos.append(df)
    if not blocos:
        return pd.DataFrame(columns=["date", "ticker", "maturity", "option_type",
                                     "strike", "price_min", "price_max", "price_avg", "n_trades"])

    out = pd.concat(blocos, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def computar_spread_por_prazo() -> pd.DataFrame:
    """Spread mediano por faixa de prazo até o vencimento."""
    from data.b3_futures import PROCESSED_PATH as FUT_PATH
    from data.option_spread import effective_spread
    from vol.roll import contract_expiry

    r = carregar_amplitudes()
    if r.empty:
        return pd.DataFrame(columns=["faixa", "mediana", "n"])

    fut = pd.read_parquet(FUT_PATH)
    fut["d"] = pd.to_datetime(fut["date"]).dt.tz_localize(None).dt.normalize()
    ref = fut.set_index("d")["settlement"]

    r["fut"] = r["date"].dt.normalize().map(ref)
    r = r[r["fut"].notna()]
    r["mny"] = (r["strike"] / r["fut"] - 1).abs()
    r["dias"] = (
        pd.to_datetime(r["maturity"].map(lambda m: contract_expiry("DOL" + str(m))))
        - r["date"].dt.normalize()
    ).dt.days

    sp = effective_spread(r[(r["mny"] <= 0.03) & (r["dias"] > 0)])
    sp["dias"] = r.loc[sp.index, "dias"]
    faixas = [(1, 10, "1–10 d"), (11, 20, "11–20 d"), (21, 35, "21–35 d"),
              (36, 60, "36–60 d"), (61, 10_000, "60+ d")]
    linhas = [
        {"faixa": rot, "mediana": float(sub["spread_pct"].median()), "n": len(sub)}
        for lo, hi, rot in faixas
        if len(sub := sp[(sp["dias"] >= lo) & (sp["dias"] <= hi)]) > 0
    ]
    return pd.DataFrame(linhas)


def gerar_todas(out_dir: Path = OUT_DIR) -> list[Path]:
    """Recalcula e grava todas as figuras. Devolve os caminhos escritos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dados = computar_dados()
    escritos = []

    for nome, fig in (
        ("sharpe_por_horizonte", fig_sharpe_por_horizonte(dados["sharpe_horizonte"])),
        ("capacidade_modelo", fig_capacidade(dados["capacidade"])),
        ("hedge_antes_depois", fig_hedge_antes_depois(dados["hedge"])),
        ("pnl_e_drawdown", fig_pnl_e_drawdown(
            dados["trades_hedge"],
            "Backtest com delta-hedge: P&L acumulado e drawdown")),
        ("distribuicao_pnl", fig_distribuicao_pnl(
            dados["trades_hedge"], "Distribuição do resultado por operação")),
        ("rv_realizada", fig_rv_historia(dados["rv_historia"])),
    ):
        caminho = out_dir / f"{nome}.png"
        fig.savefig(caminho, dpi=200, transparent=True, bbox_inches="tight")
        plt.close(fig)
        escritos.append(caminho)
        print(f"  {caminho.name}")

    spread = computar_spread_por_prazo()
    if not spread.empty:
        fig = fig_spread_por_prazo(spread)
        caminho = out_dir / "spread_por_prazo.png"
        fig.savefig(caminho, dpi=200, transparent=True, bbox_inches="tight")
        plt.close(fig)
        escritos.append(caminho)
        print(f"  {caminho.name}")
    else:
        print("  spread_por_prazo.png PULADO -- sem boletins em data/raw/b3_futures/")

    return escritos


if __name__ == "__main__":
    print("gerando figuras (roda backtests, leva alguns minutos)...")
    gerar_todas()
