"""Graficos para o pre-relatorio e a entrega final (CLAUDE.md, Fase 7).

Paleta e convencoes de cor seguem a skill de dataviz do projeto: cores
categoricas fixas por serie (nunca por rank), um eixo so (nunca eixo duplo
-- RV e IV estao na mesma unidade, vol % ao ano), gridlines/eixos recessivos,
legenda sempre presente com >=2 series. Graficos estaticos (matplotlib) para
embutir no documento do pre-relatorio -- sem camada de interacao (isso e
para artifacts web, fora do escopo aqui).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

# Paleta (ver skill de dataviz, references/palette.md) -- modo claro.
COLOR_SURFACE = "#fcfcfb"
COLOR_PRIMARY_INK = "#0b0b0b"
COLOR_SECONDARY_INK = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRIDLINE = "#e1e0d9"
COLOR_BASELINE_AXIS = "#c3c2b7"

SERIES_BLUE = "#2a78d6"
SERIES_GREEN = "#008300"
DIVERGING_BLUE = "#2a78d6"
DIVERGING_RED = "#e34948"
DIVERGING_NEUTRAL = "#f0efec"


def _break_gaps(series: pd.Series, max_gap_days: int = 5) -> pd.Series:
    """Insere NaN nos pontos logo apos um gap > `max_gap_days` na serie, pra
    matplotlib interromper a linha em vez de interpolar visualmente por cima
    de um buraco de dados real (ex.: janelas do GDELT que falharam por
    rate-limit e nao foram preenchidas).
    """
    if len(series) < 2:
        return series
    gaps = series.index.to_series().diff().dt.days
    break_points = series.index[gaps > max_gap_days]
    if len(break_points) == 0:
        return series
    filler = pd.Series(
        float("nan"), index=break_points - pd.Timedelta(days=max_gap_days / 2), name=series.name
    )
    return pd.concat([series, filler]).sort_index()


def _new_axes() -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, color=COLOR_GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_BASELINE_AXIS)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    return fig, ax


def plot_rv_iv(rv: pd.Series, iv: pd.Series, title: str = "RV realizada vs IV implicita (ATM)") -> Figure:
    """Serie temporal de RV realizada (trailing) e IV ATM implicita, mesma
    unidade (vol % ao ano) -- um eixo so, como manda a regra.
    """
    rv = _break_gaps(rv)
    iv = _break_gaps(iv)

    fig, ax = _new_axes()
    ax.plot(rv.index, rv.to_numpy(), color=SERIES_GREEN, linewidth=1.8, label="RV realizada", zorder=3)
    ax.plot(iv.index, iv.to_numpy(), color=SERIES_BLUE, linewidth=1.8, label="IV ATM", zorder=3)
    ax.set_ylabel("Vol (% a.a.)", color=COLOR_SECONDARY_INK, fontsize=10)
    ax.set_title(title, color=COLOR_PRIMARY_INK, fontsize=12, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, labelcolor=COLOR_SECONDARY_INK)
    fig.tight_layout()
    return fig


def plot_spread(rv: pd.Series, iv: pd.Series, title: str = "Spread RV - IV") -> Figure:
    """Spread RV-IV como area divergente: azul acima de zero (RV > IV,
    comprar vol), vermelho abaixo (RV < IV, vender vol) -- par divergente com
    zero como o meio-termo neutro.
    """
    aligned = pd.concat([rv.rename("rv"), iv.rename("iv")], axis=1, join="inner").dropna()
    spread = aligned["rv"] - aligned["iv"]

    fig, ax = _new_axes()
    ax.axhline(0, color=COLOR_BASELINE_AXIS, linewidth=1.0, zorder=2)
    ax.fill_between(
        spread.index, spread.to_numpy(), 0, where=spread >= 0, color=DIVERGING_BLUE, alpha=0.6, zorder=3
    )
    ax.fill_between(
        spread.index, spread.to_numpy(), 0, where=spread < 0, color=DIVERGING_RED, alpha=0.6, zorder=3
    )
    ax.set_ylabel("RV - IV (p.p.)", color=COLOR_SECONDARY_INK, fontsize=10)
    ax.set_title(title, color=COLOR_PRIMARY_INK, fontsize=12, loc="left")
    fig.tight_layout()
    return fig


def plot_ablation_folds(
    per_fold_baseline: list[dict], per_fold_com_noticia: list[dict], metric: str = "r2_oos"
) -> Figure:
    """Barras agrupadas comparando baseline vs com-noticia fold a fold --
    mesma convencao de cor (azul/verde) usada em plot_rv_iv, pra series
    manterem identidade visual entre os graficos do relatorio.
    """
    n = len(per_fold_baseline)
    x = range(n)
    width = 0.36

    fig, ax = _new_axes()
    ax.bar(
        [i - width / 2 for i in x],
        [f[metric] for f in per_fold_baseline],
        width=width,
        color=SERIES_BLUE,
        label="Baseline",
        zorder=3,
    )
    ax.bar(
        [i + width / 2 for i in x],
        [f[metric] for f in per_fold_com_noticia],
        width=width,
        color=SERIES_GREEN,
        label="Com noticia",
        zorder=3,
    )
    ax.axhline(0, color=COLOR_BASELINE_AXIS, linewidth=1.0, zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Fold {i+1}" for i in x], color=COLOR_MUTED, fontsize=9)
    ax.set_ylabel(metric, color=COLOR_SECONDARY_INK, fontsize=10)
    ax.set_title("Ablacao HAR-RV por fold (walk-forward purgado)", color=COLOR_PRIMARY_INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=COLOR_SECONDARY_INK)
    fig.tight_layout()
    return fig


def plot_series(series: pd.Series, title: str, ylabel: str, color: str = SERIES_BLUE) -> Figure:
    """Grafico de linha generico para uma unica serie (ex.: tom do GDELT)."""
    series = _break_gaps(series)
    fig, ax = _new_axes()
    ax.plot(series.index, series.to_numpy(), color=color, linewidth=1.6, zorder=3)
    ax.set_ylabel(ylabel, color=COLOR_SECONDARY_INK, fontsize=10)
    ax.set_title(title, color=COLOR_PRIMARY_INK, fontsize=12, loc="left")
    fig.tight_layout()
    return fig
