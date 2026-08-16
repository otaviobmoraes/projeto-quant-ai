"""Diagrama de blocos do pipeline de MACHINE LEARNING do projeto: das fontes
de dado ate a metrica final, com o que entrou no caminho oficial e o que foi
testado e REJEITADO.

POR QUE O DIAGRAMA MARCA OS REJEITADOS. Um diagrama que mostra so o caminho
que funcionou descreve um projeto que nao existiu: 51 configuracoes foram
testadas e a maioria falhou. As camadas rejeitadas (noticia, credibilidade,
risco fiscal, risco global, painel de moedas) consumiram a maior parte do
trabalho e sao a evidencia de que o veredito final nao e falta de tentativa.
Elas aparecem em traco pontilhado -- presentes, e visivelmente fora do fluxo.

POR QUE ESTE ARQUIVO ESTA NO REPOSITORIO. Os tres diagramas anteriores do
relatorio (`diagrama_pipeline.png`, `diagrama_modulos.png`,
`diagrama_walk_forward.png`) foram gerados por scripts de scratchpad que se
perderam -- as imagens existem e nao ha como regera-las. Este nao.

Uso: `python -m report.ml_diagram` grava em report/output/diagrama_ml.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from report.plots import (  # noqa: E402
    COLOR_BASELINE_AXIS,
    COLOR_MUTED,
    COLOR_PRIMARY_INK,
    COLOR_SECONDARY_INK,
    COLOR_SURFACE,
    SERIES_BLUE,
)

OUT_PATH = Path(__file__).resolve().parent / "output" / "diagrama_ml.png"

# Ocre para o caminho oficial (mesma familia da pagina de vereditos); o azul de
# report.plots marca as fontes de dado. Rejeitado nao ganha cor -- fica em
# cinza pontilhado, que e exatamente o peso visual que merece.
COLOR_ACCENT = "#b8752a"
COLOR_REJECTED = "#8e8a80"
COLOR_PANEL_BG = "#f4f2ed"

# (titulo da coluna, x do centro)
COLUNAS = [
    ("1 · ENTRADAS", 9.0),
    ("2 · FEATURES", 27.5),
    ("3 · MODELOS", 46.0),
    ("4 · VALIDAÇÃO", 64.5),
    ("5 · ESTRATÉGIA", 83.0),
]
LARGURA = 16.0


def _caixa(
    ax,
    x: float,
    y: float,
    titulo: str,
    detalhe: str = "",
    *,
    cor: str = COLOR_ACCENT,
    rejeitado: bool = False,
    altura: float = 7.4,
) -> None:
    """Uma caixa do diagrama. `rejeitado=True` desenha em traco pontilhado e
    tinta apagada -- a convencao visual que separa o que entrou no caminho
    oficial do que foi testado e descartado."""
    borda = COLOR_REJECTED if rejeitado else cor
    fundo = COLOR_SURFACE if rejeitado else COLOR_PANEL_BG
    tinta = COLOR_REJECTED if rejeitado else COLOR_PRIMARY_INK

    ax.add_patch(
        FancyBboxPatch(
            (x - LARGURA / 2, y - altura / 2),
            LARGURA,
            altura,
            boxstyle="round,pad=0.35,rounding_size=0.8",
            linewidth=1.4,
            linestyle=(0, (3, 2)) if rejeitado else "solid",
            edgecolor=borda,
            facecolor=fundo,
            zorder=3,
        )
    )
    dy = 1.05 if detalhe else 0.0
    ax.text(
        x, y + dy, titulo,
        ha="center", va="center", fontsize=8.6, color=tinta,
        fontweight="normal" if rejeitado else "bold", zorder=4,
    )
    if detalhe:
        ax.text(
            x, y - 1.5, detalhe,
            ha="center", va="center", fontsize=7.2,
            color=COLOR_REJECTED if rejeitado else COLOR_SECONDARY_INK, zorder=4,
        )


def _seta(ax, x0: float, y0: float, x1: float, y1: float, *, cor: str = COLOR_ACCENT) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=11,
            linewidth=1.3, color=cor, zorder=2,
            shrinkA=2, shrinkB=2,
        )
    )


def build_ml_diagram() -> Figure:
    """Monta o diagrama completo e devolve a figura (sem tocar disco)."""
    fig, ax = plt.subplots(figsize=(17.5, 9.2), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    ax.set_xlim(0, 92)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- cabeçalhos de coluna -------------------------------------------
    for titulo, x in COLUNAS:
        # espaçamento entre letras via caractere fino, já que matplotlib não
        # expõe letter-spacing em Text
        ax.text(x, 95.5, " ".join(titulo), ha="center", va="center", fontsize=8.6,
                color=COLOR_MUTED, fontweight="bold")
        ax.plot([x - 9.5, x + 9.5], [92.5, 92.5], color=COLOR_BASELINE_AXIS,
                linewidth=1.1, zorder=1)

    x1, x2, x3, x4, x5 = (c[1] for c in COLUNAS)

    # ---- 1. entradas ------------------------------------------------------
    _caixa(ax, x1, 82, "Futuro de dólar B3 (BVBG-086)",
           "OHLC + ajuste · 2.135 pregões · 2018–2026", cor=SERIES_BLUE)
    _caixa(ax, x1, 71, "Negócios de opção B3",
           "13.350 negócios → IV via Black-76", cor=SERIES_BLUE)
    _caixa(ax, x1, 60, "Spread bid-ask (MinPric/MaxPric)",
           "mediana 4,43% do prêmio", cor=SERIES_BLUE)

    ax.text(x1, 50.5, "camadas exógenas testadas", ha="center", va="center",
            fontsize=7.6, color=COLOR_REJECTED, style="italic")
    _caixa(ax, x1, 43, "GDELT — tom e volume de notícia",
           "+ FinBERT-PT-BR nas manchetes", rejeitado=True)
    _caixa(ax, x1, 33, "Focus/COPOM — credibilidade", "θ de Barro-Gordon", rejeitado=True)
    _caixa(ax, x1, 23, "VIX + DXY — risco global", "", rejeitado=True, altura=6.0)
    _caixa(ax, x1, 14, "Painel de 8 moedas EM", "~16.000 linhas", rejeitado=True)

    # ---- 2. features ------------------------------------------------------
    _caixa(ax, x2, 82, "Estimadores de variância diária",
           "Parkinson · Garman-Klass · Rogers-Satchell")
    _caixa(ax, x2, 71, "Features HAR",
           "rv_d · rv_w (5d) · rv_m (22d)")
    _caixa(ax, x2, 60, "Features livres de escala",
           "rv_d/rv_m · rv_w/rv_m  (adimensionais)")
    _caixa(ax, x2, 43, "Alvo: RV futura de h dias",
           "h ∈ {1, 5, 10, 21} · anualizada, em log")
    _caixa(ax, x2, 27, "Ablação obrigatória",
           "cada camada: com vs sem", altura=6.4)

    # ---- 3. modelos -------------------------------------------------------
    _caixa(ax, x3, 84, "Persistência (zero parâmetros)",
           "benchmark que tudo precisa bater")
    _caixa(ax, x3, 73, "HAR-RV — OLS em log",
           "baseline oficial (Corsi, 2009)")
    _caixa(ax, x3, 62, "HAR livre de escala ★",
           "alvo em razão · passa 3 portões em h=1")
    _caixa(ax, x3, 48, "GARCH(1,1)", "ganha em h=1, perde em h≥5", rejeitado=True)
    _caixa(ax, x3, 38, "XGBoost — árvore e linear",
           "perde do HAR em todo h ≥ 3", rejeitado=True)
    _caixa(ax, x3, 28, "Modelo global em painel",
           "reprova nos 3 portões", rejeitado=True)

    # ---- 4. validação -----------------------------------------------------
    _caixa(ax, x4, 84, "Walk-forward PURGADO",
           "embargo 5d · sem split aleatório")
    _caixa(ax, x4, 73, "Três portões (pré-registrados)",
           "ΔR²>0 · ≥4/5 folds · DM p<0,05")
    _caixa(ax, x4, 62, "Por subperíodo, não pooled",
           "regime mistura e infla o agregado")
    _caixa(ax, x4, 51, "Bootstrap de bloco",
           "Ledoit & Wolf (2008) · janelas sobrepostas")
    _caixa(ax, x4, 40, "Diagnóstico de limite",
           "teto em amostra · curva de aprendizado")
    _caixa(ax, x4, 29, "Deflated Sharpe",
           "desconta 51 configurações testadas")

    # ---- 5. estratégia ----------------------------------------------------
    _caixa(ax, x5, 78, "Previsão de RV (h dias)")
    _caixa(ax, x5, 67, "Sinal: RV prevista vs IV",
           "banda morta · compra ou vende vol")
    _caixa(ax, x5, 56, "Straddle ATM sobre futuro",
           "sizing por vega ou por risco")
    _caixa(ax, x5, 45, "Delta-hedge diário",
           "isola vol; corta a dispersão em ~65%")
    _caixa(ax, x5, 34, "Sharpe · PSR · DSR",
           "e drawdown em unidades absolutas")

    # ---- fluxo principal --------------------------------------------------
    for x_ini, x_fim in ((x1, x2), (x2, x3), (x3, x4), (x4, x5)):
        _seta(ax, x_ini + LARGURA / 2 + 0.6, 71, x_fim - LARGURA / 2 - 0.6, 71)

    _seta(ax, x2, 55.6, x2, 47.6)
    _seta(ax, x2 + LARGURA / 2 + 0.6, 43, x3 - LARGURA / 2 - 0.6, 62)

    # realimentação: a validação decide o que volta para o modelo
    ax.annotate(
        "", xy=(x3, 20.5), xytext=(x4, 20.5),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_REJECTED, linewidth=1.2,
                        linestyle=(0, (4, 2)), shrinkA=0, shrinkB=0),
    )
    ax.text((x3 + x4) / 2, 17.6, "reprovou → registrado em CONFIGS_TESTED",
            ha="center", va="center", fontsize=7.2, color=COLOR_REJECTED, style="italic")

    # ---- legenda ----------------------------------------------------------
    # posicionada sob as colunas 2-3, a unica faixa larga livre do diagrama
    lx, ly = 21.0, 3.0
    ax.add_patch(FancyBboxPatch((lx, ly), 26.0, 7.4,
                boxstyle="round,pad=0.3,rounding_size=0.6",
                linewidth=1.0, edgecolor=COLOR_BASELINE_AXIS,
                facecolor=COLOR_SURFACE, zorder=3))
    ax.add_patch(FancyBboxPatch((lx + 1.5, ly + 3.9), 3.0, 2.0,
                boxstyle="round,pad=0.15,rounding_size=0.4",
                linewidth=1.4, edgecolor=COLOR_ACCENT,
                facecolor=COLOR_PANEL_BG, zorder=4))
    ax.text(lx + 6.0, ly + 4.9, "no caminho oficial", ha="left", va="center",
            fontsize=7.6, color=COLOR_PRIMARY_INK, zorder=4)
    ax.add_patch(FancyBboxPatch((lx + 1.5, ly + 1.0), 3.0, 2.0,
                boxstyle="round,pad=0.15,rounding_size=0.4",
                linewidth=1.4, linestyle=(0, (3, 2)), edgecolor=COLOR_REJECTED,
                facecolor=COLOR_SURFACE, zorder=4))
    ax.text(lx + 6.0, ly + 2.0, "testado e rejeitado", ha="left", va="center",
            fontsize=7.6, color=COLOR_REJECTED, zorder=4)

    ax.text(91.5, 5.0,
            "★ único modelo que passa nos três portões — e só em h=1,\n"
            "   que não é o horizonte que a estratégia opera.",
            ha="right", va="center", fontsize=7.6, color=COLOR_SECONDARY_INK)

    fig.suptitle(
        "VolBoy — pipeline de previsão de volatilidade",
        x=0.5, y=0.985, fontsize=13.5, color=COLOR_PRIMARY_INK, fontweight="bold",
    )
    fig.text(0.5, 0.955,
             "Opções de dólar (USD/BRL) · 51 configurações registradas · "
             "o que entrou no caminho oficial e o que foi descartado",
             ha="center", fontsize=8.8, color=COLOR_MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def save_ml_diagram(path: Path = OUT_PATH, dpi: int = 200) -> Path:
    """Grava o diagrama em disco e devolve o caminho."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_ml_diagram()
    fig.savefig(path, dpi=dpi, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(f"gravado em {save_ml_diagram()}")
