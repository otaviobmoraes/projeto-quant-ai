"""Testes das figuras de veredito (report/figuras_vereditos.py).

As funcoes de figura sao PURAS de proposito -- recebem o dado pronto. E isso
que permite testa-las com dado sintetico aqui, em vez de rodar backtests de
minutos dentro da suite. `computar_dados()` (que faz I/O e roda backtest) nao
e exercitado aqui pelo mesmo motivo.
"""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from report import figuras_vereditos as fv  # noqa: E402


@pytest.fixture
def sharpe_h():
    return pd.DataFrame({
        "h": [5, 10, 21],
        "sr_sem": [2.798, 0.712, 0.015], "lo_sem": [0.82, 0.02, -0.67], "hi_sem": [4.67, 1.34, 0.57],
        "sr_com": [0.558, 0.030, -0.729], "lo_com": [-1.32, -0.69, -1.30], "hi_com": [2.39, 0.78, -0.21],
    })


@pytest.fixture
def capacidade():
    return pd.DataFrame({
        "capacidade": ["rasa", "média", "funda", "enorme"],
        "r2_dentro": [0.5242, 0.7498, 0.9983, 0.9999],
        "r2_fora": [-0.3226, -0.8455, -0.9239, -0.9732],
    })


def _fecha(fig):
    matplotlib.pyplot.close(fig)


def test_sharpe_por_horizonte_desenha_barras_e_intervalos(sharpe_h):
    fig = fv.fig_sharpe_por_horizonte(sharpe_h)
    ax = fig.axes[0]
    # 3 horizontes x 2 series
    assert len([p for p in ax.patches if p.get_height() != 0]) == 6
    assert len(ax.containers) >= 2  # barras + barras de erro
    rotulos = [t.get_text() for t in ax.get_xticklabels()]
    assert rotulos == ["h = 5 dias úteis", "h = 10 dias úteis", "h = 21 dias úteis"]
    _fecha(fig)


def test_sharpe_usa_virgula_decimal(sharpe_h):
    """Documento em português: separador decimal tem de ser vírgula."""
    fig = fv.fig_sharpe_por_horizonte(sharpe_h)
    textos = [t.get_text() for t in fig.axes[0].texts]
    assert any("+2,80" in t for t in textos)
    assert not any("." in t and t.replace("+", "").replace("-", "").replace(".", "").isdigit()
                   for t in textos)
    _fecha(fig)


def test_capacidade_tem_duas_series_e_legenda(capacidade):
    fig = fv.fig_capacidade(capacidade)
    ax = fig.axes[0]
    assert len(ax.lines) >= 2
    rotulos = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "R² dentro da amostra" in rotulos and "R² fora da amostra" in rotulos
    _fecha(fig)


def test_capacidade_mostra_a_divergencia(capacidade):
    """A figura só cumpre o papel se as duas séries terminarem em lados
    opostos do zero -- é esse contraste que demonstra a memorização."""
    fig = fv.fig_capacidade(capacidade)
    ax = fig.axes[0]
    ydentro = ax.lines[0].get_ydata()
    yfora = ax.lines[1].get_ydata()
    assert ydentro[-1] > 0.9 and yfora[-1] < 0
    _fecha(fig)


def test_spread_por_prazo(capsys):
    dados = pd.DataFrame({
        "faixa": ["1–10 d", "11–20 d", "60+ d"],
        "mediana": [0.1824, 0.0394, 0.0106],
        "n": [21, 24, 40],
    })
    fig = fv.fig_spread_por_prazo(dados)
    ax = fig.axes[0]
    assert len(ax.patches) == 3
    textos = [t.get_text() for t in ax.texts]
    assert any("18,24%" in t for t in textos)
    assert any("n=21" in t for t in textos)
    _fecha(fig)


def test_hedge_antes_depois_marca_troca_de_sinal():
    dados = pd.DataFrame({
        "banda": [0.5, 1.0, 2.0],
        "sem_hedge": [0.301, -0.707, 0.495],
        "com_hedge": [-1.365, -1.739, -0.580],
    })
    fig = fv.fig_hedge_antes_depois(dados)
    ax = fig.axes[0]
    alturas = [p.get_height() for p in ax.patches]
    # sem hedge troca de sinal; com hedge é sempre negativo
    assert any(a > 0 for a in alturas) and any(a < 0 for a in alturas)
    assert all(h < 0 for h in dados["com_hedge"])
    _fecha(fig)


@pytest.fixture
def trades():
    import numpy as np

    rng = np.random.default_rng(3)
    n = 40
    return pd.DataFrame({
        "exit_date": pd.date_range("2020-01-01", periods=n, freq="21D"),
        "pnl_net": np.concatenate([rng.normal(1500, 3000, n - 4), [-42000, -31000, -25000, -18000]]),
    })


def test_pnl_e_drawdown_tem_dois_paineis(trades):
    fig = fv.fig_pnl_e_drawdown(trades, "t")
    assert len(fig.axes) == 2
    ax1, ax2 = fig.axes
    assert "P&L acumulado" in ax1.get_ylabel()
    assert "Drawdown" in ax2.get_ylabel()
    _fecha(fig)


def test_drawdown_e_sempre_menor_ou_igual_a_zero(trades):
    """Drawdown é distância até o pico anterior: nunca positivo."""
    fig = fv.fig_pnl_e_drawdown(trades, "t")
    ydd = fig.axes[1].lines[0].get_ydata()
    assert (ydd <= 1e-9).all()
    assert ydd.min() < 0  # a série de teste tem perdas, então há drawdown
    _fecha(fig)


def test_pnl_declara_que_retorno_percentual_nao_e_calculavel(trades):
    """A ressalva não é opcional: sem ela o gráfico convida a leitura errada."""
    fig = fv.fig_pnl_e_drawdown(trades, "t")
    textos = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "não é calculável" in textos
    _fecha(fig)


def test_distribuicao_reporta_assimetria(trades):
    fig = fv.fig_distribuicao_pnl(trades, "t")
    textos = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "assimetria" in textos
    assert "operações" in textos
    _fecha(fig)


def test_figuras_usam_a_paleta_validada(sharpe_h):
    """Ocre + azul foram validados juntos no script da skill de dataviz; se
    alguém trocar por outro par, o teste avisa."""
    from report.plots import SERIES_BLUE

    assert fv.SERIES_OCRE == "#b8752a"
    fig = fv.fig_sharpe_por_horizonte(sharpe_h)
    cores = {matplotlib.colors.to_hex(p.get_facecolor()) for p in fig.axes[0].patches}
    assert fv.SERIES_OCRE in cores and SERIES_BLUE in cores
    _fecha(fig)
