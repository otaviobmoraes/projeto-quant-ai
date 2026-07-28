import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from report import plots


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _series(n: int = 30, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(10, 2, n), index=idx)


def test_plot_rv_iv_returns_figure_with_two_lines_and_legend():
    rv = _series(30, seed=1)
    iv = _series(30, seed=2)

    fig = plots.plot_rv_iv(rv, iv)

    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert ax.get_legend() is not None


def test_plot_spread_uses_diverging_fill():
    rv = _series(20, seed=3)
    iv = pd.Series(10.0, index=rv.index)

    fig = plots.plot_spread(rv, iv)

    ax = fig.axes[0]
    # fill_between com where= gera PolyCollections (pode ser 0 se nao houver
    # nenhum trecho positivo ou negativo, mas aqui ha os dois).
    assert len(ax.collections) >= 1


def test_plot_ablation_folds_has_one_bar_pair_per_fold():
    baseline = [{"r2_oos": -0.1}, {"r2_oos": -0.2}, {"r2_oos": -0.05}]
    com_noticia = [{"r2_oos": -0.05}, {"r2_oos": -0.15}, {"r2_oos": 0.0}]

    fig = plots.plot_ablation_folds(baseline, com_noticia)

    ax = fig.axes[0]
    # 2 barras por fold (baseline + com_noticia) = 2 * n_folds patches
    assert len(ax.patches) == 2 * len(baseline)
    assert len(ax.get_xticklabels()) == len(baseline)


def test_plot_series_returns_single_line_figure():
    tone = _series(15, seed=4)
    fig = plots.plot_series(tone, title="Tom GDELT", ylabel="tom")

    ax = fig.axes[0]
    assert len(ax.lines) == 1


def test_plot_cumulative_pnl_has_line_and_win_loss_markers():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    trades = pd.DataFrame(
        {
            "exit_date": idx,
            "pnl_net": [10.0, -5.0, 3.0, -1.0],
        }
    )

    fig = plots.plot_cumulative_pnl(trades)
    ax = fig.axes[0]

    assert len(ax.lines) == 2  # curva de pnl acumulado + linha de zero
    assert len(ax.collections) == 2  # scatter de ganhos + scatter de perdas
    assert ax.get_legend() is not None


def test_break_gaps_inserts_nan_after_large_gap():
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-20", "2026-01-21"])
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    result = plots._break_gaps(series, max_gap_days=5)

    assert result.isna().sum() == 1
    assert len(result) == len(series) + 1


def test_break_gaps_no_change_when_no_large_gaps():
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    series = pd.Series(range(5), index=idx, dtype=float)

    result = plots._break_gaps(series, max_gap_days=5)

    assert result.isna().sum() == 0
    assert len(result) == len(series)
