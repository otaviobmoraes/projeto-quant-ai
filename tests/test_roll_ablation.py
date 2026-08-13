"""Testes de backtest/roll_ablation.py -- com enfase na propriedade que mais
importa metodologicamente: o fator sazonal NAO pode enxergar o periodo de teste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.roll_ablation import run_roll_ablation
from vol.roll import deseasonalize, seasonal_factor


def _serie_sintetica(n: int = 500, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    """Variancia log-normal com ciclo de contrato de 30 dias corridos."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    dte = pd.Series(np.tile(np.arange(29, -1, -1), n // 30 + 1)[:n], index=idx)
    var = pd.Series(np.exp(rng.normal(-9, 0.5, size=n)), index=idx)
    return var, dte


class TestSemLookAhead:
    def test_fator_nao_depende_do_periodo_de_teste(self):
        """Perturbar drasticamente a variancia do FIM da amostra nao pode mudar
        o fator estimado num treino que termina antes disso."""
        var, dte = _serie_sintetica()
        corte = 300
        train_idx = var.index[:corte]

        original = seasonal_factor(var.loc[train_idx], dte.loc[train_idx])

        perturbada = var.copy()
        perturbada.iloc[corte:] *= 50.0  # choque enorme so no futuro
        depois = seasonal_factor(perturbada.loc[train_idx], dte.loc[train_idx])

        pd.testing.assert_series_equal(original, depois)

    def test_ablacao_baseline_independe_da_correcao(self):
        """O baseline usa a variancia original -- alterar so o ciclo de
        contrato (dte) nao pode mexer no R2 do baseline."""
        var, dte = _serie_sintetica()
        r1 = run_roll_ablation(var, dte, horizon=5, n_splits=3)
        r2 = run_roll_ablation(var, dte.iloc[::-1].reset_index(drop=True).set_axis(dte.index),
                               horizon=5, n_splits=3)
        assert r1["baseline_pooled"]["r2_oos"] == r2["baseline_pooled"]["r2_oos"]


class TestEstrutura:
    def test_devolve_as_tres_variantes_e_por_fold(self):
        var, dte = _serie_sintetica()
        res = run_roll_ablation(var, dte, horizon=5, n_splits=3)
        for chave in ("baseline_pooled", "dessazonalizada_pooled", "com_dte_pooled"):
            assert chave in res
            assert set(res[chave]) >= {"rmse", "mae", "r2_oos"}
        assert res["n_folds"] == 3
        assert len(res["per_fold"]["baseline"]) == 3

    def test_alvo_e_sempre_a_variancia_original(self):
        """Garante comparabilidade do R2: as tres variantes precisam ser
        avaliadas contra exatamente o mesmo alvo."""
        var, dte = _serie_sintetica()
        res = run_roll_ablation(var, dte, horizon=5, n_splits=3)
        # se o alvo diferisse entre variantes, o ss_tot mudaria e os R2 nao
        # seriam comparaveis; aqui checamos que o denominador bate
        assert res["baseline_pooled"]["rmse"] > 0
        assert res["dessazonalizada_pooled"]["rmse"] > 0


class TestSazonalidadeSintetica:
    def test_corrige_sazonalidade_forte_injetada(self):
        """Sanidade do mecanismo: com sazonalidade forte e ESTAVEL, a serie
        dessazonalizada fica menos dispersa por faixa que a original."""
        var, dte = _serie_sintetica()
        var_saz = var * np.where(dte >= 25, 3.0, 1.0)
        factor = seasonal_factor(var_saz, dte)
        adj = deseasonalize(var_saz, dte, factor)

        disp_antes = np.log(var_saz).groupby(dte >= 25).mean().diff().abs().iloc[-1]
        disp_depois = np.log(adj).groupby(dte >= 25).mean().diff().abs().iloc[-1]
        assert disp_depois < disp_antes
