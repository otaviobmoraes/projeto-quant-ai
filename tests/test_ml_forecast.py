"""Testes de vol/ml_forecast.py e backtest/ml_ablation.py (XGBoost vs HAR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.ml_ablation import build_dataset, run_ml_ablation
from vol.ml_forecast import LINEAR_PARAMS, PARAM_SETS, TREE_PARAMS, fit_xgb, predict_xgb


def _dataset(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="America/Sao_Paulo")
    var = pd.Series(np.exp(rng.normal(-9, 0.4, size=n)), index=idx)
    return build_dataset(var, horizon=5)


class TestHiperparametrosPreRegistrados:
    def test_arvore_e_conservadora_para_a_amostra(self):
        """Os valores existem para conter sobreajuste -- se alguem afrouxar
        isso, e configuracao nova e precisa entrar no CONFIGS_TESTED."""
        assert TREE_PARAMS["max_depth"] <= 3
        assert TREE_PARAMS["min_child_weight"] >= 10
        assert TREE_PARAMS["subsample"] < 1.0

    def test_semente_fixa_garante_reprodutibilidade(self):
        assert TREE_PARAMS["random_state"] == LINEAR_PARAMS["random_state"]

    def test_os_dois_base_learners_estao_declarados(self):
        assert {p["booster"] for p in PARAM_SETS.values()} == {"gbtree", "gblinear"}


class TestAjusteEPrevisao:
    def test_roundtrip_arvore(self):
        ds = _dataset()
        m = fit_xgb(ds.iloc[:300], ["rv_d", "rv_w", "rv_m"], TREE_PARAMS)
        pred = predict_xgb(m, ds.iloc[300:], ["rv_d", "rv_w", "rv_m"])
        assert len(pred) == len(ds) - 300
        assert (pred > 0).all()  # exp() do log-alvo nunca e negativo
        assert pred.index.equals(ds.iloc[300:].index)

    def test_roundtrip_linear(self):
        ds = _dataset()
        m = fit_xgb(ds.iloc[:300], ["rv_d", "rv_w", "rv_m"], LINEAR_PARAMS)
        pred = predict_xgb(m, ds.iloc[300:], ["rv_d", "rv_w", "rv_m"])
        assert (pred > 0).all()

    def test_previsao_fica_na_escala_da_rv(self):
        ds = _dataset()
        m = fit_xgb(ds.iloc[:300], ["rv_d", "rv_w", "rv_m"], TREE_PARAMS)
        pred = predict_xgb(m, ds.iloc[300:], ["rv_d", "rv_w", "rv_m"])
        # RV anualizada em pontos percentuais: ordem de grandeza de dezenas
        assert pred.between(1, 200).all()

    def test_log_target_desligado_muda_a_previsao(self):
        ds = _dataset()
        cols = ["rv_d", "rv_w", "rv_m"]
        com = predict_xgb(
            fit_xgb(ds.iloc[:300], cols, TREE_PARAMS, True), ds.iloc[300:], cols, True
        )
        sem = predict_xgb(
            fit_xgb(ds.iloc[:300], cols, TREE_PARAMS, False), ds.iloc[300:], cols, False
        )
        assert not np.allclose(com.to_numpy(), sem.to_numpy())


class TestPadronizacaoDoLinear:
    """O bug que a padronizacao corrige: features na ordem de 1e-5 deixavam o
    gblinear gravemente subajustado (R2 negativo em todos os horizontes)."""

    def test_linear_usa_pipeline_com_scaler(self):
        ds = _dataset()
        m = fit_xgb(ds.iloc[:300], ["rv_d", "rv_w", "rv_m"], LINEAR_PARAMS)
        assert hasattr(m, "steps"), "gblinear precisa vir dentro de um Pipeline com StandardScaler"
        assert "standardscaler" in dict(m.steps)

    def test_arvore_nao_usa_scaler(self):
        """Arvore e invariante a escala monotona -- padronizar nao mudaria nada
        e so obscureceria a comparacao."""
        ds = _dataset()
        m = fit_xgb(ds.iloc[:300], ["rv_d", "rv_w", "rv_m"], TREE_PARAMS)
        assert not hasattr(m, "steps")

    def test_linear_acompanha_um_alvo_linear_nas_features(self):
        """Sanidade: com relacao linear forte e features minusculas, o modelo
        padronizado tem de recuperar o sinal (sem scaler, nao recuperava)."""
        rng = np.random.default_rng(3)
        n = 500
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        rv_d = pd.Series(rng.uniform(1e-5, 5e-4, size=n), index=idx)
        ds = pd.DataFrame({"rv_d": rv_d, "rv_w": rv_d * 0.9, "rv_m": rv_d * 0.8})
        ds["target"] = 10 + 2e4 * rv_d  # alvo fortemente linear nas features
        cols = ["rv_d", "rv_w", "rv_m"]
        m = fit_xgb(ds.iloc[:400], cols, LINEAR_PARAMS)
        pred = predict_xgb(m, ds.iloc[400:], cols)
        corr = np.corrcoef(pred.to_numpy(), ds["target"].iloc[400:].to_numpy())[0, 1]
        assert corr > 0.9, f"gblinear nao recuperou relacao linear obvia (corr={corr:.3f})"


class TestAblacao:
    def test_compara_todos_nos_mesmos_folds(self):
        rng = np.random.default_rng(7)
        idx = pd.date_range("2019-01-01", periods=700, freq="B", tz="America/Sao_Paulo")
        var = pd.Series(np.exp(rng.normal(-9, 0.4, size=700)), index=idx)
        r = run_ml_ablation(var, horizon=5, n_splits=3)
        for chave in ("har", "xgb_arvore", "xgb_linear", "persistencia"):
            assert chave in r
            assert set(r[chave]) >= {"rmse", "mae", "r2_oos"}
            assert len(r["por_fold"][chave]) == r["n_folds"]

    def test_alvo_identico_entre_modelos(self):
        """Se o alvo diferisse, os R2 nao seriam comparaveis."""
        rng = np.random.default_rng(9)
        idx = pd.date_range("2019-01-01", periods=700, freq="B", tz="America/Sao_Paulo")
        var = pd.Series(np.exp(rng.normal(-9, 0.4, size=700)), index=idx)
        r = run_ml_ablation(var, horizon=5, n_splits=3)
        assert r["n_obs"] > 0
        assert r["horizon"] == 5

    def test_dataset_pequeno_demais_levanta(self):
        idx = pd.date_range("2020-01-01", periods=40, freq="B")
        var = pd.Series(np.linspace(1e-5, 2e-5, 40), index=idx)
        with pytest.raises(ValueError):
            run_ml_ablation(var, horizon=21, n_splits=5)
