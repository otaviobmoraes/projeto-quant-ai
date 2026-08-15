"""Testes da previsao LIVRE DE ESCALA (alvo em razao) -- vol/forecast.py.

Motivacao medida (ver docstring da secao em vol/forecast.py): o HAR em nivel
tem viés que troca de sinal entre folds (-9% a +11%) porque o treino expansivo
carrega regimes antigos. Prever a razao RV_futura/RV_corrente tira o nivel da
regressao; ele volta pela multiplicacao pela persistencia, que acompanha o
regime por construcao.
"""

import numpy as np
import pandas as pd
import pytest

from vol import forecast


def _dataset_sintetico(n: int = 400, seed: int = 0, escala: float = 1.0) -> pd.DataFrame:
    """Dataset HAR sintetico com `escala` multiplicando o NIVEL da vol --
    permite testar invariancia de escala de forma direta."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    var_diaria = pd.Series(np.exp(rng.normal(-10, 0.5, n)) * escala**2, index=idx)
    df = forecast.har_features_from_variance(var_diaria)
    df["target"] = forecast.forward_target_from_variance(var_diaria, horizon=21)
    return df.dropna()


def test_target_ratio_e_alvo_dividido_pela_persistencia():
    ds = _dataset_sintetico()
    sf = forecast.build_scale_free_dataset(ds)
    esperado = ds["target"] / forecast.persistence_forecast(ds)
    pd.testing.assert_series_equal(
        sf["target_ratio"], esperado.reindex(sf.index), check_names=False
    )


def test_features_sao_razoes_das_componentes_har():
    ds = _dataset_sintetico()
    sf = forecast.build_scale_free_dataset(ds)
    assert sf["ratio_d"].equals((ds["rv_d"] / ds["rv_m"]).reindex(sf.index))
    assert sf["ratio_w"].equals((ds["rv_w"] / ds["rv_m"]).reindex(sf.index))


def test_features_e_alvo_sao_invariantes_a_escala():
    """O ponto do desenho: dobrar o NIVEL de vol nao muda nem as features nem
    o alvo em razao. E isso que faz o modelo nao depender do regime."""
    baixa = forecast.build_scale_free_dataset(_dataset_sintetico(escala=1.0, seed=3))
    alta = forecast.build_scale_free_dataset(_dataset_sintetico(escala=3.0, seed=3))

    for col in [*forecast.SCALE_FREE_FEATURES, "target_ratio"]:
        np.testing.assert_allclose(baixa[col].to_numpy(), alta[col].to_numpy(), rtol=1e-9)

    # ja o alvo em NIVEL muda por construcao (e o que quebrava o R2 entre regimes)
    assert alta["target"].mean() == pytest.approx(3 * baixa["target"].mean(), rel=1e-9)


def test_previsao_volta_para_a_escala_de_nivel():
    ds = _dataset_sintetico()
    sf = forecast.build_scale_free_dataset(ds)
    train, test = sf.iloc[:300], sf.iloc[300:]

    model = forecast.fit_har_scale_free(train)
    pred = forecast.predict_scale_free(model, test)

    assert pred.index.equals(test.index)
    assert (pred > 0).all()
    # mesma ordem de grandeza do alvo em nivel
    assert 0.5 < pred.mean() / test["target"].mean() < 2.0


def test_correcao_de_jensen_aumenta_a_previsao():
    """exp(E[log Y]) e a MEDIANA; a media log-normal e exp(mu + sigma^2/2).
    Sem a correcao a previsao de nivel sai sistematicamente baixa."""
    ds = _dataset_sintetico()
    sf = forecast.build_scale_free_dataset(ds)
    train, test = sf.iloc[:300], sf.iloc[300:]
    model = forecast.fit_har_scale_free(train)

    com = forecast.predict_scale_free(model, test, jensen_correction=True)
    sem = forecast.predict_scale_free(model, test, jensen_correction=False)

    assert (com > sem).all()
    assert (com / sem).std() == pytest.approx(0.0, abs=1e-12)  # fator constante
    assert com.mean() / sem.mean() == pytest.approx(np.exp(model.mse_resid / 2), rel=1e-9)


def test_previsao_recupera_relacao_conhecida_na_razao():
    """Se a razao futura for uma funcao deterministica de ratio_d, o modelo
    tem que recuperar isso quase exatamente."""
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(11)
    sf = pd.DataFrame(index=idx)
    sf["ratio_d"] = rng.uniform(0.5, 1.5, n)
    sf["ratio_w"] = rng.uniform(0.8, 1.2, n)
    sf["rv_trailing"] = 12.0
    sf["target_ratio"] = np.exp(0.30 * sf["ratio_d"] - 0.10)
    sf["target"] = sf["target_ratio"] * sf["rv_trailing"]

    model = forecast.fit_har_scale_free(sf.iloc[:300])
    pred = forecast.predict_scale_free(model, sf.iloc[300:], jensen_correction=False)

    np.testing.assert_allclose(pred.to_numpy(), sf["target"].iloc[300:].to_numpy(), rtol=1e-6)


# ---------------------------------------------------------------------------
# forecast_metrics (aritmetica unica compartilhada)
# ---------------------------------------------------------------------------


def test_forecast_metrics_bate_com_calculo_manual():
    target = pd.Series([10.0, 12.0, 14.0, 16.0])
    pred = pd.Series([11.0, 11.0, 15.0, 15.0])
    m = forecast.forecast_metrics(pred, target)

    err = target - pred
    assert m["rmse"] == pytest.approx(float(np.sqrt((err**2).mean())))
    assert m["mae"] == pytest.approx(float(err.abs().mean()))
    ss_tot = float(((target - target.mean()) ** 2).sum())
    assert m["r2_oos"] == pytest.approx(1 - float((err**2).sum()) / ss_tot)
    assert m["vies_pct"] == pytest.approx((pred.mean() / target.mean() - 1) * 100)


def test_vies_pct_detecta_erro_de_nivel_que_o_r2_nao_ve():
    """O achado que motivou a metrica: um viés multiplicativo de -7% mal move
    o R2, mas desloca a previsao inteira para um lado."""
    rng = np.random.default_rng(5)
    target = pd.Series(12 + rng.normal(0, 2, 500))
    pred_neutra = target + rng.normal(0, 2, 500)
    pred_enviesada = pred_neutra * 0.93

    m_neutra = forecast.forecast_metrics(pred_neutra, target)
    m_enviesada = forecast.forecast_metrics(pred_enviesada, target)

    assert abs(m_neutra["vies_pct"]) < 1.0
    assert m_enviesada["vies_pct"] < -5.0
    # o R2 cai pouco perto do tamanho do deslocamento de nivel
    assert m_neutra["r2_oos"] - m_enviesada["r2_oos"] < 0.25


def test_evaluate_e_evaluate_persistence_usam_a_mesma_aritmetica():
    ds = _dataset_sintetico()
    train, test = ds.iloc[:300], ds.iloc[300:]
    model = forecast.fit_har(train, forecast.BASELINE_FEATURES, log_target=True)

    m = forecast.evaluate(model, test, forecast.BASELINE_FEATURES, log_target=True)
    pred = forecast.predict(model, test, forecast.BASELINE_FEATURES, log_target=True)
    assert m == forecast.forecast_metrics(pred, test["target"])

    mp = forecast.evaluate_persistence(test)
    assert mp == forecast.forecast_metrics(forecast.persistence_forecast(test), test["target"])