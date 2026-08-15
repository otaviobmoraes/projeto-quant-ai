"""Testes do modelo global em painel (vol/panel.py, backtest/panel_ablation.py)
e do coletor data/em_fx.py.

O teste mais importante e `test_purga_corta_todas_as_moedas`: o vazamento que
este desenho poderia ter e sutil -- purgar so a moeda alvo deixaria as moedas
vizinhas com alvo sobreposto ao bloco de teste, carregando informacao do mesmo
regime global de vol que se quer prever.
"""

import numpy as np
import pandas as pd
import pytest

from backtest import panel_ablation
from backtest.metrics import diebold_mariano
from data import em_fx
from vol import panel


def _ohlc(n: int, seed: int, base: float = 5.0, vol: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    close = base * np.exp(np.cumsum(rng.normal(0, vol, n)))
    amplitude = np.abs(rng.normal(0, vol, n)) + 1e-4
    return pd.DataFrame(
        {
            "date": idx,
            "open": close,
            "high": close * (1 + amplitude),
            "low": close * (1 - amplitude),
            "close": close,
        }
    )


def _painel_sintetico(moedas: dict[str, int], n: int = 600) -> pd.DataFrame:
    blocos = []
    for code, seed in moedas.items():
        b = _ohlc(n, seed)
        b["currency"] = code
        blocos.append(b)
    return pd.concat(blocos, ignore_index=True)


# ---------------------------------------------------------------------------
# Filtro de sanidade do coletor
# ---------------------------------------------------------------------------


def test_filtro_mascara_tick_sujo_e_preserva_o_resto():
    df = _ohlc(20, seed=1)
    df["currency"] = "xxx"
    df.loc[5, "low"] = df.loc[5, "low"] / 100  # erro de casa decimal, como o COP real

    limpo, removidos = em_fx.apply_sanity_filter(df)

    assert len(removidos) == 1
    assert removidos.iloc[0]["log_range"] > em_fx.MAX_LOG_RANGE
    assert pd.isna(limpo.loc[5, "high"]) and pd.isna(limpo.loc[5, "low"])
    # mascara em vez de apagar: o alinhamento por data e preservado
    assert len(limpo) == len(df)
    assert limpo.drop(index=5)[["high", "low"]].notna().all().all()


def test_filtro_nao_remove_dia_normal():
    df = _ohlc(50, seed=2)
    df["currency"] = "xxx"
    _, removidos = em_fx.apply_sanity_filter(df)
    assert removidos.empty


# ---------------------------------------------------------------------------
# Construcao do painel
# ---------------------------------------------------------------------------


def test_painel_empilha_todas_as_moedas():
    em = _painel_sintetico({"aaa": 1, "bbb": 2, "ccc": 3})
    p = panel.build_panel(em, horizon=21)

    assert set(p["currency"].unique()) == {"aaa", "bbb", "ccc"}
    assert set(panel.PANEL_COLUMNS).issubset(p.columns)
    assert p[["ratio_d", "ratio_w", "target_ratio"]].notna().all().all()
    assert (p["target_ratio"] > 0).all()


def test_features_do_painel_sao_adimensionais():
    """Duas moedas com o MESMO caminho relativo mas niveis de preco muito
    diferentes tem que produzir features identicas -- e o que permite
    empilhar."""
    barata = _ohlc(400, seed=7, base=5.0)
    cara = _ohlc(400, seed=7, base=4000.0)

    fa = panel.build_currency_frame(barata["high"], barata["low"], 21, "aaa")
    fb = panel.build_currency_frame(cara["high"], cara["low"], 21, "bbb")

    for col in ["ratio_d", "ratio_w", "target_ratio"]:
        np.testing.assert_allclose(fa[col].to_numpy(), fb[col].to_numpy(), rtol=1e-9)


def test_dia_sem_range_nao_contamina_o_painel():
    """REGRESSAO de bug real: um pregao com high == low zera a variancia de
    Parkinson. Em h=1 o alvo vira 0 e log(0) = -inf contaminava o modelo
    global INTEIRO com NaN. CLP tem 0,40% dos pregoes assim na amostra real.
    """
    df = _ohlc(300, seed=5)
    df.loc[100, "high"] = df.loc[100, "low"]  # pregao sem range

    frame = panel.build_currency_frame(df["high"], df["low"], horizon=1, currency="aaa")

    assert (frame["target_ratio"] > 0).all()
    assert np.isfinite(np.log(frame["target_ratio"])).all()
    m = panel.fit_global_model(frame)
    assert np.isfinite(m.params).all()


def test_purga_corta_todas_as_moedas():
    """O TESTE DE VAZAMENTO. A purga tem que remover as ultimas
    horizon+embargo datas de TODAS as moedas, nao so da moeda alvo."""
    em = _painel_sintetico({"aaa": 1, "bbb": 2}, n=400)
    p = panel.build_panel(em, horizon=21)

    datas = pd.DatetimeIndex(sorted(p["date"].unique()))
    train_end = datas[300]
    purgado = panel.purge_panel_by_date(p, train_end, horizon=21, embargo_days=5)

    # nenhuma moeda pode ter observacao dentro da janela purgada
    limite = datas[300 - 26]
    assert purgado["date"].max() <= limite
    for moeda in ("aaa", "bbb"):
        assert purgado[purgado["currency"] == moeda]["date"].max() <= limite
    assert set(purgado["currency"].unique()) == {"aaa", "bbb"}


def test_purga_devolve_vazio_quando_nao_ha_treino_suficiente():
    em = _painel_sintetico({"aaa": 1}, n=200)
    p = panel.build_panel(em, horizon=21)
    datas = pd.DatetimeIndex(sorted(p["date"].unique()))
    assert panel.purge_panel_by_date(p, datas[3], horizon=21, embargo_days=5).empty


# ---------------------------------------------------------------------------
# Modelo global
# ---------------------------------------------------------------------------


def test_modelo_global_recupera_relacao_comum_as_moedas():
    """Se todas as moedas seguem a MESMA relacao, o modelo global tem que
    recupera-la -- e com mais precisao que um ajuste numa moeda so."""
    rng = np.random.default_rng(0)
    blocos = []
    for code in ("aaa", "bbb", "ccc", "ddd"):
        n = 300
        d = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=n, freq="B"),
                "currency": code,
                "ratio_d": rng.uniform(0.5, 1.5, n),
                "ratio_w": rng.uniform(0.8, 1.2, n),
                "rv_trailing": 12.0,
            }
        )
        ruido = rng.normal(0, 0.05, n)
        d["target_ratio"] = np.exp(0.25 * d["ratio_d"] - 0.15 * d["ratio_w"] + ruido)
        d["target"] = d["target_ratio"] * d["rv_trailing"]
        blocos.append(d)
    pool = pd.concat(blocos, ignore_index=True)

    m = panel.fit_global_model(pool)
    assert m.params["ratio_d"] == pytest.approx(0.25, abs=0.03)
    assert m.params["ratio_w"] == pytest.approx(-0.15, abs=0.03)


def test_previsao_global_usa_o_nivel_local():
    """O painel entrega a FORMA da reversao; o nivel vem sempre da RV corrente
    do proprio ativo alvo."""
    rng = np.random.default_rng(4)
    n = 200
    pool = pd.DataFrame(
        {
            "ratio_d": rng.uniform(0.5, 1.5, n),
            "ratio_w": rng.uniform(0.8, 1.2, n),
            "target_ratio": rng.uniform(0.8, 1.2, n),
        }
    )
    m = panel.fit_global_model(pool)

    local = pd.DataFrame(
        {"ratio_d": [1.0, 1.0], "ratio_w": [1.0, 1.0], "rv_trailing": [10.0, 40.0]}
    )
    pred = panel.predict_with_global_model(m, local)

    # mesmas features, nivel 4x maior => previsao 4x maior
    assert pred.iloc[1] / pred.iloc[0] == pytest.approx(4.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------


def test_dm_detecta_modelo_melhor():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    y = pd.Series(rng.normal(10, 2, 500), index=idx)
    bom = y + rng.normal(0, 0.5, 500)
    ruim = y + rng.normal(0, 2.0, 500)

    r = diebold_mariano(y, bom, ruim)
    assert r["t_stat"] > 0  # A (bom) erra menos
    assert r["p_value"] < 0.01

    invertido = diebold_mariano(y, ruim, bom)
    assert invertido["t_stat"] < 0


def test_dm_janelas_independentes_reduzem_o_n():
    rng = np.random.default_rng(9)
    idx = pd.date_range("2020-01-01", periods=420, freq="B")
    y = pd.Series(rng.normal(10, 2, 420), index=idx)
    a = y + rng.normal(0, 1, 420)
    b = y + rng.normal(0, 1, 420)

    completo = diebold_mariano(y, a, b, horizon=21, independent_only=False)
    indep = diebold_mariano(y, a, b, horizon=21, independent_only=True)
    assert indep["n"] == pytest.approx(completo["n"] / 21, abs=1)


def test_dm_empate_perfeito():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    y = pd.Series(np.arange(50.0), index=idx)
    r = diebold_mariano(y, y + 1.0, y - 1.0)
    assert r["t_stat"] == 0.0
    assert r["p_value"] == 1.0


# ---------------------------------------------------------------------------
# Ablacao ponta a ponta
# ---------------------------------------------------------------------------


def test_ablacao_roda_e_avalia_no_mesmo_alvo():
    rng = np.random.default_rng(21)
    n = 900
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    var_brl = pd.Series(np.exp(rng.normal(-10, 0.5, n)), index=idx)
    em = _painel_sintetico({"aaa": 1, "bbb": 2, "ccc": 3}, n=n)

    res = panel_ablation.run_panel_ablation(var_brl, em, horizon=21, n_splits=5, embargo_days=5)

    assert set(res["pooled"]) == {"local_nivel", "local_razao", "global"}
    assert res["n_folds"] == len(res["por_fold"])
    assert 0 <= res["folds_melhores"] <= res["n_folds"]
    assert isinstance(res["passou_tres_portoes"], bool)
    # o painel de treino tem que ser bem maior que o treino local
    for f in res["por_fold"]:
        assert f["n_treino_painel"] > f["n_treino_local"]
    assert "VEREDITO" in panel_ablation.format_verdict(res)


def test_ablacao_sem_brl_no_painel_e_transferencia_pura():
    rng = np.random.default_rng(22)
    n = 900
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    var_brl = pd.Series(np.exp(rng.normal(-10, 0.5, n)), index=idx)
    em = _painel_sintetico({"aaa": 1, "bbb": 2}, n=n)

    com = panel_ablation.run_panel_ablation(var_brl, em, include_brl_in_panel=True)
    sem = panel_ablation.run_panel_ablation(var_brl, em, include_brl_in_panel=False)

    assert com["por_fold"][0]["n_treino_painel"] > sem["por_fold"][0]["n_treino_painel"]
    assert sem["painel_inclui_brl"] is False