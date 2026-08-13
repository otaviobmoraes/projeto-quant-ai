"""Testes dos estimadores de variancia adicionados a vol/realized.py:
Garman-Klass, Rogers-Satchell, overnight, full-day e Yang-Zhang.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol.realized import (
    ESTIMATORS,
    full_day_variance,
    garman_klass_daily_variance,
    overnight_variance,
    parkinson_daily_variance,
    rogers_satchell_daily_variance,
    yang_zhang_variance,
)


def _ohlc(n: int = 300, seed: int = 5, drift: float = 0.0) -> pd.DataFrame:
    """OHLC sintetico coerente: high >= max(open, close), low <= min(open, close)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, size=n))), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0]) * np.exp(rng.normal(0, 0.002, size=n))
    span = np.abs(rng.normal(0, 0.008, size=n))
    high = pd.Series(np.maximum(open_, close) * np.exp(span), index=idx)
    low = pd.Series(np.minimum(open_, close) * np.exp(-span), index=idx)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


class TestFormulas:
    def test_garman_klass_confere_com_a_formula(self):
        d = _ohlc(50)
        esperado = 0.5 * np.log(d["high"] / d["low"]) ** 2 - (2 * np.log(2) - 1) * np.log(
            d["close"] / d["open"]
        ) ** 2
        got = garman_klass_daily_variance(d["open"], d["high"], d["low"], d["close"])
        assert np.allclose(got.to_numpy(), esperado.to_numpy())

    def test_rogers_satchell_confere_com_a_formula(self):
        d = _ohlc(50)
        esperado = np.log(d["high"] / d["close"]) * np.log(d["high"] / d["open"]) + np.log(
            d["low"] / d["close"]
        ) * np.log(d["low"] / d["open"])
        got = rogers_satchell_daily_variance(d["open"], d["high"], d["low"], d["close"])
        assert np.allclose(got.to_numpy(), esperado.to_numpy())

    def test_rogers_satchell_e_nao_negativo(self):
        """Cada termo e produto de dois logs de mesmo sinal -- a soma nunca e < 0."""
        d = _ohlc(500, seed=11)
        rs = rogers_satchell_daily_variance(d["open"], d["high"], d["low"], d["close"])
        assert (rs >= -1e-15).all()

    def test_dia_sem_movimento_da_variancia_zero(self):
        um = pd.Series([100.0])
        for f in (garman_klass_daily_variance, rogers_satchell_daily_variance):
            assert f(um, um, um, um).iloc[0] == pytest.approx(0.0, abs=1e-12)


class TestToleranciaADrift:
    """A propriedade que motivou incluir Rogers-Satchell: Parkinson e
    Garman-Klass assumem drift zero e superestimam quando o preco tem
    tendencia -- o caso do USD/BRL na amostra."""

    def test_rs_sofre_menos_com_drift_forte(self):
        sem = _ohlc(4000, seed=3, drift=0.0)
        com = _ohlc(4000, seed=3, drift=0.004)  # tendencia forte, mesma semente

        def media(f, d):
            return f(d["open"], d["high"], d["low"], d["close"]).mean()

        vies_rs = media(rogers_satchell_daily_variance, com) / media(
            rogers_satchell_daily_variance, sem
        )
        pk_com = parkinson_daily_variance(com["high"], com["low"]).mean()
        pk_sem = parkinson_daily_variance(sem["high"], sem["low"]).mean()
        vies_pk = pk_com / pk_sem
        assert vies_rs < vies_pk, (
            f"RS deveria inflar menos que Parkinson (RS {vies_rs:.3f}, PK {vies_pk:.3f})"
        )


class TestOvernight:
    def test_calcula_o_gap(self):
        open_ = pd.Series([101.0, 102.0])
        prev = pd.Series([100.0, 101.0])
        got = overnight_variance(open_, prev)
        assert got.iloc[0] == pytest.approx(np.log(1.01) ** 2)

    def test_mascara_dia_de_rolagem(self):
        """Em dia de rolagem o gap cruza contratos diferentes -- nao e vol."""
        open_ = pd.Series([101.0, 102.0, 103.0])
        prev = pd.Series([100.0, 101.0, 102.0])
        changed = pd.Series([False, True, False])
        got = overnight_variance(open_, prev, changed)
        assert pd.isna(got.iloc[1])
        assert got.notna().sum() == 2


class TestFullDay:
    def test_soma_overnight_e_intradiario(self):
        d = _ohlc(50)
        fd = full_day_variance(d["open"], d["high"], d["low"], d["close"])
        rs = rogers_satchell_daily_variance(d["open"], d["high"], d["low"], d["close"])
        on = overnight_variance(d["open"], d["close"].shift(1)).fillna(0.0)
        assert np.allclose(fd.to_numpy(), (on + rs).to_numpy())

    def test_rolagem_zera_overnight_em_vez_de_gerar_nan(self):
        """Regressao: mascarar com NaN fazia rolling(22) virar NaN em quase toda
        a serie (rolagem a cada ~20 pregoes), eliminando o estimador antes do
        teste. Precisa cair para o valor intradiario, nao sumir."""
        d = _ohlc(100)
        changed = pd.Series(False, index=d.index)
        changed.iloc[::20] = True
        fd = full_day_variance(d["open"], d["high"], d["low"], d["close"], changed)
        assert fd.iloc[1:].notna().all(), "full_day nao pode ter NaN por causa de rolagem"
        rs = rogers_satchell_daily_variance(d["open"], d["high"], d["low"], d["close"])
        assert fd.iloc[20] == pytest.approx(rs.iloc[20])  # so a parte intradiaria


class TestYangZhang:
    def test_e_estimador_de_JANELA(self):
        d = _ohlc(200)
        yz = yang_zhang_variance(d["open"], d["high"], d["low"], d["close"], window=21)
        assert yz.iloc[:20].isna().all()  # precisa da janela cheia
        assert yz.iloc[25:].notna().all()

    def test_positivo_e_em_escala_de_variancia(self):
        d = _ohlc(300)
        yz = yang_zhang_variance(d["open"], d["high"], d["low"], d["close"]).dropna()
        assert (yz > 0).all()
        assert yz.mean() < 0.01  # variancia diaria, nao anualizada

    def test_janela_pequena_demais_levanta(self):
        d = _ohlc(50)
        with pytest.raises(ValueError):
            yang_zhang_variance(d["open"], d["high"], d["low"], d["close"], window=1)


class TestDespachante:
    def test_todos_os_estimadores_declarados_funcionam(self):
        from vol.realized import load_b3_variance

        for e in ESTIMATORS:
            precos, var = load_b3_variance(e)
            assert len(var) > 1000, f"{e} devolveu serie curta demais"
            assert var.dropna().gt(0).mean() > 0.95, f"{e} tem variancia nao-positiva demais"
            assert precos.index.equals(var.index)

    def test_estimador_desconhecido_levanta(self):
        from vol.realized import load_b3_variance

        with pytest.raises(ValueError, match="desconhecido"):
            load_b3_variance("nao_existe")
