"""Testes de vol/iv_trades.py (IV de negocios -> serie ATM -> avaliacao)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol.iv_trades import (
    daily_atm_iv,
    evaluate_forecasts,
    filter_quality,
    independent_windows,
    pair_with_forward_rv,
    variance_risk_premium,
)


def _trades_com_iv(**over) -> pd.DataFrame:
    base = {
        "iv_pct": [12.0, 15.0, 200.0, 11.0, 13.0],
        "trades": [3.0, 5.0, 4.0, 1.0, 2.0],
        "days_to_expiry": [30, 45, 30, 30, 5],
        "moneyness": [0.0, 0.01, 0.0, 0.0, 0.0],
    }
    base.update(over)
    return pd.DataFrame(base)


class TestFiltroDeQualidade:
    def test_descarta_negocio_unico(self):
        out = filter_quality(_trades_com_iv())
        assert 11.0 not in out["iv_pct"].to_numpy()  # trades=1

    def test_descarta_iv_fora_de_faixa(self):
        out = filter_quality(_trades_com_iv())
        assert 200.0 not in out["iv_pct"].to_numpy()

    def test_descarta_vencimento_curto_demais(self):
        out = filter_quality(_trades_com_iv())
        assert 13.0 not in out["iv_pct"].to_numpy()  # dte=5

    def test_mantem_os_validos(self):
        out = filter_quality(_trades_com_iv())
        assert sorted(out["iv_pct"].to_numpy()) == [12.0, 15.0]

    def test_descarta_iv_nan(self):
        df = _trades_com_iv(iv_pct=[np.nan, 15.0, 200.0, 11.0, 13.0])
        assert np.nan not in filter_quality(df)["iv_pct"].to_numpy()


class TestPareamento:
    def _variancia(self, n=200, seed=3) -> pd.Series:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2018-01-02", periods=n, freq="B", tz="America/Sao_Paulo")
        return pd.Series(np.exp(rng.normal(-9, 0.4, size=n)), index=idx)

    def test_alinha_apesar_de_fusos_diferentes(self):
        """A IV vem sem fuso e a variancia com fuso -- o pareamento tem de
        funcionar mesmo assim (fonte silenciosa de desalinhamento)."""
        var = self._variancia()
        iv = pd.Series(
            12.0, index=pd.DatetimeIndex([d.date() for d in var.index[50:120]]), name="iv_pct"
        )
        out = pair_with_forward_rv(iv, var)
        assert len(out) > 0
        assert set(out.columns) == {"iv", "target", "persist"}

    def test_target_e_rv_futura_e_persist_e_passada(self):
        var = self._variancia()
        iv = pd.Series(
            12.0, index=pd.DatetimeIndex([d.date() for d in var.index[50:120]]), name="iv_pct"
        )
        out = pair_with_forward_rv(iv, var, horizon=21)
        # ambos positivos e em pontos percentuais de vol anualizada
        assert (out["target"] > 0).all() and (out["persist"] > 0).all()
        assert out["target"].between(1, 200).all()

    def test_sem_sobreposicao_de_datas_devolve_vazio(self):
        var = self._variancia()
        iv = pd.Series(12.0, index=pd.DatetimeIndex(["2030-01-01"]), name="iv_pct")
        assert pair_with_forward_rv(iv, var).empty


class TestAvaliacao:
    def _pareado(self, n=210, seed=5) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2018-01-02", periods=n, freq="B")
        alvo = pd.Series(rng.normal(14, 3, size=n), index=idx).clip(4, 40)
        return pd.DataFrame(
            {
                # IV = alvo + ruido pequeno -> deve prever bem
                "iv": alvo + rng.normal(0, 0.5, size=n),
                "target": alvo,
                # persistencia = ruido puro -> deve prever mal
                "persist": pd.Series(rng.normal(14, 3, size=n), index=idx),
            }
        )

    def test_devolve_as_duas_versoes(self):
        out = evaluate_forecasts(self._pareado())
        assert set(out) == {"sobreposto", "independente"}
        assert out["independente"]["n"] < out["sobreposto"]["n"]

    def test_janelas_independentes_espacadas_por_data(self):
        """Em serie de dias consecutivos, espacar por data ~ espacar por
        posicao -- mas o criterio e a DATA."""
        pareado = self._pareado(n=210)
        sel = independent_windows(pareado, horizon=21)
        gaps = pd.Series(sel.index).diff().dropna().dt.days
        assert (gaps >= 29).all()  # 21 dias uteis ~ 29 corridos

    def test_serie_esparsa_nao_e_dizimada(self):
        """O caso que o iloc[::21] estragava: linhas ja espacadas de ~21 dias
        uteis devem ser TODAS aproveitadas, nao 1 a cada 21."""
        idx = pd.date_range("2018-01-02", periods=40, freq="30D")
        esparso = pd.DataFrame(
            {"iv": 12.0, "target": 13.0, "persist": 11.0}, index=idx
        )
        sel = independent_windows(esparso, horizon=21)
        assert len(sel) == len(esparso)

    def test_serie_mista_densa_e_esparsa(self):
        densa = pd.date_range("2018-01-01", periods=60, freq="D")
        esparsa = pd.date_range("2019-01-01", periods=10, freq="30D")
        idx = densa.append(esparsa)
        misto = pd.DataFrame({"iv": 12.0, "target": 13.0, "persist": 11.0}, index=idx)
        sel = independent_windows(misto, horizon=21)
        # do bloco denso sobra ~2; do esparso, todos os 10
        assert 10 <= len(sel) <= 14
        gaps = pd.Series(sel.index).diff().dropna().dt.days
        assert (gaps >= 29).all()

    def test_previsor_bom_ganha_do_ruim(self):
        out = evaluate_forecasts(self._pareado())["sobreposto"]
        assert out["iv"]["r2_oos"] > out["persistencia"]["r2_oos"]
        assert out["iv"]["rmse"] < out["persistencia"]["rmse"]

    def test_correlacao_reportada(self):
        out = evaluate_forecasts(self._pareado())["sobreposto"]
        assert 0.9 < out["corr_iv_rv"] <= 1.0


class TestPremioDeRisco:
    def test_calcula_as_duas_razoes(self):
        df = pd.DataFrame({"iv": [12.0, 14.0], "target": [10.0, 10.0], "persist": [8.0, 8.0]})
        out = variance_risk_premium(df)
        assert out["iv_over_forward"] == pytest.approx(1.3)
        assert out["iv_over_trailing"] == pytest.approx(1.625)
        assert out["share_iv_acima"] == 1.0

    def test_share_quando_iv_fica_abaixo(self):
        df = pd.DataFrame({"iv": [8.0, 14.0], "target": [10.0, 10.0], "persist": [8.0, 8.0]})
        assert variance_risk_premium(df)["share_iv_acima"] == pytest.approx(0.5)


class TestPipelineCompleto:
    def test_dos_negocios_ate_a_serie_diaria(self):
        """Caminho completo com dados sinteticos: negocios -> IV ATM diaria."""
        from vol.black76 import call_price

        F, K, sigma, dias = 4075.666, 4075.0, 0.15, 30
        preco = call_price(F, K, dias / 365, sigma)
        datas = pd.to_datetime(["2018-03-01", "2018-03-02"]).tz_localize("America/Sao_Paulo")
        trades = pd.DataFrame(
            {
                "date": datas,
                "ticker": ["DOLJ18C004075"] * 2,
                "maturity": ["J18"] * 2,
                "option_type": ["C"] * 2,
                "strike": [K] * 2,
                "price": [preco] * 2,
                "trades": [4.0, 4.0],
                "open_interest": [100.0, 100.0],
                "future_settlement": [F, F],
            }
        )
        serie = daily_atm_iv(trades)
        assert len(serie) == 2
        # J18 vence em 2018-04-02; de 01/03 sao 32 dias, dentro de DTE_RANGE
        assert serie.iloc[0] == pytest.approx(sigma * 100, rel=0.05)

    def test_sem_negocio_utilizavel_devolve_serie_vazia(self):
        trades = pd.DataFrame(
            {
                "date": pd.to_datetime(["2018-03-01"]).tz_localize("America/Sao_Paulo"),
                "ticker": ["DOLJ18C004075"],
                "maturity": ["J18"],
                "option_type": ["C"],
                "strike": [4075.0],
                "price": [0.0],  # preco invalido -> nao inverte
                "trades": [4.0],
                "open_interest": [100.0],
                "future_settlement": [4075.666],
            }
        )
        assert daily_atm_iv(trades).empty
