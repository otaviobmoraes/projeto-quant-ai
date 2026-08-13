"""Testes de vol/roll.py (correcao do ciclo de rolagem do futuro de dolar)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from vol.roll import (
    DEFAULT_BUCKET_EDGES,
    bucket_days_to_expiry,
    contract_expiry,
    days_to_expiry,
    deseasonalize,
    seasonal_factor,
)


class TestContractExpiry:
    def test_mapeia_codigo_de_mes(self):
        # K = maio; 1/5/2023 e segunda-feira (dia util)
        assert contract_expiry("DOLK23") == date(2023, 5, 1)
        # M = junho; 1/6/2023 e quinta-feira
        assert contract_expiry("DOLM23") == date(2023, 6, 1)
        # Z = dezembro
        assert contract_expiry("DOLZ24") == date(2024, 12, 2)  # 1/12/2024 = domingo

    def test_pula_fim_de_semana(self):
        # 1/6/2025 e domingo -> primeiro dia util e 2/6
        assert contract_expiry("DOLM25") == date(2025, 6, 2)
        # 1/2/2025 e sabado -> 3/2 (segunda)
        assert contract_expiry("DOLG25") == date(2025, 2, 3)

    def test_ticker_invalido(self):
        with pytest.raises(ValueError):
            contract_expiry("DOLA23")  # 'A' nao e codigo de mes
        with pytest.raises(ValueError):
            contract_expiry("DOL")


class TestDaysToExpiry:
    def test_conta_dias_corridos_ate_o_vencimento(self):
        dates = pd.Series(pd.to_datetime(["2023-04-10", "2023-04-27"]))
        tickers = pd.Series(["DOLK23", "DOLK23"])  # vence 2023-05-01
        dte = days_to_expiry(dates, tickers)
        assert dte.tolist() == [21, 4]

    def test_aceita_indice_tz_aware(self):
        dates = pd.Series(pd.to_datetime(["2023-04-10"]).tz_localize("America/Sao_Paulo"))
        dte = days_to_expiry(dates, pd.Series(["DOLK23"]))
        assert dte.iloc[0] == 21


class TestBuckets:
    def test_faixas_sao_monotonicas_em_dte(self):
        dte = pd.Series([0, 6, 7, 13, 14, 20, 21, 27, 28, 35])
        buckets = bucket_days_to_expiry(dte)
        assert buckets.is_monotonic_increasing
        assert buckets.nunique() == len(DEFAULT_BUCKET_EDGES)


class TestSeasonalFactor:
    def test_serie_sem_sazonalidade_da_fator_unitario(self):
        rng = np.random.default_rng(0)
        var = pd.Series(np.exp(rng.normal(size=400)))
        dte = pd.Series(np.tile(np.arange(0, 40, 2), 20))
        factor = seasonal_factor(var, dte)
        assert np.allclose(factor.to_numpy(), 1.0, atol=0.35)

    def test_recupera_sazonalidade_conhecida(self):
        # variancia constante multiplicada por 2 na faixa de contrato "novo"
        dte = pd.Series(np.tile(np.arange(0, 35), 20))
        var = pd.Series(np.where(dte >= 28, 2.0, 1.0), dtype=float)
        factor = seasonal_factor(var, dte)
        # a ultima faixa deve ter fator ~2x a primeira
        assert factor.iloc[-1] / factor.iloc[0] == pytest.approx(2.0, rel=1e-6)

    def test_media_geometrica_normalizada_em_um(self):
        rng = np.random.default_rng(1)
        var = pd.Series(np.exp(rng.normal(size=500)))
        dte = pd.Series(rng.integers(0, 35, size=500))
        factor = seasonal_factor(var, dte)
        # media ponderada em log deve ficar proxima de zero (fator ~1)
        assert abs(float(np.log(factor).mean())) < 0.5


class TestDeseasonalize:
    def test_remove_a_sazonalidade_injetada(self):
        dte = pd.Series(np.tile(np.arange(0, 35), 20))
        base = pd.Series(np.full(len(dte), 1.0))
        var = base * np.where(dte >= 28, 2.0, 1.0)
        factor = seasonal_factor(var, dte)
        adj = deseasonalize(var, dte, factor)
        # apos corrigir, as duas faixas ficam no mesmo nivel
        alto = adj[dte >= 28].mean()
        baixo = adj[dte < 28].mean()
        assert alto == pytest.approx(baixo, rel=1e-6)

    def test_faixa_ausente_no_fator_nao_corrige(self):
        var = pd.Series([1.0, 2.0, 3.0])
        dte = pd.Series([1, 10, 30])
        factor = pd.Series([2.0], index=[1])  # so a faixa 1 foi estimada
        adj = deseasonalize(var, dte, factor)
        assert adj.iloc[0] == pytest.approx(0.5)
        assert adj.iloc[1] == pytest.approx(2.0)  # fator 1.0 -> inalterado
        assert adj.iloc[2] == pytest.approx(3.0)

    def test_preserva_indice(self):
        idx = pd.date_range("2024-01-01", periods=3)
        var = pd.Series([1.0, 2.0, 3.0], index=idx)
        dte = pd.Series([1, 10, 30], index=idx)
        adj = deseasonalize(var, dte, seasonal_factor(var, dte))
        assert adj.index.equals(idx)
