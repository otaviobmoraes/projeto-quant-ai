"""Testes das metricas de desempenho de backtest (backtest/metrics.py):
drawdown maximo, retorno anualizado e o resumo completo."""

from __future__ import annotations

import numpy as np
import pytest

from backtest.metrics import annualized_return, max_drawdown, performance_summary


class TestMaxDrawdown:
    def test_sequencia_so_de_ganhos_nao_tem_rebaixamento(self):
        assert max_drawdown([0.01, 0.02, 0.03])["max_drawdown"] == pytest.approx(0.0)

    def test_queda_conhecida(self):
        # +100% e depois -50%: patrimonio 1 -> 2 -> 1, rebaixamento de 50%
        d = max_drawdown([1.0, -0.5])
        assert d["max_drawdown"] == pytest.approx(-0.5)
        assert d["vale"] == 1

    def test_localiza_pico_e_vale(self):
        d = max_drawdown([0.1, 0.1, -0.3, -0.2, 0.05])
        assert d["vale"] == 3
        assert d["pico"] == 1  # o topo antes da queda

    def test_serie_vazia_nao_quebra(self):
        assert np.isnan(max_drawdown([])["max_drawdown"])


class TestAnnualizedReturn:
    def test_e_geometrico_e_nao_aritmetico(self):
        """+50% seguido de -50% da media aritmetica ZERO mas retorno real
        de -25%. Reportar a media superestimaria o desempenho."""
        r = [0.5, -0.5]
        assert np.mean(r) == pytest.approx(0.0)
        anual = annualized_return(r, trades_per_year=2.0)  # 1 ano
        assert anual == pytest.approx(-0.25)

    def test_um_ano_exato(self):
        r = [0.1] * 12
        assert annualized_return(r, 12.0) == pytest.approx(1.1**12 - 1)

    def test_dois_anos_anualiza_pela_raiz(self):
        r = [0.0] * 12 + [0.0] * 12
        assert annualized_return(r, 12.0) == pytest.approx(0.0)

    def test_perda_total_devolve_nan(self):
        """Retorno de -100% zera o patrimonio; anualizar dali nao tem sentido."""
        assert np.isnan(annualized_return([-1.0, 0.2], 12.0))

    def test_vazio_devolve_nan(self):
        assert np.isnan(annualized_return([], 12.0))


class TestPerformanceSummary:
    def test_campos_esperados(self):
        rng = np.random.default_rng(1)
        s = performance_summary(rng.normal(0.01, 0.1, size=30), n_trials=10)
        for k in ("n_trades", "sharpe", "retorno_anualizado", "max_drawdown",
                  "win_rate", "psr_vs_zero", "deflated_sharpe", "n_trials_usado"):
            assert k in s

    def test_sem_trades(self):
        assert performance_summary([], n_trials=10) == {"n_trades": 0}

    def test_mais_configuracoes_testadas_reduz_o_dsr(self):
        """O ponto central do Deflated Sharpe: o mesmo desempenho vale menos
        quando muitas configuracoes foram testadas."""
        rng = np.random.default_rng(2)
        r = rng.normal(0.03, 0.1, size=60)
        poucas = performance_summary(r, n_trials=2)["deflated_sharpe"]
        muitas = performance_summary(r, n_trials=200)["deflated_sharpe"]
        assert muitas < poucas

    def test_dispersao_maior_entre_trials_reduz_o_dsr(self):
        rng = np.random.default_rng(3)
        r = rng.normal(0.03, 0.1, size=60)
        estreita = performance_summary(r, n_trials=20, sr_trials_std=0.1)["deflated_sharpe"]
        larga = performance_summary(r, n_trials=20, sr_trials_std=2.0)["deflated_sharpe"]
        assert larga < estreita

    def test_win_rate_e_coerente(self):
        s = performance_summary([0.1, -0.1, 0.2, -0.3], n_trials=5)
        assert s["win_rate"] == pytest.approx(0.5)
        assert s["n_trades"] == 4

    def test_dsr_indefinido_com_menos_de_duas_configuracoes(self):
        """Nao existe 'maximo esperado por acaso' sobre uma amostra de uma --
        o resumo devolve NaN em vez de estourar."""
        s = performance_summary([0.1, -0.1, 0.2], n_trials=1)
        assert np.isnan(s["deflated_sharpe"])
        assert not np.isnan(s["sharpe"])  # o resto continua utilizavel

    def test_estrategia_perdedora_tem_sharpe_negativo(self):
        rng = np.random.default_rng(4)
        s = performance_summary(rng.normal(-0.04, 0.1, size=50), n_trials=5)
        assert s["sharpe"] < 0
        assert s["retorno_anualizado"] < 0
