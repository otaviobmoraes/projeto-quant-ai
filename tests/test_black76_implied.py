"""Testes do inversor de vol implicita (Black-76) -- vol/black76.implied_vol."""

from __future__ import annotations

import pytest

from vol.black76 import call_price, implied_vol, put_price


class TestRoundTrip:
    """A prova mais forte do inversor: apreçar com uma sigma conhecida e
    recuperar exatamente essa sigma a partir do preco.

    Os strikes usados aqui (4000 a 4150 contra futuro em 4075,666) sao os
    EFETIVAMENTE NEGOCIADOS no pregao de 2019-08-14 -- +-2% do dinheiro. Nao
    e uma faixa escolhida para o teste passar: e onde o mercado de opcao de
    dolar da B3 realmente opera, e tambem onde a inversao e bem-posta. Fora
    dela, com prazo curto e vol baixa, o premio vira intrinseco puro e
    nenhuma sigma e recuperavel -- ver TestMalPosto.
    """

    @pytest.mark.parametrize("sigma", [0.05, 0.0916, 0.15, 0.30, 0.75])
    @pytest.mark.parametrize("K", [4000.0, 4050.0, 4075.0, 4100.0, 4150.0])
    def test_call(self, sigma, K):
        F, T = 4075.666, 19 / 365
        preco = call_price(F, K, T, sigma)
        recuperada = implied_vol(preco, F, K, T, "C")
        assert recuperada == pytest.approx(sigma, rel=1e-6)

    @pytest.mark.parametrize("sigma", [0.05, 0.0916, 0.15, 0.30, 0.75])
    @pytest.mark.parametrize("K", [4000.0, 4050.0, 4075.0, 4100.0, 4150.0])
    def test_put(self, sigma, K):
        F, T = 4075.666, 19 / 365
        preco = put_price(F, K, T, sigma)
        recuperada = implied_vol(preco, F, K, T, "P")
        assert recuperada == pytest.approx(sigma, rel=1e-6)

    def test_com_desconto_nao_nulo(self):
        F, K, T, sigma, r = 4075.0, 4050.0, 0.25, 0.12, 0.105
        preco = call_price(F, K, T, sigma, r)
        assert implied_vol(preco, F, K, T, "C", r) == pytest.approx(sigma, rel=1e-6)

    def test_prazo_muito_curto_ainda_inverte(self):
        """Perto do vencimento o vega vai a zero -- e onde Newton quebraria."""
        F, K, T, sigma = 4075.0, 4075.0, 1 / 365, 0.20
        preco = call_price(F, K, T, sigma)
        assert implied_vol(preco, F, K, T, "C") == pytest.approx(sigma, rel=1e-5)


class TestMalPosto:
    """Fundo dentro do dinheiro o premio e intrinseco puro ate a precisao do
    float: nenhuma sigma explica o preco melhor que outra. O inversor devolve
    None em vez de inventar precisao -- e isso define o filtro de moneyness
    que a coleta de negocios reais precisa aplicar.
    """

    def test_put_fundo_dentro_do_dinheiro_devolve_none(self):
        F, K, T, sigma = 4075.666, 4400.0, 19 / 365, 0.05
        preco = put_price(F, K, T, sigma)
        assert preco == pytest.approx(K - F, abs=1e-6)  # intrinseco puro
        assert implied_vol(preco, F, K, T, "P") is None

    def test_call_fundo_dentro_do_dinheiro_devolve_none(self):
        F, K, T, sigma = 4075.666, 3000.0, 19 / 365, 0.05
        preco = call_price(F, K, T, sigma)
        assert implied_vol(preco, F, K, T, "C") is None

    def test_a_mesma_opcao_com_vol_alta_ja_e_recuperavel(self):
        """Confirma que o corte e sobre VALOR NO TEMPO, nao sobre o strike:
        o mesmo strike fundo ITM, com vol alta, volta a ter valor no tempo
        suficiente e inverte normalmente."""
        F, K, T, sigma = 4075.666, 4400.0, 19 / 365, 0.60
        preco = put_price(F, K, T, sigma)
        assert implied_vol(preco, F, K, T, "P") == pytest.approx(sigma, rel=1e-6)


class TestRejeicoes:
    """Casos sem solucao devolvem None em vez de levantar -- numa serie de
    negocios esparsos, parte dos registros e sempre inutilizavel."""

    def test_preco_abaixo_do_intrinseco(self):
        # call com F-K = 100, negociada a 50: arbitragem estatica
        assert implied_vol(50.0, 4150.0, 4050.0, 0.05, "C") is None

    def test_preco_acima_do_teto(self):
        assert implied_vol(5000.0, 4075.0, 4050.0, 0.05, "C") is None
        assert implied_vol(5000.0, 4075.0, 4050.0, 0.05, "P") is None

    def test_entradas_degeneradas(self):
        assert implied_vol(10.0, 4075.0, 4050.0, 0.0, "C") is None   # T = 0
        assert implied_vol(0.0, 4075.0, 4050.0, 0.05, "C") is None   # preco 0
        assert implied_vol(-1.0, 4075.0, 4050.0, 0.05, "C") is None
        assert implied_vol(10.0, 0.0, 4050.0, 0.05, "C") is None     # F = 0

    def test_tipo_invalido(self):
        with pytest.raises(ValueError):
            implied_vol(42.0, 4075.0, 4050.0, 0.05, "X")


class TestConsistencia:
    def test_paridade_put_call_da_mesma_vol(self):
        """Call e put no mesmo strike, precificadas com a mesma sigma, tem de
        devolver a mesma vol implicita -- e o teste que aplicaremos aos
        negocios reais da B3 como filtro de qualidade."""
        F, K, T, sigma = 4075.666, 4050.0, 19 / 365, 0.113
        iv_c = implied_vol(call_price(F, K, T, sigma), F, K, T, "C")
        iv_p = implied_vol(put_price(F, K, T, sigma), F, K, T, "P")
        assert iv_c == pytest.approx(iv_p, rel=1e-6)

    def test_monotonicidade_no_preco(self):
        """Premio maior -> vol implicita maior (a funcao e crescente em sigma)."""
        F, K, T = 4075.0, 4075.0, 30 / 365
        ivs = [implied_vol(p, F, K, T, "C") for p in (20.0, 40.0, 60.0, 80.0)]
        assert all(a < b for a, b in zip(ivs, ivs[1:]))

    def test_nivel_plausivel_num_negocio_real_de_2019(self):
        """DOLU19C004050 negociada a 42,10 em 2019-08-14, futuro DOLU19 a
        4075,666, vencimento 2019-09-02 (19 dias corridos). A vol implicita
        tem de cair numa faixa plausivel para USD/BRL."""
        iv = implied_vol(42.10, 4075.666, 4050.0, 19 / 365, "C")
        assert iv is not None
        assert 0.05 < iv < 0.35, f"vol implicita implausivel: {iv:.4f}"
