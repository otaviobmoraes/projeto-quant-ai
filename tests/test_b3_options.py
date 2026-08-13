"""Testes de data/b3_options.py (negocios de opcao de dolar + IV invertida).

Os fixtures montam um pacote BVBG-086 sintetico (zip dentro de zip, 4 XMLs
progressivos) para testar o parser sem tocar a rede.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.b3_options import add_implied_vols, atm_iv_by_date, parse_dol_options
from vol.black76 import call_price


def _pric_rpt(ticker: str, **campos) -> str:
    corpo = "".join(f"<{k}>{v}</{k}>" for k, v in campos.items())
    return f"<PricRpt><SctyId><TckrSymb>{ticker}</TckrSymb></SctyId>{corpo}</PricRpt>"


def _pacote(xml: str) -> bytes:
    """Empacota como a B3 faz: zip externo -> zip interno -> 4 XMLs, sendo o
    ULTIMO (ordem alfabetica) o snapshot completo."""
    interno = io.BytesIO()
    with zipfile.ZipFile(interno, "w") as z:
        for i in range(3):
            z.writestr(f"BVBG.086.01_{i}.xml", "<Doc></Doc>")
        z.writestr("BVBG.086.01_9.xml", xml)
    externo = io.BytesIO()
    with zipfile.ZipFile(externo, "w") as z:
        z.writestr("interno.zip", interno.getvalue())
    return externo.getvalue()


@pytest.fixture
def pacote_simples() -> bytes:
    xml = (
        "<Doc>"
        + _pric_rpt("DOLU19", AdjstdQt="4075.666", MaxPric="4090.0", MinPric="4060.0")
        + _pric_rpt("DOLV19", AdjstdQt="4120.500")
        # negociada
        + _pric_rpt("DOLU19C004050", TradAvrgPric="42.10", TradQty="4", OpnIntrst="1500")
        # negociada, so com FrstPric (sem preco medio)
        + _pric_rpt("DOLU19P004050", FrstPric="54.50", TradQty="1")
        # NAO negociada -- deve ser descartada
        + _pric_rpt("DOLU19C004200", OpnIntrst="800")
        # vencimento sem futuro com ajuste -> future_settlement NaN
        + _pric_rpt("DOLZ19C004300", TradAvrgPric="80.00", TradQty="2")
        + "</Doc>"
    )
    return _pacote(xml)


class TestParse:
    def test_so_series_negociadas_entram(self, pacote_simples):
        df = parse_dol_options(pacote_simples, date(2019, 8, 14))
        assert set(df["ticker"]) == {"DOLU19C004050", "DOLU19P004050", "DOLZ19C004300"}

    def test_campos_decompostos_do_ticker(self, pacote_simples):
        df = parse_dol_options(pacote_simples, date(2019, 8, 14)).set_index("ticker")
        linha = df.loc["DOLU19C004050"]
        assert linha["maturity"] == "U19"
        assert linha["option_type"] == "C"
        assert linha["strike"] == 4050.0
        assert linha["price"] == 42.10
        assert linha["trades"] == 4.0
        assert linha["open_interest"] == 1500.0

    def test_anexa_ajuste_do_futuro_do_mesmo_vencimento(self, pacote_simples):
        df = parse_dol_options(pacote_simples, date(2019, 8, 14)).set_index("ticker")
        assert df.loc["DOLU19C004050", "future_settlement"] == 4075.666
        assert df.loc["DOLU19P004050", "future_settlement"] == 4075.666
        # Z19 nao tem futuro com ajuste no pacote
        assert pd.isna(df.loc["DOLZ19C004300", "future_settlement"])

    def test_usa_frstpric_quando_nao_ha_preco_medio(self, pacote_simples):
        df = parse_dol_options(pacote_simples, date(2019, 8, 14)).set_index("ticker")
        assert df.loc["DOLU19P004050", "price"] == 54.50

    def test_pacote_sem_opcao_devolve_vazio_com_colunas(self):
        pacote = _pacote("<Doc>" + _pric_rpt("DOLU19", AdjstdQt="4075.0") + "</Doc>")
        df = parse_dol_options(pacote, date(2019, 8, 14))
        assert df.empty
        assert "future_settlement" in df.columns

    def test_le_o_ultimo_snapshot(self):
        """Os 3 primeiros XMLs sao vazios; o parser tem de ler o 4o."""
        xml = "<Doc>" + _pric_rpt("DOLU19C004050", TradAvrgPric="42.10", TradQty="1") + "</Doc>"
        assert len(parse_dol_options(_pacote(xml), date(2019, 8, 14))) == 1


class TestImpliedVols:
    def test_inverte_e_devolve_em_pontos_percentuais(self, pacote_simples):
        df = add_implied_vols(parse_dol_options(pacote_simples, date(2019, 8, 14)))
        linha = df.set_index("ticker").loc["DOLU19C004050"]
        assert 5.0 < linha["iv_pct"] < 35.0  # faixa plausivel para USD/BRL

    def test_roundtrip_exato(self):
        """Preco gerado por Black-76 com sigma conhecida volta como iv_pct."""
        F, K, sigma = 4075.666, 4050.0, 0.1234
        dias = (date(2019, 9, 2) - date(2019, 8, 14)).days
        preco = call_price(F, K, dias / 365, sigma)
        xml = (
            "<Doc>"
            + _pric_rpt("DOLU19", AdjstdQt=str(F))
            + _pric_rpt("DOLU19C004050", TradAvrgPric=f"{preco:.6f}", TradQty="3")
            + "</Doc>"
        )
        df = add_implied_vols(parse_dol_options(_pacote(xml), date(2019, 8, 14)))
        assert df["iv_pct"].iloc[0] == pytest.approx(sigma * 100, rel=1e-5)

    def test_sem_futuro_nao_inverte(self, pacote_simples):
        df = add_implied_vols(parse_dol_options(pacote_simples, date(2019, 8, 14)))
        assert pd.isna(df.set_index("ticker").loc["DOLZ19C004300", "iv_pct"])

    def test_descarta_fora_da_faixa_de_moneyness(self, pacote_simples):
        df = add_implied_vols(
            parse_dol_options(pacote_simples, date(2019, 8, 14)), max_moneyness=0.001
        )
        assert df["iv_pct"].isna().all()

    def test_dataframe_vazio_nao_quebra(self):
        vazio = parse_dol_options(_pacote("<Doc></Doc>"), date(2019, 8, 14))
        out = add_implied_vols(vazio)
        assert out.empty


class TestAtmAggregation:
    def _com_iv(self) -> pd.DataFrame:
        base = pd.Timestamp("2019-08-14", tz="America/Sao_Paulo")
        return pd.DataFrame(
            {
                "date": [base] * 4,
                "maturity": ["U19", "U19", "U19", "V19"],
                "iv_pct": [10.0, 20.0, 99.0, 15.0],
                "trades": [1.0, 3.0, 5.0, 2.0],
                "moneyness": [0.005, -0.005, 0.50, 0.01],  # o 3o esta longe do ATM
            }
        )

    def test_pondera_por_numero_de_negocios(self):
        out = atm_iv_by_date(self._com_iv(), max_moneyness=0.02).set_index("maturity")
        # U19: (10*1 + 20*3)/4 = 17.5 -- a serie de moneyness 0.50 fica fora
        assert out.loc["U19", "iv_pct"] == pytest.approx(17.5)
        assert out.loc["U19", "n_trades"] == 4.0
        assert out.loc["U19", "n_series"] == 2.0

    def test_separa_por_vencimento(self):
        out = atm_iv_by_date(self._com_iv(), max_moneyness=0.02)
        assert set(out["maturity"]) == {"U19", "V19"}

    def test_filtro_de_negocios_minimos(self):
        out = atm_iv_by_date(self._com_iv(), max_moneyness=0.02, min_trades=3)
        assert out.set_index("maturity").loc["U19", "iv_pct"] == pytest.approx(20.0)

    def test_sem_dado_utilizavel_devolve_vazio_com_colunas(self):
        df = self._com_iv()
        df["iv_pct"] = np.nan
        out = atm_iv_by_date(df)
        assert out.empty
        assert list(out.columns) == ["date", "maturity", "iv_pct", "n_trades", "n_series"]
