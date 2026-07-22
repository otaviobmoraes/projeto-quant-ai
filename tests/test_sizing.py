import pytest

from strategy import sizing


def test_straddle_vega_per_contract_positive():
    v = sizing.straddle_vega_per_contract(spot=5.10, ttm_days=21, iv_pct=9.0)
    assert v > 0


def test_size_straddle_scales_linearly_with_target_vega():
    n1 = sizing.size_straddle(target_vega=1000, spot=5.10, ttm_days=21, iv_pct=9.0)
    n2 = sizing.size_straddle(target_vega=2000, spot=5.10, ttm_days=21, iv_pct=9.0)

    assert n2 == pytest.approx(2 * n1)


def test_size_straddle_fewer_contracts_needed_for_longer_maturity():
    # Vega ATM cresce com sqrt(T), entao pro mesmo alvo de vega, um straddle
    # de vencimento mais longo precisa de MENOS contratos.
    n_short = sizing.size_straddle(target_vega=1000, spot=5.10, ttm_days=10, iv_pct=9.0)
    n_long = sizing.size_straddle(target_vega=1000, spot=5.10, ttm_days=63, iv_pct=9.0)

    assert n_long < n_short


def test_size_straddle_matches_manual_calc():
    spot, ttm, iv = 5.10, 21, 9.0
    vega_per_contract = sizing.straddle_vega_per_contract(spot, ttm, iv)
    n = sizing.size_straddle(target_vega=500, spot=spot, ttm_days=ttm, iv_pct=iv)

    assert n * vega_per_contract == pytest.approx(500)
