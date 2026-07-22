import pytest

from vol import black76

F = 5.10
K = 5.10
T = 21 / 365
SIGMA = 0.10
R = 0.05


def test_put_call_parity():
    c = black76.call_price(F, K, T, SIGMA, R)
    p = black76.put_price(F, K, T, SIGMA, R)
    df = black76._discount_factor(R, T)

    assert (c - p) == pytest.approx(df * (F - K), abs=1e-10)


def test_atm_call_and_put_have_equal_vega():
    v_call_bump = (
        black76.call_price(F, K, T, SIGMA + 1e-4, R) - black76.call_price(F, K, T, SIGMA - 1e-4, R)
    ) / 2e-4
    v_put_bump = (
        black76.put_price(F, K, T, SIGMA + 1e-4, R) - black76.put_price(F, K, T, SIGMA - 1e-4, R)
    ) / 2e-4

    assert v_call_bump == pytest.approx(v_put_bump, rel=1e-3)
    assert black76.vega(F, K, T, SIGMA, R) == pytest.approx(v_call_bump, rel=1e-3)


def test_vega_matches_finite_difference_of_price():
    bump = 1e-4
    finite_diff = (
        black76.call_price(F, K, T, SIGMA + bump, R) - black76.call_price(F, K, T, SIGMA - bump, R)
    ) / (2 * bump)
    assert black76.vega(F, K, T, SIGMA, R) == pytest.approx(finite_diff, rel=1e-4)


def test_vega_positive_and_peaks_near_atm():
    v_atm = black76.vega(F, K=F, T=T, sigma=SIGMA)
    v_otm = black76.vega(F, K=F * 1.15, T=T, sigma=SIGMA)
    v_itm = black76.vega(F, K=F * 0.85, T=T, sigma=SIGMA)

    assert v_atm > 0
    assert v_atm > v_otm
    assert v_atm > v_itm


def test_straddle_vega_is_twice_single_leg_vega():
    assert black76.straddle_vega(F, K, T, SIGMA, R) == pytest.approx(
        2 * black76.vega(F, K, T, SIGMA, R)
    )


def test_call_price_positive_and_bounded_by_forward():
    price = black76.call_price(F, K, T, SIGMA, R)
    df = black76._discount_factor(R, T)
    assert 0 < price < df * F


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        black76.vega(F, K, T=0, sigma=SIGMA)
    with pytest.raises(ValueError):
        black76.vega(F, K, T, sigma=0)
