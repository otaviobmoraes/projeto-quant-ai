import pytest

from backtest import costs


def test_transaction_cost_scales_with_contracts_and_spread():
    base = costs.transaction_cost(premium=0.10, n_contracts=100, spread_pct=0.05)
    assert base == pytest.approx(100 * 0.10 * 0.05)

    double_contracts = costs.transaction_cost(premium=0.10, n_contracts=200, spread_pct=0.05)
    assert double_contracts == pytest.approx(2 * base)


def test_transaction_cost_uses_absolute_value_of_contracts():
    long_cost = costs.transaction_cost(premium=0.10, n_contracts=100, spread_pct=0.05)
    short_cost = costs.transaction_cost(premium=0.10, n_contracts=-100, spread_pct=0.05)
    assert long_cost == pytest.approx(short_cost)


def test_round_trip_cost_is_sum_of_open_and_close():
    open_cost = costs.transaction_cost(premium=0.10, n_contracts=100, spread_pct=0.05)
    close_cost = costs.transaction_cost(premium=0.12, n_contracts=100, spread_pct=0.05)

    round_trip = costs.round_trip_cost(premium_open=0.10, premium_close=0.12, n_contracts=100, spread_pct=0.05)
    assert round_trip == pytest.approx(open_cost + close_cost)
