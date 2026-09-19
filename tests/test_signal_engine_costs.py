from __future__ import annotations

import math

from signal_engine.costs import cost_in_r, expectancy_r


def test_cost_in_r_basic():
    # sl_distance=0.0020, spread=0.0002, no slippage, no commission -> 0.0002/0.0020 = 0.1 R
    result = cost_in_r(sl_distance_price=0.0020, spread_price=0.0002, slippage_price=0.0, commission_r=0.0)
    assert math.isclose(result, 0.1)


def test_cost_in_r_scales_with_multiplier():
    base = cost_in_r(sl_distance_price=0.0020, spread_price=0.0002, slippage_price=0.0, commission_r=0.0, cost_multiplier=1.0)
    doubled = cost_in_r(sl_distance_price=0.0020, spread_price=0.0002, slippage_price=0.0, commission_r=0.0, cost_multiplier=2.0)
    assert math.isclose(doubled, base * 2)


def test_cost_in_r_adds_commission_flat():
    result = cost_in_r(sl_distance_price=0.0020, spread_price=0.0, slippage_price=0.0, commission_r=0.05)
    assert math.isclose(result, 0.05)


def test_cost_in_r_invalid_sl_distance_returns_nan():
    result = cost_in_r(sl_distance_price=0.0, spread_price=0.0002, slippage_price=0.0, commission_r=0.0)
    assert math.isnan(result)


def test_expectancy_r_breakeven_at_theoretical_probability():
    # for RR=2, breakeven p = 1/3 with zero costs
    rr = 2.0
    p_breakeven = 1 / (1 + rr)
    e = expectancy_r(p_breakeven, rr, cost_r=0.0)
    assert math.isclose(e, 0.0, abs_tol=1e-9)


def test_expectancy_r_positive_above_breakeven():
    e = expectancy_r(p_win=0.5, risk_reward=2.0, cost_r=0.0)
    assert e > 0


def test_expectancy_r_costs_reduce_expectancy():
    e_no_cost = expectancy_r(p_win=0.5, risk_reward=2.0, cost_r=0.0)
    e_with_cost = expectancy_r(p_win=0.5, risk_reward=2.0, cost_r=0.1)
    assert e_with_cost < e_no_cost
