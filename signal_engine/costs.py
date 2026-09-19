"""Cost model, expressed in R units (R = the stop-loss distance).

Historical H1 OHLC bars carry no historical bid/ask spread, so the spread
input here is a snapshot from the live MT5 symbol (a documented
approximation — see REPORT.md Limitations). Slippage is a configured price
amount added to spread before converting to R; commission is a flat R
amount (0 for typical retail FX CFD accounts, configurable otherwise).
`cost_multiplier` scales the spread+slippage component for the 0.5x/1x/2x
sensitivity table the spec asks for.
"""
from __future__ import annotations


def cost_in_r(
    sl_distance_price: float,
    spread_price: float,
    slippage_price: float,
    commission_r: float,
    cost_multiplier: float = 1.0,
) -> float:
    if sl_distance_price <= 0:
        return float("nan")
    base_r = (spread_price + slippage_price) / sl_distance_price
    return base_r * cost_multiplier + commission_r


def expectancy_r(p_win: float, risk_reward: float, cost_r: float) -> float:
    """E = p*TP_R - (1-p)*1 - costs_R"""
    return p_win * risk_reward - (1 - p_win) * 1.0 - cost_r
