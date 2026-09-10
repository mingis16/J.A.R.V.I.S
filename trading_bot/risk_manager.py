from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PositionSize:
    lots: float
    risk_amount: float
    sl_distance_price: float
    blocked_reason: str | None = None


def round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


def compute_position_size(
    *,
    equity: float,
    risk_per_trade_pct: float,
    sl_distance_price: float,
    tick_value: float,
    tick_size: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
) -> PositionSize:
    """Sizes a position so that hitting the stop loss loses ~risk_per_trade_pct of equity.

    value_per_price_unit_per_lot is the account-currency P&L change for a
    1.0 price-unit move, for 1.0 lot, derived from the symbol's tick value
    and tick size (as reported by the broker for the current symbol).
    """
    if sl_distance_price <= 0:
        return PositionSize(0.0, 0.0, sl_distance_price, "invalid stop-loss distance <= 0")
    if tick_size <= 0 or tick_value <= 0:
        return PositionSize(0.0, 0.0, sl_distance_price, "invalid symbol tick_size/tick_value")

    risk_amount = equity * (risk_per_trade_pct / 100.0)
    value_per_price_unit_per_lot = tick_value / tick_size
    loss_per_lot = sl_distance_price * value_per_price_unit_per_lot
    if loss_per_lot <= 0:
        return PositionSize(0.0, risk_amount, sl_distance_price, "non-positive loss_per_lot")

    raw_lots = risk_amount / loss_per_lot
    lots = round_to_step(raw_lots, volume_step)
    lots = max(volume_min, min(volume_max, lots))

    if lots < volume_min:
        return PositionSize(0.0, risk_amount, sl_distance_price, "sized lots below broker volume_min")

    return PositionSize(lots, risk_amount, sl_distance_price, None)


@dataclass
class DailyDrawdownGuard:
    """Circuit breaker: blocks new trades once daily loss exceeds the cap."""

    max_daily_loss_pct: float
    _day_start_equity: float | None = None
    _day_key: str | None = None

    def check(self, current_equity: float, day_key: str) -> tuple[bool, str | None]:
        if self._day_key != day_key:
            self._day_key = day_key
            self._day_start_equity = current_equity
            return True, None

        assert self._day_start_equity is not None
        floor = self._day_start_equity * (1 - self.max_daily_loss_pct / 100.0)
        if current_equity < floor:
            loss_pct = (1 - current_equity / self._day_start_equity) * 100
            return False, f"daily loss {loss_pct:.2f}% exceeds cap {self.max_daily_loss_pct:.2f}%"
        return True, None
