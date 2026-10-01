"""Hard trading limits for Alex's desk — enforced here, in code, on every
order. The model never passes a lot size and can't change these values: the
desk's tool surface has no file/shell access, and every place_trade call is
checked against check_new_trade() before anything reaches the broker.

Pure functions only (no MT5, no API), so every rule is unit-tested.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeskLimits:
    risk_per_trade_pct: float
    max_daily_loss_pct: float
    max_open_positions: int
    max_trades_per_day: int
    min_reward_risk: float

    @classmethod
    def from_cfg(cls, limits_cfg: dict) -> "DeskLimits":
        return cls(
            risk_per_trade_pct=float(limits_cfg["risk_per_trade_pct"]),
            max_daily_loss_pct=float(limits_cfg["max_daily_loss_pct"]),
            max_open_positions=int(limits_cfg["max_open_positions"]),
            max_trades_per_day=int(limits_cfg["max_trades_per_day"]),
            min_reward_risk=float(limits_cfg["min_reward_risk"]),
        )

    def describe(self) -> str:
        return (
            f"risk per trade at most {self.risk_per_trade_pct:g}% of equity (code sizes the lots); "
            f"daily loss stop {self.max_daily_loss_pct:g}% (no new trades for the rest of the UTC day); "
            f"at most {self.max_open_positions} open positions, one per symbol; "
            f"at most {self.max_trades_per_day} new trades per day; "
            f"stop-loss and take-profit required, reward:risk at least {self.min_reward_risk:g}"
        )


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    point: float
    stops_level_points: int


def daily_loss_pct(day_start_balance: float, equity: float) -> float:
    """Loss since the start of the UTC day, as a positive percentage (0 if up).
    Uses equity, so open losing positions count toward the daily stop."""
    if day_start_balance <= 0:
        return 0.0
    return max(0.0, (day_start_balance - equity) / day_start_balance * 100.0)


def min_stop_distance(quote: Quote) -> float:
    """Broker minimum, and never tighter than 3 spreads — a stop inside a few
    spreads is mostly paying the spread to get stopped out by noise."""
    spread = quote.ask - quote.bid
    return max(quote.stops_level_points * quote.point, 3 * spread)


def check_new_trade(
    *,
    direction: str,
    stop_loss: float,
    take_profit: float,
    risk_pct: float,
    quote: Quote,
    limits: DeskLimits,
    symbol: str,
    open_position_symbols: list[str],
    trades_today: int,
    loss_today_pct: float,
) -> str | None:
    """Returns why the trade is rejected, or None if it passes every limit."""
    if direction not in ("buy", "sell"):
        return f"direction must be 'buy' or 'sell', got {direction!r}"
    if loss_today_pct >= limits.max_daily_loss_pct:
        return (
            f"daily loss stop hit ({loss_today_pct:.2f}% >= {limits.max_daily_loss_pct:g}%) — "
            "no new trades until the next UTC day"
        )
    if trades_today >= limits.max_trades_per_day:
        return f"already placed {trades_today} trades today (max {limits.max_trades_per_day})"
    if len(open_position_symbols) >= limits.max_open_positions:
        return f"{len(open_position_symbols)} positions already open (max {limits.max_open_positions})"
    if symbol in open_position_symbols:
        return f"already have an open position on {symbol}"
    if not (0 < risk_pct <= limits.risk_per_trade_pct):
        return f"risk_pct must be above 0 and at most {limits.risk_per_trade_pct:g}"

    is_buy = direction == "buy"
    entry = quote.ask if is_buy else quote.bid
    if is_buy and not (stop_loss < entry < take_profit):
        return f"for a buy, stop_loss must be below entry ({entry}) and take_profit above it"
    if not is_buy and not (take_profit < entry < stop_loss):
        return f"for a sell, stop_loss must be above entry ({entry}) and take_profit below it"

    risk_distance = abs(entry - stop_loss)
    reward_distance = abs(take_profit - entry)
    minimum = min_stop_distance(quote)
    if risk_distance < minimum:
        return f"stop is {risk_distance:.5f} from entry; minimum is {minimum:.5f} (broker level / 3x spread)"
    if reward_distance / risk_distance < limits.min_reward_risk:
        return (
            f"reward:risk is {reward_distance / risk_distance:.2f}; "
            f"must be at least {limits.min_reward_risk:g}"
        )
    return None


def check_stop_move(*, is_buy: bool, current_stop: float, new_stop: float, bid: float, ask: float) -> str | None:
    """A stop may only move toward profit (tighten), never away — widening a
    stop is how a planned small loss becomes an unplanned large one."""
    if is_buy:
        if current_stop and new_stop <= current_stop:
            return f"a buy's stop can only move up (current {current_stop}, requested {new_stop})"
        if new_stop >= bid:
            return f"a buy's stop must stay below the current bid ({bid})"
    else:
        if current_stop and new_stop >= current_stop:
            return f"a sell's stop can only move down (current {current_stop}, requested {new_stop})"
        if new_stop <= ask:
            return f"a sell's stop must stay above the current ask ({ask})"
    return None
