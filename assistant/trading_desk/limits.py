"""Hard trading limits for Alex's desk — enforced here, in code, on every
order. The model never passes a lot size and can't change these values: the
desk's tool surface has no file/shell access, and every place_trade call is
checked against check_new_trade() before anything reaches the broker.

Pure functions only (no MT5, no API), so every rule is unit-tested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime


def fx_market_open(now: datetime) -> bool:
    """Spot FX trades from about Sunday 22:00 to Friday 21:00 UTC. Checked before
    connecting to MT5, so closed hours cost nothing (and don't hang on a
    terminal that has no live prices to give)."""
    weekday = now.weekday()  # Monday=0 .. Sunday=6
    if weekday == 5:
        return False
    if weekday == 6:
        return now.hour >= 22
    if weekday == 4:
        return now.hour < 21
    return True


@dataclass(frozen=True)
class DeskLimits:
    risk_per_trade_pct: float
    max_daily_loss_pct: float
    max_open_positions: int
    max_trades_per_day: int
    min_reward_risk: float
    max_drawdown_pct: float = 30.0
    losing_streak_reduce: int = 3
    losing_streak_halt: int = 5
    max_spread_multiple: float = 2.5
    min_margin_level_pct: float = 500.0
    friday_cutoff_hour_utc: int = 19

    @classmethod
    def from_cfg(cls, limits_cfg: dict) -> "DeskLimits":
        optional = {
            k: type(getattr(cls, k))(limits_cfg[k])
            for k in (
                "max_drawdown_pct", "losing_streak_reduce", "losing_streak_halt",
                "max_spread_multiple", "min_margin_level_pct", "friday_cutoff_hour_utc",
            )
            if k in limits_cfg
        }
        return cls(
            risk_per_trade_pct=float(limits_cfg["risk_per_trade_pct"]),
            max_daily_loss_pct=float(limits_cfg["max_daily_loss_pct"]),
            max_open_positions=int(limits_cfg["max_open_positions"]),
            max_trades_per_day=int(limits_cfg["max_trades_per_day"]),
            min_reward_risk=float(limits_cfg["min_reward_risk"]),
            **optional,
        )

    def describe(self) -> str:
        return (
            f"risk per trade at most {self.risk_per_trade_pct:g}% of equity (code sizes the lots); "
            f"daily loss stop {self.max_daily_loss_pct:g}% (no new trades for the rest of the UTC day); "
            f"peak drawdown halt at {self.max_drawdown_pct:g}% (stays halted until the user re-arms); "
            f"{self.losing_streak_reduce} losses in a row today halve the risk, {self.losing_streak_halt} halt the day; "
            f"at most {self.max_open_positions} open positions, one per symbol, never doubling one currency's "
            f"exposure; at most {self.max_trades_per_day} new trades per day; no new trades when the spread is over "
            f"{self.max_spread_multiple:g}x normal, prices are stale, margin level is under "
            f"{self.min_margin_level_pct:g}%, or after {self.friday_cutoff_hour_utc}:00 UTC Friday; "
            f"stop-loss and take-profit required, reward:risk at least {self.min_reward_risk:g}"
        )


# ----- Risk Intelligence: pre-cycle gate (department 9, code) ----------------

NORMAL, REDUCED, HALT_NEW = "NORMAL", "REDUCED", "HALT_NEW"


@dataclass
class RiskDirective:
    """The code gate's verdict for this cycle. It only ever restricts: no other
    department, team, or the CEO can raise max_risk_pct or lift a halt."""

    state: str
    max_risk_pct: float
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        why = "; ".join(self.reasons) if self.reasons else "no rule triggered"
        return f"{self.state} — max risk {self.max_risk_pct:g}% per trade ({why})"


def losing_streak(closed_results: list[float]) -> int:
    """Consecutive losses at the end of today's closed-trade results (oldest first)."""
    streak = 0
    for result in reversed(closed_results):
        if result >= 0:
            break
        streak += 1
    return streak


def pre_cycle_directive(
    *,
    limits: DeskLimits,
    peak_equity: float,
    equity: float,
    closed_results_today: list[float],
    margin_level_pct: float,
    has_positions: bool,
    drawdown_halted: bool,
) -> RiskDirective:
    """Most-restrictive-wins evaluation of the account-level rules."""
    reasons: list[str] = []
    state, max_risk = NORMAL, limits.risk_per_trade_pct

    drawdown = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
    if drawdown_halted or drawdown >= limits.max_drawdown_pct:
        state = HALT_NEW
        reasons.append(
            f"peak drawdown {drawdown:.1f}% (limit {limits.max_drawdown_pct:g}%) — halted until the user re-arms"
        )

    streak = losing_streak(closed_results_today)
    if streak >= limits.losing_streak_halt:
        state = HALT_NEW
        reasons.append(f"{streak} losses in a row today — no new trades until tomorrow")
    elif streak >= limits.losing_streak_reduce:
        if state == NORMAL:
            state = REDUCED
        max_risk = limits.risk_per_trade_pct / 2
        reasons.append(f"{streak} losses in a row today — risk per trade halved")

    if has_positions and 0 < margin_level_pct < limits.min_margin_level_pct:
        state = HALT_NEW
        reasons.append(f"margin level {margin_level_pct:.0f}% is under {limits.min_margin_level_pct:g}%")

    return RiskDirective(state=state, max_risk_pct=max_risk, reasons=reasons)


def currencies(symbol: str) -> tuple[str, str]:
    letters = re.sub(r"[^A-Z]", "", symbol.upper())[:6]
    return letters[:3], letters[3:6]


def currency_concentration(symbol: str, is_buy: bool, open_positions: list[tuple[str, bool]]) -> str | None:
    """Rejects a trade that would double one currency's net exposure — long
    EURUSD plus long GBPUSD (or plus long gold) is two bets against the USD,
    not two independent trades."""
    exposure: dict[str, int] = {}
    for sym, buy in [*open_positions, (symbol, is_buy)]:
        base, quote = currencies(sym)
        sign = 1 if buy else -1
        exposure[base] = exposure.get(base, 0) + sign
        exposure[quote] = exposure.get(quote, 0) - sign
    doubled = [f"{'long' if n > 0 else 'short'} {cur} x{abs(n)}" for cur, n in exposure.items() if abs(n) >= 2]
    if doubled:
        return f"would concentrate exposure ({', '.join(doubled)}) — that's one bet taken twice"
    return None


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
