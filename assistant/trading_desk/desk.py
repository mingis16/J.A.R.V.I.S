"""Alex's autonomous trading desk.

Once an hour during the configured session (run by the daemon), code builds
a compact market brief from MT5, and Alex decides whether to open, manage, or
close positions through a deliberately small tool surface: candles,
place_trade, close_position, move_stop_loss. No shell, no file access — so
nothing here can loosen the limits in config.yaml.

Every order goes through limits.check_new_trade() and is sized by code from
the risk limit; the model never chooses a lot size. Real orders additionally
require trading_desk.live: true AND JARVIS_CONFIRM_LIVE in .env — otherwise
trades are logged as paper only. Every trade (paper or live) lands in
state/trades.jsonl, which the Telegram bot pushes to the user's phone.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from assistant.tools import Tool, ToolRegistry
from assistant.trading_desk.limits import (
    DeskLimits,
    Quote,
    check_new_trade,
    check_stop_move,
    daily_loss_pct,
    min_stop_distance,
)
from assistant.trading_desk.state import DeskStateStore, roll_day
from trading_bot.config import live_trading_confirmed
from trading_bot.risk_manager import compute_position_size
from trading_bot.strategy import atr, ema, generate_signal, rsi
from trading_bot.trade_log import TradeLog

logger = logging.getLogger("assistant.trading_desk")

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_REQUESTS_PER_CYCLE = 6
MARKET_STALE_SECONDS = 30 * 60
CANDLE_TIMEFRAMES = ("M5", "M15", "H1", "H4", "D1")

SYSTEM_PROMPT = """\
You are Alex, running the user's trading desk on their Exness MetaTrader 5 account. Each hour \
during the London/New York session you receive a market brief, then decide whether to open, \
manage, or close positions using your tools. The brief states whether this is LIVE (real money) \
or PAPER (logged only).

The hard limits listed in the brief are enforced in code: any order that breaks one is rejected, \
code sizes every position from the risk limit, and nothing you can do changes them.

How to trade:
- Capital preservation comes first. The user's goal sets direction, not urgency: never take a \
trade you wouldn't take without it, never widen a stop or size up to make back a loss, and \
standing aside is a normal, good outcome — most cycles should end with no new trade.
- Trade only when the brief gives a concrete reason: trend agreement across timeframes, a clear \
level, sensible volatility. Put the stop beyond the level that would prove the idea wrong, and \
the take-profit at a realistic target that meets the minimum reward:risk.
- Spread is a real cost on a small account. Prefer the tightest-spread symbols and avoid stops \
that are only a few spreads wide.
- Use the long-term context in the brief (8 years of daily history per symbol: where price sits \
in its multi-year and 52-week range, typical daily range) to judge whether a target is realistic \
and a stop is outside normal noise. Historical evidence: an 8.2-year walk-forward backtest of the \
EMA-cross setup on EURUSD, GBPUSD and USDJPY found no edge after costs in any period — treat the \
"EMA-cross bot signal" line as context, never as a reason to trade on its own.
- Manage open positions: close or tighten a stop when the reason for the trade is gone, \
otherwise let the stop-loss and take-profit do their job.
- The user sees every trade on their phone with your reasoning — write it in 1-3 plain \
sentences they can follow.
- Finish with a 1-3 sentence summary of what you did and why (or why you stood aside). It is \
saved to your journal and shown to you next cycle, so note anything you want to remember.
"""


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _round_to_step(volume: float, step: float) -> float:
    decimals = max(0, -int(math.floor(math.log10(step)))) if step > 0 else 2
    return round(volume, decimals)


def summarize_timeframe(df: pd.DataFrame) -> dict[str, float | str]:
    close = df["close"]
    ema20 = float(ema(close, 20).iloc[-1])
    ema50 = float(ema(close, 50).iloc[-1])
    return {
        "close": float(close.iloc[-1]),
        "ema20": ema20,
        "ema50": ema50,
        "trend": "up" if ema20 > ema50 else "down",
        "rsi14": float(rsi(close, 14).iloc[-1]),
        "atr14": float(atr(df, 14).iloc[-1]),
    }


def long_term_context(d1: pd.DataFrame, digits: int) -> str:
    """Multi-year daily context: where price sits in its full-history and
    52-week ranges, and the typical daily range (for realistic stops/targets)."""
    close = float(d1["close"].iloc[-1])
    years = (d1["time"].iloc[-1] - d1["time"].iloc[0]).days / 365.25
    low, high = float(d1["low"].min()), float(d1["high"].max())
    position = (close - low) / (high - low) * 100 if high > low else 50.0
    year = d1.tail(260)
    avg_daily_range = float((d1["high"] - d1["low"]).tail(20).mean())
    return (
        f"    {years:.1f}y daily history: range {_fmt(low, digits)}-{_fmt(high, digits)}, price at {position:.0f}% of it; "
        f"52-week range {_fmt(float(year['low'].min()), digits)}-{_fmt(float(year['high'].max()), digits)}; "
        f"avg daily range (20d) {_fmt(avg_daily_range, digits)}"
    )


def money(amount: float, currency: str) -> str:
    if currency == "USC":
        return f"{amount:.2f} USC (~${amount / 100:.2f})"
    return f"{amount:.2f} {currency}"


class DeskJournal:
    """Append-only record of every desk cycle — decisions, actions, token use."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines[-n:] if line]


class TradingDesk:
    def __init__(self, repo_root: Path, cfg: dict, client: Any, adapter: Any = None):
        self.cfg = cfg
        self.desk_cfg = cfg["trading_desk"]
        self.limits = DeskLimits.from_cfg(self.desk_cfg["limits"])
        self.trade_log = TradeLog(repo_root / cfg["assistant"]["trade_log_path"])
        self.journal = DeskJournal(repo_root / "logs" / "trading_desk.jsonl")
        self.store = DeskStateStore(repo_root / "state")
        self.client = client
        if adapter is None:
            from trading_bot.config import get_mt5_credentials
            from trading_bot.mt5_adapter import MT5Adapter

            adapter = MT5Adapter(get_mt5_credentials())
        self.adapter = adapter
        self.magic = int(self.desk_cfg["magic_number"])
        self.symbols: dict[str, str] = {}  # base -> broker name
        self._day = None

    # ----- mode / account -------------------------------------------------

    def is_live(self) -> bool:
        return bool(self.desk_cfg.get("live")) and live_trading_confirmed()

    def _sizing_equity(self, account: Any) -> float:
        return float(account.equity) if self.is_live() else float(self.desk_cfg["paper_equity"])

    def _desk_positions(self) -> list[Any]:
        return [p for p in self.adapter.get_open_positions() if p.magic == self.magic]

    def _resolve(self, name: str) -> str:
        if name in self.symbols.values():
            return name
        if name.upper() in self.symbols:
            return self.symbols[name.upper()]
        raise ValueError(f"unknown symbol {name!r}; tradable here: {sorted(self.symbols.values())}")

    # ----- cycle ------------------------------------------------------------

    def run_cycle(self, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        if self.store.control().paused:
            return "paused by the user — skipped"
        start_hour, end_hour = self.desk_cfg.get("session_hours_utc", [0, 24])
        if not (start_hour <= now.hour < end_hour):
            return f"outside session hours ({start_hour:02d}:00-{end_hour:02d}:00 UTC) — skipped"

        self.adapter.connect()
        try:
            return self._run_connected(now)
        finally:
            self.adapter.disconnect()

    def _run_connected(self, now: datetime) -> str:
        live = self.is_live()
        account = self.adapter.get_account_info()
        day = roll_day(
            self.store.day(),
            now.date().isoformat(),
            float(account.balance) if live else float(self.desk_cfg["paper_equity"]),
        )
        self._day = day
        self.store.save_day(day)

        if day.cycles_today >= int(self.desk_cfg.get("max_cycles_per_day", 16)):
            return f"already ran {day.cycles_today} cycles today (cap) — skipped"
        if live and not account.trade_allowed:
            return "the broker has trading disabled on this account (unfunded or restricted) — skipped"

        self.symbols = {}
        for base in self.desk_cfg["symbols"]:
            name = self.adapter.resolve_symbol(base)
            if name:
                self.symbols[base.upper()] = name
        if not self.symbols:
            return f"none of {self.desk_cfg['symbols']} are tradable on this account — skipped"

        now_ts = now.timestamp()
        ticks = {name: self.adapter.get_tick(name) for name in self.symbols.values()}
        if all(now_ts - t.time > MARKET_STALE_SECONDS for t in ticks.values()):
            return "market closed or MT5 offline (no fresh prices in 30 min) — skipped"

        positions = self._desk_positions()
        loss = daily_loss_pct(day.day_start_balance, float(account.equity)) if live else 0.0
        if loss >= self.limits.max_daily_loss_pct and not positions:
            return f"daily loss stop hit ({loss:.2f}%) and nothing open to manage — skipped"

        brief = self.build_brief(now, account, day, loss, positions, ticks)
        day.cycles_today += 1
        self.store.save_day(day)

        summary, actions, usage = self._ask_alex(brief)
        self.journal.record(
            {
                "ts": now.isoformat(),
                "mode": "live" if live else "paper",
                "summary": summary,
                "actions": actions,
                "usage": usage,
                "est_cost_usd": self._estimate_cost(usage),
            }
        )
        return summary

    def build_brief(self, now: datetime, account: Any, day: Any, loss: float, positions: list[Any], ticks: dict) -> str:
        live = self.is_live()
        currency = account.currency
        sizing_equity = self._sizing_equity(account)
        goal = self.store.control().goal or "(no goal set — trade conservatively for steady growth)"
        lines = [
            f"Time: {now:%Y-%m-%d %H:%M} UTC",
            f"Mode: {'LIVE — real money' if live else 'PAPER — orders are logged, not sent'}",
            f"User's goal: {goal}",
            f"Account: balance {money(account.balance, currency)}, equity {money(account.equity, currency)}, "
            f"free margin {money(account.margin_free, currency)}"
            + ("" if live else f"; paper sizing uses {money(sizing_equity, currency)}"),
            f"Today: loss vs day start {loss:.2f}% (stop at {self.limits.max_daily_loss_pct:g}%), "
            f"trades placed {day.trades_today}/{self.limits.max_trades_per_day}, "
            f"open positions {len(positions)}/{self.limits.max_open_positions}",
            f"Hard limits (enforced in code): {self.limits.describe()}.",
            "",
            "Open desk positions:" if positions else "Open desk positions: none",
        ]
        for p in positions:
            info = self.adapter.get_symbol_info(p.symbol)
            d = info.digits
            side = "BUY" if p.type == 0 else "SELL"
            lines.append(
                f"  #{p.ticket} {p.symbol} {side} {p.volume} @ {_fmt(p.price_open, d)} "
                f"SL {_fmt(p.sl, d)} TP {_fmt(p.tp, d)} now {_fmt(p.price_current, d)} "
                f"P/L {money(p.profit, currency)}"
            )

        lines += ["", "Markets (EMA20/EMA50 trend, RSI14, ATR14):"]
        for base, name in self.symbols.items():
            info = self.adapter.get_symbol_info(name)
            tick = ticks[name]
            d = info.digits
            spread_points = round((tick.ask - tick.bid) / info.point)
            if now.timestamp() - tick.time > MARKET_STALE_SECONDS:
                lines.append(f"  {name}: no fresh prices (closed) — don't trade it this cycle")
                continue
            parts = [f"  {name}: bid {_fmt(tick.bid, d)} ask {_fmt(tick.ask, d)} spread {spread_points} points"]
            h1 = None
            for tf in ("H4", "H1", "M15"):
                df = self.adapter.get_rates(name, tf, 120)
                s = summarize_timeframe(df)
                parts.append(
                    f"    {tf}: close {_fmt(s['close'], d)}, trend {s['trend']} "
                    f"(EMA20 {_fmt(s['ema20'], d)} / EMA50 {_fmt(s['ema50'], d)}), "
                    f"RSI {s['rsi14']:.1f}, ATR {_fmt(s['atr14'], d)}"
                )
                if tf == "H1":
                    h1 = df
                if tf == "M15" and "trading" in self.cfg:
                    parts.append(f"    EMA-cross bot signal (M15): {generate_signal(df, self.cfg['trading']).action}")
            if h1 is not None:
                last24 = h1.tail(24)
                parts.append(
                    f"    last 24h: high {_fmt(last24['high'].max(), d)}, low {_fmt(last24['low'].min(), d)}; "
                    f"last 6 H1 closes: {', '.join(_fmt(c, d) for c in h1['close'].tail(6))}"
                )
            parts.append(long_term_context(self.adapter.get_rates(name, "D1", 2100), d))
            lines += parts

        journal = self.journal.tail(int(self.desk_cfg.get("journal_entries_in_brief", 6)))
        lines += ["", "Your recent journal (oldest first):" if journal else "Your recent journal: empty (first cycle)"]
        for entry in journal:
            lines.append(f"  {entry.get('ts', '')[:16]} [{entry.get('mode')}] {entry.get('summary', '')}")
        return "\n".join(lines)

    def _ask_alex(self, brief: str) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
        registry = self.build_registry()
        tools = registry.as_api_tools()
        messages: list[dict[str, Any]] = [{"role": "user", "content": brief}]
        actions: list[dict[str, Any]] = []
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

        for _ in range(MAX_REQUESTS_PER_CYCLE):
            response = self.client.beta.messages.create(
                model=self.desk_cfg["model"],
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
                output_config={"effort": self.desk_cfg.get("effort", "low")},
                cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
            for key in usage:
                usage[key] += int(getattr(response.usage, key, 0) or 0)

            if response.stop_reason == "refusal":
                return "the model declined this cycle (refusal) — no action taken", actions, usage
            if response.stop_reason != "tool_use":
                text = "\n".join(b.text for b in response.content if b.type == "text").strip()
                return text or "(no summary)", actions, usage

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                output = registry.execute(block.name, block.input)
                if block.name != "get_candles":
                    actions.append({"tool": block.name, "input": block.input, "result": output})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            messages.append({"role": "user", "content": results})
        return "(hit the per-cycle request limit before finishing)", actions, usage

    def _estimate_cost(self, usage: dict[str, int]) -> float:
        prices = self.desk_cfg.get("price_per_mtok", {"input": 4.0, "output": 20.0})
        p_in, p_out = float(prices["input"]), float(prices["output"])
        cost = (
            usage["input_tokens"] * p_in
            + usage["cache_creation_input_tokens"] * p_in * 1.25
            + usage["cache_read_input_tokens"] * p_in * 0.1
            + usage["output_tokens"] * p_out
        ) / 1_000_000
        return round(cost, 4)

    # ----- tools ------------------------------------------------------------

    def build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="get_candles",
                description="Recent OHLC candles for one symbol, oldest first, for a closer look than the brief gives.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "timeframe": {"type": "string", "enum": list(CANDLE_TIMEFRAMES)},
                        "count": {"type": "integer", "minimum": 5, "maximum": 100},
                    },
                    "required": ["symbol", "timeframe"],
                    "additionalProperties": False,
                },
                handler=self._tool_get_candles,
            )
        )
        registry.register(
            Tool(
                name="place_trade",
                description=(
                    "Open a market position with a stop-loss and take-profit. Code sizes the position "
                    "from risk_pct (default and maximum: the per-trade risk limit) and rejects anything "
                    "that breaks a hard limit, saying why. Entry is the current ask for a buy, bid for a sell."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "direction": {"type": "string", "enum": ["buy", "sell"]},
                        "stop_loss": {"type": "number"},
                        "take_profit": {"type": "number"},
                        "risk_pct": {"type": "number", "description": "Optional; lower than the limit when less confident."},
                        "reasoning": {"type": "string", "description": "1-3 sentences, sent to the user."},
                    },
                    "required": ["symbol", "direction", "stop_loss", "take_profit", "reasoning"],
                    "additionalProperties": False,
                },
                handler=self._tool_place_trade,
            )
        )
        registry.register(
            Tool(
                name="close_position",
                description="Close one of the desk's open positions at market.",
                input_schema={
                    "type": "object",
                    "properties": {"ticket": {"type": "integer"}, "reasoning": {"type": "string"}},
                    "required": ["ticket", "reasoning"],
                    "additionalProperties": False,
                },
                handler=self._tool_close_position,
            )
        )
        registry.register(
            Tool(
                name="move_stop_loss",
                description="Move an open position's stop-loss toward profit (tighten only; widening is rejected).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticket": {"type": "integer"},
                        "new_stop_loss": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["ticket", "new_stop_loss", "reasoning"],
                    "additionalProperties": False,
                },
                handler=self._tool_move_stop_loss,
            )
        )
        return registry

    def _tool_get_candles(self, inp: dict[str, Any]) -> str:
        symbol = self._resolve(inp["symbol"])
        timeframe = inp["timeframe"]
        if timeframe not in CANDLE_TIMEFRAMES:
            return f"Error: timeframe must be one of {CANDLE_TIMEFRAMES}"
        count = min(max(int(inp.get("count", 50)), 5), 100)
        df = self.adapter.get_rates(symbol, timeframe, count)
        d = self.adapter.get_symbol_info(symbol).digits
        rows = [
            f"{row.time:%m-%d %H:%M} O {_fmt(row.open, d)} H {_fmt(row.high, d)} L {_fmt(row.low, d)} C {_fmt(row.close, d)}"
            for row in df.itertuples()
        ]
        return f"{symbol} {timeframe}, {len(rows)} candles (UTC):\n" + "\n".join(rows)

    def _tool_place_trade(self, inp: dict[str, Any]) -> str:
        symbol = self._resolve(inp["symbol"])
        direction = str(inp["direction"]).lower()
        reasoning = str(inp["reasoning"]).strip()
        risk_pct = float(inp.get("risk_pct") or self.limits.risk_per_trade_pct)
        live = self.is_live()

        info = self.adapter.get_symbol_info(symbol)
        tick = self.adapter.get_tick(symbol)
        account = self.adapter.get_account_info()
        quote = Quote(bid=tick.bid, ask=tick.ask, point=info.point, stops_level_points=info.trade_stops_level)
        stop_loss = round(float(inp["stop_loss"]), info.digits)
        take_profit = round(float(inp["take_profit"]), info.digits)
        day = self._day
        loss = daily_loss_pct(day.day_start_balance, float(account.equity)) if live else 0.0

        rejection = check_new_trade(
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_pct=risk_pct,
            quote=quote,
            limits=self.limits,
            symbol=symbol,
            open_position_symbols=[p.symbol for p in self._desk_positions()],
            trades_today=day.trades_today,
            loss_today_pct=loss,
        )
        if rejection:
            return f"REJECTED: {rejection}"

        is_buy = direction == "buy"
        entry = tick.ask if is_buy else tick.bid
        sl_distance = abs(entry - stop_loss)
        sizing = compute_position_size(
            equity=self._sizing_equity(account),
            risk_per_trade_pct=risk_pct,
            sl_distance_price=sl_distance,
            tick_value=info.trade_tick_value,
            tick_size=info.trade_tick_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
        )
        if sizing.blocked_reason:
            return (
                f"REJECTED: {sizing.blocked_reason} — at {risk_pct:g}% risk this stop is too wide for the "
                "account's smallest position size. Use a tighter, still-valid stop or skip the trade."
            )
        lots = _round_to_step(sizing.lots, info.volume_step)
        actual_risk = sl_distance * (info.trade_tick_value / info.trade_tick_size) * lots
        reward_risk = abs(take_profit - entry) / sl_distance

        order_id = None
        if live:
            margin = self.adapter.calc_margin(symbol, is_buy, lots, entry)
            if margin > 0.5 * float(account.margin_free):
                return f"REJECTED: needs {margin:.2f} margin, more than half of free margin ({account.margin_free:.2f})"
            result = self.adapter.place_market_order(
                symbol=symbol,
                is_buy=is_buy,
                volume=lots,
                sl=stop_loss,
                tp=take_profit,
                deviation=int(self.desk_cfg.get("deviation_points", 20)),
                magic=self.magic,
                comment=self.desk_cfg.get("order_comment", "alex-desk"),
            )
            status = "filled" if result.success else f"failed:{result.retcode}:{result.comment}"
            entry = result.price or entry
            order_id = result.order_id
        else:
            status = "simulated"

        self.trade_log.record(
            self.trade_log.new_record(
                symbol=symbol,
                action="BUY" if is_buy else "SELL",
                mode="live" if live else "paper",
                lots=lots,
                price=entry,
                sl=stop_loss,
                tp=take_profit,
                reason=reasoning,
                order_id=order_id,
                status=status,
                extra={
                    "source": "alex_desk",
                    "risk_pct": risk_pct,
                    "risk_amount": round(actual_risk, 2),
                    "reward_risk": round(reward_risk, 2),
                    "currency": account.currency,
                },
            )
        )
        if status not in ("filled", "simulated"):
            return f"Order FAILED at the broker: {status}. Nothing was opened."

        day.trades_today += 1
        self.store.save_day(day)
        return (
            f"{'Opened' if live else 'Paper-logged'} {direction.upper()} {lots} {symbol} @ {entry} "
            f"SL {stop_loss} TP {take_profit}; risk {money(actual_risk, account.currency)}, R:R {reward_risk:.2f}"
            + (f", ticket #{order_id}" if order_id else "")
        )

    def _find_position(self, ticket: int) -> Any:
        for p in self._desk_positions():
            if p.ticket == ticket:
                return p
        return None

    def _tool_close_position(self, inp: dict[str, Any]) -> str:
        position = self._find_position(int(inp["ticket"]))
        if position is None:
            return f"Error: no open desk position #{inp['ticket']}"
        result = self.adapter.close_position(position, deviation=int(self.desk_cfg.get("deviation_points", 20)))
        account = self.adapter.get_account_info()
        self.trade_log.record(
            self.trade_log.new_record(
                symbol=position.symbol,
                action="CLOSE",
                mode="live",
                lots=position.volume,
                price=result.price or position.price_current,
                sl=position.sl,
                tp=position.tp,
                reason=str(inp["reasoning"]).strip(),
                order_id=position.ticket,
                status="closed" if result.success else f"failed:{result.retcode}:{result.comment}",
                extra={"source": "alex_desk", "profit": position.profit, "currency": account.currency},
            )
        )
        if not result.success:
            return f"Close FAILED: {result.comment}"
        return f"Closed #{position.ticket} {position.symbol}, P/L about {money(position.profit, account.currency)}"

    def _tool_move_stop_loss(self, inp: dict[str, Any]) -> str:
        position = self._find_position(int(inp["ticket"]))
        if position is None:
            return f"Error: no open desk position #{inp['ticket']}"
        info = self.adapter.get_symbol_info(position.symbol)
        tick = self.adapter.get_tick(position.symbol)
        new_stop = round(float(inp["new_stop_loss"]), info.digits)
        is_buy = position.type == 0
        rejection = check_stop_move(
            is_buy=is_buy, current_stop=position.sl, new_stop=new_stop, bid=tick.bid, ask=tick.ask
        )
        minimum = min_stop_distance(Quote(tick.bid, tick.ask, info.point, info.trade_stops_level))
        price = tick.bid if is_buy else tick.ask
        if rejection is None and abs(price - new_stop) < minimum:
            rejection = f"new stop must be at least {minimum:.5f} from the current price"
        if rejection:
            return f"REJECTED: {rejection}"

        result = self.adapter.modify_position_sltp(position, new_stop, position.tp)
        account = self.adapter.get_account_info()
        self.trade_log.record(
            self.trade_log.new_record(
                symbol=position.symbol,
                action="MOVE_SL",
                mode="live",
                lots=position.volume,
                price=price,
                sl=new_stop,
                tp=position.tp,
                reason=str(inp["reasoning"]).strip(),
                order_id=position.ticket,
                status="modified" if result.success else f"failed:{result.retcode}:{result.comment}",
                extra={"source": "alex_desk", "previous_sl": position.sl, "currency": account.currency},
            )
        )
        if not result.success:
            return f"Stop move FAILED: {result.comment}"
        return f"Moved #{position.ticket} stop to {new_stop}"
