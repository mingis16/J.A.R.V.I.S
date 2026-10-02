"""Chat-Alex's controls for the trading desk: the user sets the goal, checks
status, pauses/resumes, or hits the emergency close-all — by text, voice, or
Telegram. None of these can change the hard limits in config.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from assistant.tools import Tool, ToolRegistry
from assistant.trading_desk.desk import DeskJournal, money
from assistant.trading_desk.state import DeskStateStore
from trading_bot.config import live_trading_confirmed
from trading_bot.trade_log import TradeLog


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def register_desk_tools(registry: ToolRegistry, repo_root: Path, cfg: dict) -> None:
    desk_cfg = cfg.get("trading_desk")
    if not desk_cfg:
        return
    store = DeskStateStore(repo_root / "state")
    journal = DeskJournal(repo_root / "logs" / "trading_desk.jsonl")
    risk_state_path = repo_root / "state" / "trading_desk_risk.json"
    briefing_path = repo_root / "state" / "daily_briefing.json"
    trade_log = TradeLog(repo_root / cfg["assistant"]["trade_log_path"])
    magic = int(desk_cfg["magic_number"])

    def connect():
        from trading_bot.config import get_mt5_credentials
        from trading_bot.mt5_adapter import MT5Adapter

        adapter = MT5Adapter(get_mt5_credentials())
        adapter.connect()
        return adapter

    def status(_: dict[str, Any]) -> str:
        live = bool(desk_cfg.get("live")) and live_trading_confirmed()
        report: dict[str, Any] = {
            "enabled": bool(desk_cfg.get("enabled")),
            "mode": "LIVE (real money)" if live else "PAPER (logged only)",
            "live_gate": {
                "config_trading_desk.live": bool(desk_cfg.get("live")),
                "env_JARVIS_CONFIRM_LIVE": live_trading_confirmed(),
            },
            "control": store.control().__dict__,
            "today": store.day().__dict__,
            "limits": desk_cfg["limits"],
            "recent_cycles": [
                {"ts": e.get("ts"), "summary": e.get("summary"), "est_cost_usd": e.get("est_cost_usd")}
                for e in journal.tail(5)
            ],
            # What each department/team said last cycle, and which ones the CEO sent back.
            "last_cycle_team_reports": {
                team: (text[:2500] + "…" if len(text) > 2500 else text)
                for team, text in (journal.tail(1)[0].get("reports") or {}).items()
            } if journal.tail(1) else {},
            "last_cycle_followups": journal.tail(1)[0].get("followups") if journal.tail(1) else [],
            "last_cycle_risk_gate": journal.tail(1)[0].get("risk_gate") if journal.tail(1) else None,
            "risk_state": _read_json(risk_state_path),
            "daily_briefing": {k: _read_json(briefing_path).get(k) for k in ("date", "created_utc", "web", "text")},
            "api_spend_estimate_usd_all_cycles": round(sum(e.get("est_cost_usd") or 0 for e in journal.tail(10**6)), 2),
        }
        try:
            adapter = connect()
            try:
                account = adapter.get_account_info()
                report["account"] = {
                    "balance": money(account.balance, account.currency),
                    "equity": money(account.equity, account.currency),
                    "trading_allowed_by_broker": bool(account.trade_allowed),
                }
                report["open_desk_positions"] = [
                    {"ticket": p.ticket, "symbol": p.symbol, "side": "BUY" if p.type == 0 else "SELL",
                     "lots": p.volume, "open": p.price_open, "sl": p.sl, "tp": p.tp,
                     "pnl": money(p.profit, account.currency)}
                    for p in adapter.get_open_positions() if p.magic == magic
                ]
            finally:
                adapter.disconnect()
        except Exception as exc:
            report["account"] = f"couldn't reach MT5: {type(exc).__name__}: {exc}"
        return json.dumps(report, indent=2, default=str)

    def set_goal(inp: dict[str, Any]) -> str:
        goal = str(inp["goal"]).strip()
        store.update_control(goal=goal)
        return (
            f"Trading goal set: {goal!r}. The desk sees it from its next hourly cycle. The hard limits "
            "still apply unchanged — the goal steers decisions, it can't loosen risk."
        )

    def pause(inp: dict[str, Any]) -> str:
        store.update_control(paused=True)
        return "Trading desk paused: no new cycles until resumed. Open positions keep their stop-loss/take-profit."

    def resume(_: dict[str, Any]) -> str:
        store.update_control(paused=False)
        risk = _read_json(risk_state_path)
        if risk.get("drawdown_halted"):
            # Re-arming after a peak-drawdown halt is the user's call, and only
            # theirs: reset the peak to wherever equity is now.
            _write_json(risk_state_path, {"peak_equity": 0.0, "drawdown_halted": False, "rearmed_after": risk})
            return (
                "Trading desk resumed and re-armed after the drawdown halt: the peak-drawdown limit now "
                "measures from current equity. It runs again at the next hourly cycle."
            )
        return "Trading desk resumed; it runs again at the next hourly cycle."

    def close_all(inp: dict[str, Any]) -> str:
        reasoning = str(inp.get("reasoning") or "User asked to close everything.")
        store.update_control(paused=True)
        adapter = connect()
        try:
            positions = [p for p in adapter.get_open_positions() if p.magic == magic]
            currency = adapter.get_account_info().currency
            lines = []
            for p in positions:
                result = adapter.close_position(p, deviation=int(desk_cfg.get("deviation_points", 20)))
                trade_log.record(
                    trade_log.new_record(
                        symbol=p.symbol, action="CLOSE", mode="live", lots=p.volume,
                        price=result.price or p.price_current, sl=p.sl, tp=p.tp, reason=reasoning,
                        order_id=p.ticket,
                        status="closed" if result.success else f"failed:{result.retcode}:{result.comment}",
                        extra={"source": "alex_desk", "profit": p.profit, "currency": currency},
                    )
                )
                lines.append(f"#{p.ticket} {p.symbol}: {'closed' if result.success else 'FAILED ' + result.comment}")
        finally:
            adapter.disconnect()
        closed = "\n".join(lines) if lines else "No open desk positions."
        return f"{closed}\nDesk paused so it doesn't reopen anything; resume when ready."

    registry.register(Tool(
        name="trading_desk_status",
        description=(
            "Status of the autonomous trading desk: live/paper mode, goal, paused state, today's "
            "loss/trade counts, recent decisions, estimated API spend, account balance, open positions."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=status,
    ))
    registry.register(Tool(
        name="trading_desk_set_goal",
        description=(
            "Set the goal the trading desk works toward (e.g. 'grow the account 10% this month'). "
            "It steers decisions only; the code-enforced risk limits never change."
        ),
        input_schema={
            "type": "object",
            "properties": {"goal": {"type": "string"}},
            "required": ["goal"],
            "additionalProperties": False,
        },
        handler=set_goal,
    ))
    registry.register(Tool(
        name="trading_desk_pause",
        description="Pause the trading desk (no new cycles). Open positions keep their stops.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=pause,
    ))
    registry.register(Tool(
        name="trading_desk_resume",
        description=(
            "Resume a paused trading desk. Also re-arms it after a peak-drawdown halt by Risk "
            "Intelligence — only do that when the user explicitly asks to resume or re-arm."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=resume,
    ))
    registry.register(Tool(
        name="trading_desk_close_all",
        description=(
            "EMERGENCY: close every open desk position at market and pause the desk. Only when the "
            "user explicitly asks to close everything / stop trading now."
        ),
        input_schema={
            "type": "object",
            "properties": {"reasoning": {"type": "string"}},
            "additionalProperties": False,
        },
        handler=close_all,
    ))
