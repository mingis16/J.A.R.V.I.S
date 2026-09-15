"""Tool registry for the daemon's unattended Claude calls.

Deliberately excludes write_file, run_command, and trading_bot_start/stop —
an autonomous, unsupervised LLM call gets read access and a way to propose
changes, never a way to execute them. This is enforced here in code, not
left to prompt instructions the model could be talked out of.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from assistant.daemon.proposals import ProposalLog
from assistant.tools import Tool, ToolRegistry, _read_file
from trading_bot.trade_log import TradeLog


def build_readonly_registry(repo_root: Path, cfg: dict, memory, proposal_log: ProposalLog) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="read_file",
            description="Read a UTF-8 text file from the local filesystem (max 200KB).",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute or relative file path."}},
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=_read_file,
        )
    )
    registry.register(
        Tool(
            name="recall",
            description="Look up a previously remembered fact by key, or list all remembered facts if key is omitted.",
            input_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": [],
                "additionalProperties": False,
            },
            handler=lambda inp: json.dumps(
                memory.recall_fact(inp["key"]) if inp.get("key") else memory.all_facts(), default=str
            ),
        )
    )

    pid_path = repo_root / "state" / "trading_bot.pid"
    trade_log = TradeLog(repo_root / cfg["assistant"]["trade_log_path"])

    def trading_bot_status(_: dict[str, Any]) -> str:
        running = "running (pid file present)" if pid_path.exists() else "not running (no pid file)"
        return json.dumps({"process": running, "recent_trades": trade_log.tail(10)}, indent=2, default=str)

    registry.register(
        Tool(
            name="trading_bot_status",
            description="Get the trading bot process status and its most recent trade log entries. Read-only.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=trading_bot_status,
        )
    )

    def propose_change(inp: dict[str, Any]) -> str:
        proposal_log.record(
            routine=inp.get("routine", "unknown"),
            title=inp["title"],
            description=inp["description"],
            suggested_action=inp["suggested_action"],
        )
        return f"Proposal recorded: {inp['title']}"

    registry.register(
        Tool(
            name="propose_change",
            description=(
                "Record a suggested change for the user to review later — you cannot make changes "
                "yourself while running unattended. Use this instead of write_file/run_command."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "routine": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "suggested_action": {
                        "type": "string",
                        "description": "The concrete command or change you'd make, for the user to run/apply.",
                    },
                },
                "required": ["title", "description", "suggested_action"],
                "additionalProperties": False,
            },
            handler=propose_change,
        )
    )

    return registry
