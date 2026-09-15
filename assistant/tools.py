from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Patterns that are almost never intended by a legitimate assistant command
# and are cheap to block outright. This is a safety net, not a sandbox —
# run_command still executes arbitrary shell commands on the user's machine.
_DANGEROUS_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/(\s|$)"),
    re.compile(r"format\s+[a-zA-Z]:", re.IGNORECASE),
    re.compile(r"del\s+/f\s+/s\s+/q\s+[a-zA-Z]:\\?\s*$", re.IGNORECASE),
    re.compile(r"remove-item.+-recurse.+-force.+[a-zA-Z]:\\\s*$", re.IGNORECASE),
    re.compile(r"diskpart", re.IGNORECASE),
    re.compile(r"mkfs\.", re.IGNORECASE),
    re.compile(r"shutdown\s+/s", re.IGNORECASE),
]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def as_api_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._tools.values()
        ]

    def execute(self, name: str, tool_input: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return tool.handler(tool_input)
        except Exception as exc:  # tool failures must surface as tool_result, not crash the loop
            return f"Error: {type(exc).__name__}: {exc}"


def _read_file(inp: dict[str, Any]) -> str:
    path = Path(inp["path"]).expanduser()
    if not path.exists():
        return f"Error: file not found: {path}"
    if not path.is_file():
        return f"Error: not a file: {path}"
    max_bytes = 200_000
    data = path.read_bytes()
    if len(data) > max_bytes:
        return (
            f"Error: file is {len(data)} bytes, exceeds {max_bytes} byte read limit. "
            "Ask for a narrower excerpt or use run_command with a tool like findstr/grep."
        )
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return f"Error: file is not valid UTF-8 text ({path})"


def _write_file(inp: dict[str, Any]) -> str:
    path = Path(inp["path"]).expanduser()
    content = inp["content"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {path}"


def _make_run_command(audit_path: Path) -> Callable[[dict[str, Any]], str]:
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    def run_command(inp: dict[str, Any]) -> str:
        command = inp["command"]
        timeout = int(inp.get("timeout_seconds", 60))

        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(command):
                _audit(audit_path, command, "BLOCKED", "matched dangerous-pattern denylist")
                return "Error: command blocked by safety denylist (looked destructive to a whole drive/filesystem)."

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            outcome = f"exit_code={proc.returncode}"
            _audit(audit_path, command, outcome, "")
            output = (proc.stdout or "") + (proc.stderr or "")
            output = output[:8000]
            return f"exit_code: {proc.returncode}\n{output}"
        except subprocess.TimeoutExpired:
            _audit(audit_path, command, "TIMEOUT", f"exceeded {timeout}s")
            return f"Error: command timed out after {timeout}s"

    return run_command


def _audit(audit_path: Path, command: str, outcome: str, note: str) -> None:
    line = json.dumps(
        {"ts": datetime.now(timezone.utc).isoformat(), "command": command, "outcome": outcome, "note": note}
    )
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _make_trading_tools(repo_root: Path, cfg: dict) -> tuple[Callable, Callable, Callable]:
    from trading_bot.trade_log import TradeLog

    pid_path = repo_root / "state" / "trading_bot.pid"
    trade_log = TradeLog(repo_root / cfg["assistant"]["trade_log_path"])

    def start(_: dict[str, Any]) -> str:
        if pid_path.exists():
            return "Trading bot appears to already be running (pid file exists). Use trading_bot_status first."
        proc = subprocess.Popen(
            ["python", "-m", "trading_bot.main"],
            cwd=str(repo_root),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP") else 0,
        )
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(proc.pid), encoding="utf-8")
        return f"Started trading bot loop as PID {proc.pid}."

    def stop(_: dict[str, Any]) -> str:
        if not pid_path.exists():
            return "No PID file found; trading bot doesn't appear to be running (via this control path)."
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        finally:
            pid_path.unlink(missing_ok=True)
        return f"Stopped trading bot PID {pid}."

    def status(_: dict[str, Any]) -> str:
        running = "running (pid file present)" if pid_path.exists() else "not running (no pid file)"
        recent = trade_log.tail(10)
        return json.dumps({"process": running, "recent_trades": recent}, indent=2, default=str)

    return start, stop, status


def build_registry(repo_root: Path, cfg: dict, memory) -> ToolRegistry:
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
            name="write_file",
            description="Write (create or overwrite) a UTF-8 text file, creating parent directories as needed.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=_write_file,
        )
    )
    registry.register(
        Tool(
            name="run_command",
            description=(
                "Execute a shell command on the user's local Windows machine and return its "
                "stdout/stderr/exit code. Every call is logged to an audit file. Obviously "
                "destructive whole-drive commands are blocked. Use for builds, tests, git, "
                "launching tools, querying system state, etc."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "default": 60},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            handler=_make_run_command(repo_root / cfg["assistant"]["command_audit_path"]),
        )
    )
    registry.register(
        Tool(
            name="remember",
            description="Persist a durable fact about the user or ongoing work for future sessions.",
            input_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["key", "value"],
                "additionalProperties": False,
            },
            handler=lambda inp: (memory.remember_fact(inp["key"], inp["value"]), f"Remembered '{inp['key']}'.")[1],
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

    start, stop, status = _make_trading_tools(repo_root, cfg)
    registry.register(
        Tool(
            name="trading_bot_start",
            description="Start the forex trading bot as a background process (paper or live per config.yaml).",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=start,
        )
    )
    registry.register(
        Tool(
            name="trading_bot_stop",
            description="Stop the background trading bot process.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=stop,
        )
    )
    registry.register(
        Tool(
            name="trading_bot_status",
            description="Get the trading bot process status and its most recent trade log entries.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=status,
        )
    )
    registry.register(
        Tool(
            name="list_proposals",
            description=(
                "List things the background daemon (scripts/run_daemon.py) noticed while running "
                "unattended but did not act on itself — e.g. overnight monitoring findings or "
                "suggested code changes. Use this when the user asks what happened overnight or "
                "what's pending their review. Also points at the latest overnight summary file, if any."
            ),
            input_schema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 20}},
                "additionalProperties": False,
            },
            handler=_make_list_proposals(repo_root, cfg),
        )
    )

    return registry


def _make_list_proposals(repo_root: Path, cfg: dict) -> Callable[[dict[str, Any]], str]:
    from assistant.daemon.proposals import ProposalLog

    daemon_cfg = cfg.get("daemon", {})
    proposal_log = ProposalLog(repo_root / daemon_cfg.get("proposals_path", "logs/proposals.jsonl"))
    logs_dir = repo_root / "logs"

    def list_proposals(inp: dict[str, Any]) -> str:
        proposals = proposal_log.tail(int(inp.get("limit", 20)))
        summaries = sorted(logs_dir.glob("summary_*.md")) if logs_dir.exists() else []
        latest_summary = str(summaries[-1]) if summaries else None
        return json.dumps(
            {"pending_proposals": proposals, "latest_overnight_summary_file": latest_summary},
            indent=2,
            default=str,
        )

    return list_proposals
