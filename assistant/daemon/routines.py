"""Fixed daemon routines. Each is a plain function(context) -> None so the
scheduler can run them on independent intervals. Anything that would change
state is written to the proposal log instead of being executed directly —
see assistant/daemon/restricted_tools.py for the enforcement point.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import anthropic

from assistant.daemon.proposals import ProposalLog
from assistant.daemon.restricted_tools import build_readonly_registry
from trading_bot.trade_log import TradeLog

logger = logging.getLogger("assistant.daemon")


@dataclass
class RoutineContext:
    repo_root: Path
    cfg: dict
    memory: object
    proposal_log: ProposalLog
    client: anthropic.Anthropic | None


def trading_bot_health_check(ctx: RoutineContext) -> None:
    pid_path = ctx.repo_root / "state" / "trading_bot.pid"
    trade_log = TradeLog(ctx.repo_root / ctx.cfg["assistant"]["trade_log_path"])
    recent = trade_log.tail(5)

    if not pid_path.exists():
        logger.info("trading_bot_health_check: bot not running (no pid file) — nothing to check.")
        return

    logger.info("trading_bot_health_check: running, %d recent trade log entries.", len(recent))
    errors = [r for r in recent if r.get("status") not in (None, "filled", "simulated")]
    if errors:
        ctx.proposal_log.record(
            routine="trading_bot_health_check",
            title="Trading bot has recent non-filled/simulated trade entries",
            description=f"Last 5 trade log entries include unexpected statuses: {errors}",
            suggested_action="Review state/trades.jsonl and the trading bot's own logs before its next cycle.",
        )


def dev_agent_routine(ctx: RoutineContext) -> None:
    project_path = ctx.cfg.get("daemon", {}).get("dev_project_path")
    if not project_path:
        logger.info("dev_agent_routine: no daemon.dev_project_path configured — skipping.")
        return

    repo = Path(project_path)
    if not repo.is_dir():
        logger.warning("dev_agent_routine: configured dev_project_path %s does not exist — skipping.", repo)
        return

    if ctx.client is None:
        logger.warning("dev_agent_routine: no ANTHROPIC_API_KEY configured — skipping (repo checks need Claude).")
        return

    def run(cmd: list[str]) -> str:
        try:
            proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, timeout=120)
            return f"$ {' '.join(cmd)}\nexit_code={proc.returncode}\n{proc.stdout}{proc.stderr}"
        except Exception as exc:  # subprocess failures must not kill the daemon
            return f"$ {' '.join(cmd)}\nError: {type(exc).__name__}: {exc}"

    report = "\n\n".join(
        [
            run(["git", "status", "--short"]),
            run(["git", "diff", "--stat"]),
            run(["pytest", "-q"]),
        ]
    )
    logger.info("dev_agent_routine: gathered repo status for %s.", repo)

    registry = build_readonly_registry(ctx.repo_root, ctx.cfg, ctx.memory, ctx.proposal_log)
    system_prompt = (
        "You are Alex's Dev Agent, investigating a software project unattended overnight. "
        "You have read-only tools. You CANNOT write files or run commands yourself. If you find "
        "something worth fixing or changing, call propose_change with a concrete suggested_action "
        "the user can run/apply themselves tomorrow. If everything looks fine, say so briefly and "
        "don't propose anything."
    )
    messages = [
        {
            "role": "user",
            "content": f"Here is the current state of the project at {repo}:\n\n{report}\n\n"
            "Investigate and propose any changes worth making.",
        }
    ]
    tools = registry.as_api_tools()
    for _ in range(8):
        response = ctx.client.messages.create(
            model=ctx.cfg["assistant"]["model"],
            max_tokens=4000,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )
        if response.stop_reason != "tool_use":
            texts = [b.text for b in response.content if b.type == "text"]
            logger.info("dev_agent_routine: %s", "\n".join(texts) or "(no text output)")
            return
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = registry.execute(block.name, block.input)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": tool_results})
    logger.warning("dev_agent_routine: hit its iteration limit before finishing.")


def overnight_summary(ctx: RoutineContext) -> None:
    if ctx.client is None:
        logger.warning("overnight_summary: no ANTHROPIC_API_KEY configured — skipping.")
        return

    log_path = ctx.repo_root / ctx.cfg["daemon"]["log_path"]
    proposals = ctx.proposal_log.tail(50)
    trade_log = TradeLog(ctx.repo_root / ctx.cfg["assistant"]["trade_log_path"])
    recent_trades = trade_log.tail(20)

    log_tail = ""
    if log_path.exists():
        log_tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-200:])

    prompt = (
        "Write a short, plain-English overnight summary for the user of what the background "
        "daemon did while they were away. Be concise — a few sentences plus a bullet list of "
        "anything that needs their attention (proposals). If nothing happened, say that plainly.\n\n"
        f"Daemon log (tail):\n{log_tail or '(empty)'}\n\n"
        f"Pending proposals ({len(proposals)}):\n{proposals}\n\n"
        f"Recent trade log entries ({len(recent_trades)}):\n{recent_trades}\n"
    )
    response = ctx.client.messages.create(
        model=ctx.cfg["assistant"]["model"],
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    texts = [b.text for b in response.content if b.type == "text"]
    summary = "\n".join(texts) or "(summary generation produced no text)"

    out_path = ctx.repo_root / "logs" / f"summary_{date.today().isoformat()}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"# Overnight summary — {datetime.now(timezone.utc).isoformat()}\n\n{summary}\n", encoding="utf-8")
    logger.info("overnight_summary: wrote %s", out_path)


RoutineFn = Callable[[RoutineContext], None]

ROUTINES: dict[str, RoutineFn] = {
    "trading_bot_health_check": trading_bot_health_check,
    "dev_agent_routine": dev_agent_routine,
    "overnight_summary": overnight_summary,
}
