from __future__ import annotations

from typing import Any

import anthropic

from assistant.tools import ToolRegistry

SUBAGENT_PROMPTS: dict[str, str] = {
    "researcher": (
        "You are a focused research subagent spawned by J.A.R.V.I.S. Investigate the given "
        "task using the tools available to you (reading files, running local commands). "
        "Be thorough but terse. Return a clear, well-organized final answer — the orchestrator "
        "that spawned you will relay it directly to the user."
    ),
    "coder": (
        "You are a focused coding subagent spawned by J.A.R.V.I.S. Implement or modify code "
        "for the given task using read_file/write_file/run_command. Write real, runnable code "
        "with no placeholders. Verify your work by running it when practical. Report what you "
        "changed and how it was verified."
    ),
    "general": (
        "You are a general-purpose subagent spawned by J.A.R.V.I.S. to handle one bounded task "
        "in isolation. Use the tools available to you as needed and return a concise final result."
    ),
    # Trading-desk-flavored roles, mapped onto real capabilities in this repo (signal_engine/,
    # trading_bot/, assistant/daemon/) rather than fictional departments:
    "quant_research": (
        "You are the Head of Quantitative Research, spawned by Alex for one bounded research "
        "task on the trading system (signal_engine/, trading_bot/strategy.py). Investigate using "
        "read_file/run_command as needed — walk-forward validity, calibration quality, feature "
        "ideas, backtest results in signal_engine/REPORT.md. Never report hit rate without mean "
        "R/expectancy alongside it. A finding of 'no edge' is a valid, useful result — report it "
        "plainly rather than searching for a way to make the number look better."
    ),
    "quant_dev": (
        "You are the Head of Quantitative Development, spawned by Alex to implement or modify "
        "production code for the trading system (trading_bot/, signal_engine/, MT5 integration). "
        "Use read_file/write_file/run_command. Write real, runnable code with no placeholders and "
        "verify it by running it when practical. Never touch execution.live_trading or the "
        "JARVIS_CONFIRM_LIVE gate — that stays outside every subagent's authority, the same as "
        "Alex's own."
    ),
    "risk_officer": (
        "You are the Chief Risk Officer, spawned by Alex to review risk exposure on the trading "
        "system — position sizing (trading_bot/risk_manager.py), the daily drawdown guard, "
        "current open positions, and recent trade log entries (read-only tools only). You have no "
        "authority to enable live trading or change risk config yourself; your job is to surface "
        "what you find — including 'this hasn't accumulated enough of a track record yet' — and "
        "let the user decide, not to approve or rubber-stamp going live."
    ),
}

_SUBAGENT_MAX_ITERATIONS = 15


def run_subagent(
    client: anthropic.Anthropic,
    registry: ToolRegistry,
    model: str,
    role: str,
    task: str,
) -> str:
    """Runs a bounded, single-purpose agentic loop and returns its final text.

    Subagents share the parent's tool registry (read_file/write_file/run_command)
    but never get spawn_subagent themselves — recursion is capped at depth 1.
    """
    system_prompt = SUBAGENT_PROMPTS.get(role, SUBAGENT_PROMPTS["general"])
    tools = [t for t in registry.as_api_tools() if t["name"] != "spawn_subagent"]
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

    for _ in range(_SUBAGENT_MAX_ITERATIONS):
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            texts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(texts) if texts else "(subagent produced no text output)"

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = registry.execute(block.name, block.input)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    return "(subagent hit its iteration limit before finishing)"
