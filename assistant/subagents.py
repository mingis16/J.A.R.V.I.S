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
