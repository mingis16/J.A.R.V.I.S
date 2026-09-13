from __future__ import annotations

from typing import Any

import anthropic

from assistant.memory import Memory
from assistant.subagents import run_subagent
from assistant.tools import Tool, ToolRegistry, build_registry

SYSTEM_PROMPT = """\
You are Alex, the user's personal AI assistant and orchestrator, running locally on their \
machine (this project is called J.A.R.V.I.S., but your name is Alex). You have real tools: \
you can read and write files, run shell commands on their Windows PC, remember durable facts \
across sessions, control a forex trading bot, and spawn focused subagents (researcher / coder \
/ general) for bounded subtasks. You may also be talked to by voice — keep replies conversational \
and reasonably short, since long replies get read aloud via text-to-speech.

Ground rules:
- Be direct and useful. Don't pad responses with filler or fake enthusiasm.
- Use tools whenever they get a better answer than guessing — don't claim to have done \
something you didn't actually call a tool for.
- The trading bot defaults to PAPER (simulated) mode. It will only place real orders if the \
user has explicitly set execution.live_trading: true in config/config.yaml AND the \
JARVIS_CONFIRM_LIVE environment variable, which only the user controls outside this chat. \
Never tell the user you've enabled live trading yourself — you can't; that gate is by design \
outside your reach.
- run_command executes real shell commands on the user's machine. Prefer the least \
destructive command that answers the question. For anything that deletes, overwrites, force- \
pushes, or otherwise can't be undone, tell the user what you're about to run and why before \
running it, unless they've already told you to just proceed.
- If a task is large or independent enough to isolate (research a topic, write a script, \
investigate a bug in one file), consider spawn_subagent instead of doing it all inline.
"""


class Orchestrator:
    def __init__(self, repo_root, cfg: dict):
        self.cfg = cfg
        self.client = anthropic.Anthropic()
        self.model = cfg["assistant"]["model"]
        self.max_tokens = cfg["assistant"]["max_tokens"]
        self.max_iterations = cfg["assistant"]["max_tool_iterations"]
        self.memory = Memory(repo_root / cfg["assistant"]["memory_path"])
        self.registry = build_registry(repo_root, cfg, self.memory)
        self._register_spawn_subagent()
        self.messages: list[dict[str, Any]] = []

    def _register_spawn_subagent(self) -> None:
        def handler(inp: dict[str, Any]) -> str:
            role = inp.get("role", "general")
            task = inp["task"]
            return run_subagent(self.client, self.registry, self.model, role, task)

        self.registry.register(
            Tool(
                name="spawn_subagent",
                description=(
                    "Delegate a bounded, self-contained task to a fresh subagent with its own "
                    "context window. Use for research, coding, or investigation subtasks that "
                    "don't need the ongoing conversation's context. Returns the subagent's final answer."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["researcher", "coder", "general"]},
                        "task": {"type": "string", "description": "Full, self-contained task description."},
                    },
                    "required": ["task"],
                    "additionalProperties": False,
                },
                handler=handler,
            )
        )

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        self.memory.append_turn("user", user_message)

        tools = self.registry.as_api_tools()
        final_text = ""

        for _ in range(self.max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=tools,
                output_config={"effort": "high"},
                messages=self.messages,
            )

            if response.stop_reason == "refusal":
                final_text = "I can't help with that request."
                break

            if response.stop_reason != "tool_use":
                texts = [b.text for b in response.content if b.type == "text"]
                final_text = "\n".join(texts)
                self.messages.append({"role": "assistant", "content": response.content})
                break

            self.messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self.registry.execute(block.name, block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
            self.messages.append({"role": "user", "content": tool_results})
        else:
            final_text = final_text or "(hit the tool-call iteration limit before finishing — ask me to continue.)"

        self.memory.append_turn("assistant", final_text)
        return final_text
