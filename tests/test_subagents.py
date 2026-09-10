from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from assistant.subagents import run_subagent
from assistant.tools import Tool, ToolRegistry


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeResponse:
    content: list[Any]
    stop_reason: str


class FakeMessages:
    def __init__(self, responses: list[FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        kwargs["messages"] = list(kwargs["messages"])
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]):
        self.messages = FakeMessages(responses)


def _registry_with_spawn_subagent() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="spawn_subagent",
            description="delegate",
            input_schema={"type": "object", "properties": {}},
            handler=lambda inp: "should never be reachable from a subagent",
        )
    )
    registry.register(
        Tool(
            name="echo",
            description="echoes input",
            input_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
            handler=lambda inp: f"echo: {inp['msg']}",
        )
    )
    return registry


def test_subagent_excludes_spawn_subagent_from_its_own_tools():
    registry = _registry_with_spawn_subagent()
    client = FakeClient([FakeResponse(content=[FakeTextBlock("ok")], stop_reason="end_turn")])

    run_subagent(client, registry, "claude-opus-5", "researcher", "look into X")

    tools_offered = client.messages.calls[0]["tools"]
    tool_names = {t["name"] for t in tools_offered}
    assert "spawn_subagent" not in tool_names
    assert "echo" in tool_names


def test_subagent_runs_tool_then_returns_final_text():
    client = FakeClient(
        [
            FakeResponse(
                content=[FakeToolUseBlock(id="t1", name="echo", input={"msg": "hi"})],
                stop_reason="tool_use",
            ),
            FakeResponse(content=[FakeTextBlock("final answer")], stop_reason="end_turn"),
        ]
    )
    registry = _registry_with_spawn_subagent()

    result = run_subagent(client, registry, "claude-opus-5", "coder", "do a thing")

    assert result == "final answer"
    assert len(client.messages.calls) == 2


def test_subagent_uses_correct_system_prompt_per_role():
    client = FakeClient([FakeResponse(content=[FakeTextBlock("ok")], stop_reason="end_turn")])
    registry = _registry_with_spawn_subagent()

    run_subagent(client, registry, "claude-opus-5", "coder", "task")

    assert "coding subagent" in client.messages.calls[0]["system"]


def test_subagent_unknown_role_falls_back_to_general():
    client = FakeClient([FakeResponse(content=[FakeTextBlock("ok")], stop_reason="end_turn")])
    registry = _registry_with_spawn_subagent()

    run_subagent(client, registry, "claude-opus-5", "not-a-real-role", "task")

    assert "general-purpose subagent" in client.messages.calls[0]["system"]


def test_subagent_returns_placeholder_when_no_text_produced():
    client = FakeClient([FakeResponse(content=[], stop_reason="end_turn")])
    registry = _registry_with_spawn_subagent()

    result = run_subagent(client, registry, "claude-opus-5", "general", "task")

    assert result == "(subagent produced no text output)"


def test_subagent_hits_iteration_limit():
    responses = [
        FakeResponse(
            content=[FakeToolUseBlock(id=f"t{i}", name="echo", input={"msg": "x"})],
            stop_reason="tool_use",
        )
        for i in range(15)
    ]
    client = FakeClient(responses)
    registry = _registry_with_spawn_subagent()

    result = run_subagent(client, registry, "claude-opus-5", "general", "loop forever")

    assert "iteration limit" in result
