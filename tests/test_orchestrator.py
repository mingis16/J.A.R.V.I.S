from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from assistant.orchestrator import Orchestrator


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
        # self.messages is mutated in place by the orchestrator after this call
        # returns, so snapshot it now or every stored call would alias the same,
        # eventually-fully-appended list.
        kwargs["messages"] = list(kwargs["messages"])
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeMessages.create called more times than responses were queued")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]):
        self.messages = FakeMessages(responses)


CFG = {
    "assistant": {
        "model": "claude-opus-5",
        "max_tokens": 1000,
        "memory_path": "state/memory.json",
        "trade_log_path": "state/trades.jsonl",
        "command_audit_path": "state/command_audit.log",
        "max_tool_iterations": 3,
    }
}


def _make_orchestrator(tmp_path, responses: list[FakeResponse], monkeypatch) -> Orchestrator:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    orch = Orchestrator(tmp_path, CFG)
    orch.client = FakeClient(responses)
    return orch


def test_chat_returns_text_on_first_turn(tmp_path, monkeypatch):
    responses = [FakeResponse(content=[FakeTextBlock("hello there")], stop_reason="end_turn")]
    orch = _make_orchestrator(tmp_path, responses, monkeypatch)

    reply = orch.chat("hi")

    assert reply == "hello there"
    assert orch.client.messages.calls[0]["messages"][0] == {"role": "user", "content": "hi"}


def test_chat_executes_tool_then_returns_final_text(tmp_path, monkeypatch):
    responses = [
        FakeResponse(
            content=[FakeToolUseBlock(id="t1", name="recall", input={})],
            stop_reason="tool_use",
        ),
        FakeResponse(content=[FakeTextBlock("done")], stop_reason="end_turn"),
    ]
    orch = _make_orchestrator(tmp_path, responses, monkeypatch)

    reply = orch.chat("remember stuff?")

    assert reply == "done"
    assert len(orch.client.messages.calls) == 2
    # second call must carry the tool_result threaded back in
    second_call_messages = orch.client.messages.calls[1]["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "t1"


def test_chat_stops_on_refusal(tmp_path, monkeypatch):
    responses = [FakeResponse(content=[], stop_reason="refusal")]
    orch = _make_orchestrator(tmp_path, responses, monkeypatch)

    reply = orch.chat("do something bad")

    assert reply == "I can't help with that request."


def test_chat_hits_iteration_limit(tmp_path, monkeypatch):
    # CFG caps max_tool_iterations at 3; always return tool_use so it never resolves.
    responses = [
        FakeResponse(content=[FakeToolUseBlock(id=f"t{i}", name="recall", input={})], stop_reason="tool_use")
        for i in range(3)
    ]
    orch = _make_orchestrator(tmp_path, responses, monkeypatch)

    reply = orch.chat("loop forever")

    assert "iteration limit" in reply
    assert len(orch.client.messages.calls) == 3


def test_chat_persists_turns_to_memory(tmp_path, monkeypatch):
    responses = [FakeResponse(content=[FakeTextBlock("hi back")], stop_reason="end_turn")]
    orch = _make_orchestrator(tmp_path, responses, monkeypatch)

    orch.chat("hello")

    history = orch.memory.recent_history()
    assert history[-2] == {"role": "user", "content": "hello", "ts": history[-2]["ts"]}
    assert history[-1]["content"] == "hi back"
