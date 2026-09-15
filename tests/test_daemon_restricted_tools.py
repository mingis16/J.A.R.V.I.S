from __future__ import annotations

from assistant.daemon.proposals import ProposalLog
from assistant.daemon.restricted_tools import build_readonly_registry
from assistant.memory import Memory

CFG = {
    "assistant": {"trade_log_path": "state/trades.jsonl"},
}


def _build(tmp_path):
    memory = Memory(tmp_path / "memory.json")
    proposal_log = ProposalLog(tmp_path / "proposals.jsonl")
    return build_readonly_registry(tmp_path, CFG, memory, proposal_log)


def test_readonly_registry_excludes_write_and_execute_tools(tmp_path):
    registry = _build(tmp_path)
    names = {t["name"] for t in registry.as_api_tools()}

    for forbidden in ("write_file", "run_command", "trading_bot_start", "trading_bot_stop"):
        assert forbidden not in names


def test_readonly_registry_includes_read_and_propose_tools(tmp_path):
    registry = _build(tmp_path)
    names = {t["name"] for t in registry.as_api_tools()}

    assert {"read_file", "recall", "trading_bot_status", "propose_change"} <= names


def test_propose_change_tool_writes_to_proposal_log(tmp_path):
    proposal_log = ProposalLog(tmp_path / "proposals.jsonl")
    memory = Memory(tmp_path / "memory.json")
    registry = build_readonly_registry(tmp_path, CFG, memory, proposal_log)

    result = registry.execute(
        "propose_change",
        {
            "routine": "dev_agent_routine",
            "title": "Fix flaky test",
            "description": "test_foo intermittently fails",
            "suggested_action": "add a retry",
        },
    )

    assert "Proposal recorded" in result
    assert proposal_log.tail(1)[0]["title"] == "Fix flaky test"
