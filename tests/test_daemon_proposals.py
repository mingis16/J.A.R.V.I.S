from __future__ import annotations

from assistant.daemon.proposals import ProposalLog


def test_record_then_tail_roundtrip(tmp_path):
    log = ProposalLog(tmp_path / "proposals.jsonl")

    log.record(
        routine="trading_bot_health_check",
        title="Bot looks down",
        description="No pid file found.",
        suggested_action="python scripts/run_trading_bot.py",
    )

    entries = log.tail(10)
    assert len(entries) == 1
    assert entries[0]["routine"] == "trading_bot_health_check"
    assert entries[0]["title"] == "Bot looks down"
    assert "timestamp" in entries[0]


def test_tail_respects_limit_and_order(tmp_path):
    log = ProposalLog(tmp_path / "proposals.jsonl")
    for i in range(5):
        log.record(routine="r", title=f"t{i}", description="d", suggested_action="a")

    entries = log.tail(2)

    assert [e["title"] for e in entries] == ["t3", "t4"]


def test_tail_on_missing_file_returns_empty_list(tmp_path):
    log = ProposalLog(tmp_path / "does_not_exist.jsonl")
    assert log.tail() == []
