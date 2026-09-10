from __future__ import annotations

from assistant.memory import Memory
from assistant.tools import ToolRegistry, Tool, _make_run_command, _read_file, _write_file


def test_write_then_read_file_roundtrip(tmp_path):
    path = tmp_path / "sub" / "note.txt"
    write_result = _write_file({"path": str(path), "content": "hello jarvis"})
    assert "Wrote" in write_result
    assert path.read_text(encoding="utf-8") == "hello jarvis"

    read_result = _read_file({"path": str(path)})
    assert read_result == "hello jarvis"


def test_read_file_missing_returns_error_not_exception():
    result = _read_file({"path": "C:/definitely/not/a/real/path.txt"})
    assert result.startswith("Error: file not found")


def test_run_command_executes_and_captures_output(tmp_path):
    audit_path = tmp_path / "audit.log"
    run_command = _make_run_command(audit_path)

    result = run_command({"command": "echo hello-jarvis"})

    assert "exit_code: 0" in result
    assert "hello-jarvis" in result
    assert audit_path.exists()


def test_run_command_blocks_dangerous_patterns(tmp_path):
    audit_path = tmp_path / "audit.log"
    run_command = _make_run_command(audit_path)

    result = run_command({"command": "rm -rf /"})

    assert result.startswith("Error: command blocked")
    assert "BLOCKED" in audit_path.read_text(encoding="utf-8")


def test_registry_execute_unknown_tool_returns_error_string():
    registry = ToolRegistry()
    result = registry.execute("no_such_tool", {})
    assert "unknown tool" in result


def test_registry_execute_catches_handler_exceptions():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="boom",
            description="raises",
            input_schema={"type": "object", "properties": {}},
            handler=lambda inp: (_ for _ in ()).throw(ValueError("kaboom")),
        )
    )
    result = registry.execute("boom", {})
    assert "ValueError" in result
    assert "kaboom" in result


def test_memory_remember_and_recall_persists_across_instances(tmp_path):
    path = tmp_path / "memory.json"
    mem1 = Memory(path)
    mem1.remember_fact("favorite_pair", "EURUSD")

    mem2 = Memory(path)
    assert mem2.recall_fact("favorite_pair") == "EURUSD"
    assert mem2.recall_fact("nonexistent") is None


def test_memory_history_is_bounded(tmp_path):
    path = tmp_path / "memory.json"
    mem = Memory(path)
    for i in range(510):
        mem.append_turn("user", f"turn {i}")
    assert len(mem.recent_history(limit=1000)) == 500
