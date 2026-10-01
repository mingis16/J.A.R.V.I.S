from __future__ import annotations

import json
import os
import subprocess
import sys

from assistant.daemon.proposals import ProposalLog
from assistant.daemon.routines import RoutineContext, trading_bot_health_check
from assistant.tools import _make_trading_tools
from trading_bot.pidfile import pid_alive, read_live_pid, remove_pid_if_owned, write_pid


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_pid_alive_true_for_current_process():
    assert pid_alive(os.getpid()) is True


def test_pid_alive_false_for_exited_process():
    assert pid_alive(_dead_pid()) is False


def test_read_live_pid_returns_pid_of_running_process(tmp_path):
    path = tmp_path / "bot.pid"
    write_pid(path)
    assert read_live_pid(path) == os.getpid()
    assert path.exists()


def test_read_live_pid_clears_stale_file(tmp_path):
    path = tmp_path / "bot.pid"
    write_pid(path, _dead_pid())
    assert read_live_pid(path) is None
    assert not path.exists()


def test_remove_pid_if_owned_leaves_another_processes_file(tmp_path):
    path = tmp_path / "bot.pid"
    write_pid(path, os.getpid() + 1)
    remove_pid_if_owned(path)
    assert path.exists()

    write_pid(path)
    remove_pid_if_owned(path)
    assert not path.exists()


def test_trading_status_reports_not_running_for_stale_pid(tmp_path):
    cfg = {"assistant": {"trade_log_path": "state/trades.jsonl"}}
    pid_path = tmp_path / "state" / "trading_bot.pid"
    write_pid(pid_path, _dead_pid())
    _, _, status = _make_trading_tools(tmp_path, cfg)

    result = json.loads(status({}))

    assert result["process"] == "not running"
    assert not pid_path.exists()


def test_health_check_flags_dead_bot_and_clears_stale_pid(tmp_path):
    cfg = {"assistant": {"trade_log_path": "state/trades.jsonl"}}
    pid_path = tmp_path / "state" / "trading_bot.pid"
    write_pid(pid_path, _dead_pid())
    proposal_log = ProposalLog(tmp_path / "logs" / "proposals.jsonl")
    ctx = RoutineContext(repo_root=tmp_path, cfg=cfg, memory=None, proposal_log=proposal_log, client=None)

    trading_bot_health_check(ctx)

    proposals = proposal_log.tail(10)
    assert [p["title"] for p in proposals] == ["Trading bot stopped unexpectedly"]
    assert not pid_path.exists()

    trading_bot_health_check(ctx)  # no PID file now -> no duplicate proposal
    assert len(proposal_log.tail(10)) == 1
