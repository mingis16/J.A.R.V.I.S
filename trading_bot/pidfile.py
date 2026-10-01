"""PID-file helpers for the trading bot's poll loop.

A PID file's mere existence doesn't mean the bot is running — a crash, a
reboot, or a killed terminal leaves it behind. Everything that asks "is the
bot running?" (Alex's trading tools, the daemon health check) goes through
read_live_pid(), which checks the process is actually alive and clears the
file if it isn't.
"""
from __future__ import annotations

import os
from pathlib import Path


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is NOT a liveness probe on Windows (signal 0 is
        # CTRL_C_EVENT), so ask the kernel for the process's exit code instead.
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def read_live_pid(path: Path) -> int | None:
    """PID from the file if that process is alive; otherwise removes the
    stale file and returns None."""
    pid = read_pid(path)
    if pid is not None and pid_alive(pid):
        return pid
    path.unlink(missing_ok=True)
    return None


def write_pid(path: Path, pid: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid if pid is not None else os.getpid()), encoding="utf-8")


def remove_pid_if_owned(path: Path, pid: int | None = None) -> None:
    """Only removes the file if it still names this process, so a newer bot
    instance's PID file isn't deleted by an older one shutting down."""
    if read_pid(path) == (pid if pid is not None else os.getpid()):
        path.unlink(missing_ok=True)
