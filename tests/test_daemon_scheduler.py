from __future__ import annotations

from datetime import datetime

from assistant.daemon.scheduler import RoutineSpec, Scheduler, is_due


def _noop(ctx) -> None:
    pass


def test_interval_routine_due_when_never_run():
    spec = RoutineSpec("r", _noop, interval_s=900)
    assert is_due(spec, last_run=None, now=datetime(2026, 1, 1, 12, 0)) is True


def test_interval_routine_not_due_before_interval_elapses():
    spec = RoutineSpec("r", _noop, interval_s=900)
    last_run = datetime(2026, 1, 1, 12, 0)
    now = datetime(2026, 1, 1, 12, 10)  # only 10 minutes later
    assert is_due(spec, last_run, now) is False


def test_interval_routine_due_after_interval_elapses():
    spec = RoutineSpec("r", _noop, interval_s=900)
    last_run = datetime(2026, 1, 1, 12, 0)
    now = datetime(2026, 1, 1, 12, 15)  # exactly 15 minutes later
    assert is_due(spec, last_run, now) is True


def test_daily_routine_not_due_before_scheduled_time():
    spec = RoutineSpec("r", _noop, daily_at="07:00")
    now = datetime(2026, 1, 1, 6, 59)
    assert is_due(spec, last_run=None, now=now) is False


def test_daily_routine_due_at_scheduled_time_if_not_run_today():
    spec = RoutineSpec("r", _noop, daily_at="07:00")
    now = datetime(2026, 1, 1, 7, 0)
    assert is_due(spec, last_run=None, now=now) is True


def test_daily_routine_not_due_again_same_day():
    spec = RoutineSpec("r", _noop, daily_at="07:00")
    last_run = datetime(2026, 1, 1, 7, 0)
    now = datetime(2026, 1, 1, 20, 0)
    assert is_due(spec, last_run, now) is False


def test_daily_routine_due_again_next_day():
    spec = RoutineSpec("r", _noop, daily_at="07:00")
    last_run = datetime(2026, 1, 1, 7, 0)
    now = datetime(2026, 1, 2, 7, 0)
    assert is_due(spec, last_run, now) is True


def test_one_failing_routine_does_not_stop_others():
    calls: list[str] = []

    def boom(ctx) -> None:
        raise RuntimeError("kaboom")

    def works(ctx) -> None:
        calls.append("ran")

    specs = [RoutineSpec("boom", boom, interval_s=1), RoutineSpec("works", works, interval_s=1)]
    scheduler = Scheduler(specs, ctx=None)

    scheduler.run_once()

    assert calls == ["ran"]
    assert "boom" in scheduler.last_run
    assert "works" in scheduler.last_run
