"""Persistent desk state, split by writer so processes never clobber each
other: control (goal, paused) is written by the user via chat-Alex; day stats
(day-start balance, trade/cycle counts) are written only by the daemon's
desk cycle. Both survive restarts, so a reboot can't reset the daily stop.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DeskControl:
    goal: str = ""
    paused: bool = False
    updated_at: str = ""


@dataclass
class DeskDay:
    day: str = ""
    day_start_balance: float = 0.0
    trades_today: int = 0
    cycles_today: int = 0
    mode: str = ""  # "paper" or "live" — the balance the anchor was taken from


def _load(path: Path, cls: type) -> Any:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return cls()
    fields = cls.__dataclass_fields__
    return cls(**{k: v for k, v in data.items() if k in fields})


def _save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(obj), indent=2), encoding="utf-8")
    tmp.replace(path)


class DeskStateStore:
    def __init__(self, state_dir: Path):
        self.control_path = state_dir / "trading_desk_control.json"
        self.day_path = state_dir / "trading_desk_day.json"

    def control(self) -> DeskControl:
        return _load(self.control_path, DeskControl)

    def update_control(self, **changes: Any) -> DeskControl:
        control = self.control()
        for key, value in changes.items():
            setattr(control, key, value)
        control.updated_at = datetime.now(timezone.utc).isoformat()
        _save(self.control_path, control)
        return control

    def day(self) -> DeskDay:
        return _load(self.day_path, DeskDay)

    def save_day(self, day: DeskDay) -> None:
        _save(self.day_path, day)


def roll_day(day: DeskDay, today: str, balance: float, mode: str) -> DeskDay:
    """New UTC day, or a paper<->live switch -> reset counters and anchor the
    daily stop on the current balance (a paper anchor of 1000 against a live
    equity of 11 read as a 98.9% loss and tripped the stop on 2026-10-02).
    Also re-anchors if the anchor is still 0 (e.g. the account was funded
    after the day started), since a 0 anchor would disable the stop."""
    if day.day != today or day.mode != mode:
        return DeskDay(day=today, day_start_balance=balance, mode=mode)
    if day.day_start_balance <= 0 < balance:
        return DeskDay(
            day=today, day_start_balance=balance, trades_today=day.trades_today,
            cycles_today=day.cycles_today, mode=mode,
        )
    return day
