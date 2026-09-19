"""Append-only record of every live signal_engine check — whether or not it
produced a signal. This is what makes the "how accurate were the signals"
question answerable at the end of the week: a complete log, not just the
hits. Mirrors trading_bot/trade_log.py's TradeLog pattern.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class LiveCheck:
    checked_at_utc: str
    pair: str
    has_signal: bool
    signal: dict[str, Any] | None


class LiveCheckLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, pair: str, signal: dict[str, Any] | None) -> LiveCheck:
        check = LiveCheck(
            checked_at_utc=datetime.now(timezone.utc).isoformat(),
            pair=pair,
            has_signal=signal is not None,
            signal=signal,
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(check)) + "\n")
        return check

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return []
        return [json.loads(line) for line in lines[-n:]]

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines if line]
