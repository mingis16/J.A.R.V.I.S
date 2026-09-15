"""Append-only log of things the unattended daemon thinks should change but
did not do itself — the user reviews these and, if they agree, tells Alex
normally to carry them out. Mirrors trading_bot/trade_log.py's TradeLog.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Proposal:
    timestamp: str
    routine: str
    title: str
    description: str
    suggested_action: str


class ProposalLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, routine: str, title: str, description: str, suggested_action: str) -> Proposal:
        proposal = Proposal(
            timestamp=datetime.now(timezone.utc).isoformat(),
            routine=routine,
            title=title,
            description=description,
            suggested_action=suggested_action,
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(proposal)) + "\n")
        return proposal

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return []
        return [json.loads(line) for line in lines[-n:]]
