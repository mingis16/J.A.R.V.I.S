from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from trading_bot.mt5_adapter import utc_now

Mode = Literal["paper", "live"]


@dataclass
class TradeRecord:
    timestamp: str
    symbol: str
    action: str
    mode: Mode
    lots: float
    price: float
    sl: float | None
    tp: float | None
    reason: str
    order_id: int | None = None
    status: str = "filled"
    extra: dict[str, Any] | None = None


class TradeLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trade: TradeRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(trade)) + "\n")

    def new_record(self, **kwargs: Any) -> TradeRecord:
        kwargs.setdefault("timestamp", utc_now().isoformat())
        return TradeRecord(**kwargs)

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines[-n:]]
