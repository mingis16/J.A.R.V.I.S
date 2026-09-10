from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Memory:
    """JSON-backed conversation history + long-term key/value facts.

    Deliberately simple (no vector DB) so it has zero extra infra to run.
    Swap in a real store later if the fact set outgrows this.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {"facts": {}, "history": []}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        self._data.setdefault("facts", {})
        self._data.setdefault("history", [])

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def remember_fact(self, key: str, value: Any) -> None:
        self._data["facts"][key] = value
        self._save()

    def recall_fact(self, key: str) -> Any:
        return self._data["facts"].get(key)

    def all_facts(self) -> dict[str, Any]:
        return dict(self._data["facts"])

    def append_turn(self, role: str, content: Any) -> None:
        self._data["history"].append(
            {"role": role, "content": content, "ts": datetime.now(timezone.utc).isoformat()}
        )
        # Keep the on-disk log bounded; the live API history is managed separately in-process.
        self._data["history"] = self._data["history"][-500:]
        self._save()

    def recent_history(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._data["history"][-limit:]
