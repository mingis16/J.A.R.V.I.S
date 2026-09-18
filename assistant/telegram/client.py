"""Minimal Telegram Bot API client — plain HTTP, no extra dependency needed."""
from __future__ import annotations

import json
import urllib.request
from typing import Any

API_URL = "https://api.telegram.org/bot{token}/{method}"


class TelegramClient:
    def __init__(self, token: str):
        self.token = token

    def _call(self, method: str, **params: Any) -> dict[str, Any]:
        url = API_URL.format(token=self.token, method=method)
        data = json.dumps(params).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=params.get("timeout", 10) + 10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_updates(self, offset: int | None = None, timeout: int = 25) -> dict[str, Any]:
        params: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        return self._call("getUpdates", **params)

    def send_message(self, chat_id: int, text: str) -> dict[str, Any]:
        return self._call("sendMessage", chat_id=chat_id, text=text)
