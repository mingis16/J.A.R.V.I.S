"""Minimal Telegram Bot API client.

Uses httpx (already a dependency via the anthropic SDK) rather than
urllib.request — on this machine, urllib's long-held HTTPS connections to
api.telegram.org were reset/hung (likely local AV/firewall inspection
behaving differently per-process/per-connection-pattern), while httpx and
curl both worked fine.
"""
from __future__ import annotations

import socket
from typing import Any

import httpx

API_URL = "https://api.telegram.org/bot{token}/{method}"

_dns_cache: dict[tuple, Any] = {}
_real_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_with_fallback(host, port, *args, **kwargs):
    key = (host, port, args, tuple(sorted(kwargs.items())))
    try:
        result = _real_getaddrinfo(host, port, *args, **kwargs)
    except socket.gaierror:
        if key in _dns_cache:
            return _dns_cache[key]
        raise
    _dns_cache[key] = result
    return result


def install_dns_fallback() -> None:
    """Reuse the last good DNS answer when a lookup fails.

    This machine's network DNS (a carrier/hotspot resolver; public resolvers
    are blocked) intermittently fails api.telegram.org lookups with
    getaddrinfo errors while the connection itself works — it took the bot
    offline for minutes at a time. TLS still verifies the certificate against
    the hostname, so a stale cached address can only fail, never be spoofed.
    """
    socket.getaddrinfo = _getaddrinfo_with_fallback


class TelegramClient:
    def __init__(self, token: str):
        self.token = token
        self._client = httpx.Client()

    def _call(self, method: str, http_timeout: float = 10.0, **params: Any) -> dict[str, Any]:
        url = API_URL.format(token=self.token, method=method)
        resp = self._client.get(url, params=params, timeout=http_timeout + 10)
        resp.raise_for_status()
        return resp.json()

    def get_updates(self, offset: int | None = None, timeout: int = 25) -> dict[str, Any]:
        params: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        return self._call("getUpdates", http_timeout=timeout, **params)

    def send_message(self, chat_id: int, text: str) -> dict[str, Any]:
        return self._call("sendMessage", chat_id=chat_id, text=text)
