"""Telegram front-end for Alex: talk to it and get pushed trading signals
from your phone. Polling-based (getUpdates) — no public server/webhook, no
port to open, works from behind any NAT/firewall.

Security: this bot has the SAME shell/file access as the CLI and dashboard.
Only Telegram user IDs listed in TELEGRAM_ALLOWED_USER_IDS get real
responses; anyone else is silently ignored. If that list is empty, the bot
runs in "discovery mode" — it tells whoever messages it their own numeric
user ID (and nothing else) so the user can find their ID and lock things
down, then does not touch the orchestrator at all.
"""
from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Any

from assistant.orchestrator import Orchestrator
from assistant.telegram.client import TelegramClient
from trading_bot.trade_log import TradeLog

logger = logging.getLogger("assistant.telegram")

MESSAGE_LIMIT = 4000  # Telegram's real limit is 4096; leave headroom


def split_message(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    if not text:
        return [""]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def parse_allowed_ids(raw: str) -> set[int]:
    return {int(x) for x in raw.split(",") if x.strip()}


def new_trades_since(all_entries: list[dict[str, Any]], last_count: int) -> list[dict[str, Any]]:
    if last_count >= len(all_entries):
        return []
    return all_entries[last_count:]


def format_trade_message(entry: dict[str, Any]) -> str:
    return (
        f"📈 {entry.get('symbol')} {entry.get('action')} ({entry.get('mode')})\n"
        f"lots={entry.get('lots')} price={entry.get('price')} "
        f"sl={entry.get('sl')} tp={entry.get('tp')}\n"
        f"reason: {entry.get('reason')}"
    )


class TelegramBot:
    def __init__(self, repo_root: Path, cfg: dict, token: str, allowed_ids: set[int]):
        self.client = TelegramClient(token)
        self.allowed_ids = allowed_ids
        self.orchestrator = Orchestrator(repo_root, cfg) if allowed_ids else None
        self.trade_log = TradeLog(repo_root / cfg["assistant"]["trade_log_path"])
        self.notify_on_trade = cfg.get("telegram", {}).get("notify_on_trade", True)
        self._last_trade_count = len(self.trade_log.tail(10**9))
        self._offset: int | None = None
        self._shutdown = False

    def _handle_sigint(self, signum, frame) -> None:
        self._shutdown = True
        logger.info("Shutdown requested, will stop after this poll.")

    def _send(self, chat_id: int, text: str) -> None:
        for chunk in split_message(text):
            try:
                self.client.send_message(chat_id, chunk)
            except Exception:
                logger.exception("Failed to send Telegram message to chat_id=%s", chat_id)

    def _handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if not message or "text" not in message:
            return
        user_id = message.get("from", {}).get("id")
        chat_id = message["chat"]["id"]
        text = message["text"].strip()
        if not text:
            return

        if not self.allowed_ids:
            self._send(
                chat_id,
                f"Your Telegram user ID is {user_id}. Add it to TELEGRAM_ALLOWED_USER_IDS "
                "in .env and restart me to start chatting.",
            )
            return

        if user_id not in self.allowed_ids:
            logger.warning("Ignoring message from unauthorized Telegram user_id=%s", user_id)
            return

        logger.info("Telegram message from %s: %s", user_id, text)
        try:
            reply = self.orchestrator.chat(text)
        except Exception as exc:
            reply = f"Error talking to my brain: {type(exc).__name__}: {exc}"
        self._send(chat_id, reply)

    def _check_new_trades(self) -> None:
        if not self.notify_on_trade or not self.allowed_ids:
            return
        entries = self.trade_log.tail(10**9)
        for entry in new_trades_since(entries, self._last_trade_count):
            message = format_trade_message(entry)
            for uid in self.allowed_ids:
                self._send(uid, message)
        self._last_trade_count = len(entries)

    def run_forever(self) -> None:
        signal.signal(signal.SIGINT, self._handle_sigint)
        if self.allowed_ids:
            logger.info("Telegram bot started, allowed user IDs: %s", self.allowed_ids)
        else:
            logger.warning(
                "TELEGRAM_ALLOWED_USER_IDS is empty — running in discovery mode. "
                "Message the bot to find your user ID."
            )
        while not self._shutdown:
            try:
                updates = self.client.get_updates(offset=self._offset, timeout=25)
                for update in updates.get("result", []):
                    self._offset = update["update_id"] + 1
                    try:
                        self._handle_update(update)
                    except Exception:
                        logger.exception("Failed handling update %s", update.get("update_id"))
            except Exception:
                logger.exception("Telegram poll failed; retrying shortly.")
                time.sleep(5)
            self._check_new_trades()
        logger.info("Telegram bot stopped.")


def main() -> int:
    import os

    from trading_bot.config import REPO_ROOT, load_env, load_yaml_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN is not set. Add it to .env (see .env.example).")
        return 1

    cfg = load_yaml_config()
    allowed_ids = parse_allowed_ids(os.environ.get("TELEGRAM_ALLOWED_USER_IDS", ""))
    if allowed_ids and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example).")
        return 1

    bot = TelegramBot(REPO_ROOT, cfg, token, allowed_ids)
    bot.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
