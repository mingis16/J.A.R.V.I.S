from __future__ import annotations

from assistant.telegram.bot import (
    format_trade_message,
    new_trades_since,
    parse_allowed_ids,
    split_message,
)


def test_parse_allowed_ids_handles_commas_and_whitespace():
    assert parse_allowed_ids("123, 456 ,789") == {123, 456, 789}


def test_parse_allowed_ids_empty_string_is_empty_set():
    assert parse_allowed_ids("") == set()


def test_split_message_under_limit_returns_single_chunk():
    assert split_message("hello", limit=100) == ["hello"]


def test_split_message_splits_long_text():
    text = "a" * 250
    chunks = split_message(text, limit=100)
    assert chunks == ["a" * 100, "a" * 100, "a" * 50]


def test_new_trades_since_returns_only_new_entries():
    entries = [{"id": 1}, {"id": 2}, {"id": 3}]
    assert new_trades_since(entries, last_count=1) == [{"id": 2}, {"id": 3}]


def test_new_trades_since_no_new_entries():
    entries = [{"id": 1}, {"id": 2}]
    assert new_trades_since(entries, last_count=2) == []


def test_format_trade_message_includes_key_fields():
    entry = {
        "symbol": "EURUSD",
        "action": "buy",
        "mode": "paper",
        "lots": 0.1,
        "price": 1.085,
        "sl": 1.08,
        "tp": 1.09,
        "reason": "bullish EMA cross",
    }
    msg = format_trade_message(entry)
    assert "EURUSD" in msg
    assert "buy" in msg
    assert "bullish EMA cross" in msg
