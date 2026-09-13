from __future__ import annotations

from assistant.voice.voice_assistant import extract_command


def test_extract_command_wake_word_only():
    woke, command = extract_command("alex", "alex")
    assert woke is True
    assert command == ""


def test_extract_command_wake_word_with_trailing_command():
    woke, command = extract_command("alex what's my trading bot's status", "alex")
    assert woke is True
    assert command == "what's my trading bot's status"


def test_extract_command_is_case_insensitive():
    woke, command = extract_command("Hey Alex, start the bot", "alex")
    assert woke is True
    assert command == "start the bot"


def test_extract_command_no_wake_word():
    woke, command = extract_command("start the bot", "alex")
    assert woke is False
    assert command == ""


def test_extract_command_does_not_match_substring():
    # "alexander" contains "alex" but should not trigger on a bare substring match.
    woke, _ = extract_command("alexander graham bell", "alex")
    assert woke is False
