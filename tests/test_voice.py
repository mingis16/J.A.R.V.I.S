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


def test_extract_command_tolerates_whisper_punctuation_inside_wake_phrase():
    # Real transcripts from logs/voice_stdout.log that used to be ignored.
    for transcript in ("Hey, Alex.", "Hey, Alex!", "hey - alex"):
        woke, command = extract_command(transcript, "hey alex")
        assert woke is True, transcript
        assert command == "", transcript


def test_extract_command_bare_punctuation_is_not_a_command():
    # "Hey Alex!" used to send "!" to the assistant as the command.
    woke, command = extract_command("Hey Alex!", "hey alex")
    assert woke is True
    assert command == ""


def test_extract_command_multiword_wake_word_with_command():
    woke, command = extract_command("Hey, Alex, what's going on?", "hey alex")
    assert woke is True
    assert command == "what's going on"
