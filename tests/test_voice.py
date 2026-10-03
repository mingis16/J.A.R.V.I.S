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


def test_extract_command_accepts_other_greetings_and_bare_name():
    # "Hi Alex" was silently ignored when only "hey alex" matched.
    for transcript, expected in [
        ("Hi Alex", ""),
        ("Hello, Alex, what's the time?", "what's the time"),
        ("Okay Alex check the trading desk", "check the trading desk"),
        ("Alex, are you there?", "are you there"),
        ("Alec are you there", "are you there"),  # common mishearing
    ]:
        woke, command = extract_command(transcript, "hey alex")
        assert woke is True, transcript
        assert command == expected, transcript


def test_extract_command_ignores_the_name_mid_sentence_without_a_greeting():
    # Background TV mentioning someone called Alex shouldn't wake the assistant.
    assert extract_command("I told Alex about it yesterday", "hey alex") == (False, "")
    assert extract_command("hey alexander", "hey alex") == (False, "")


def test_listener_flush_drops_queued_audio():
    import numpy as np

    from assistant.voice.listener import Listener

    listener = Listener()
    for _ in range(5):
        listener._frames.put(np.zeros(1600, dtype=np.float32))
    listener.flush()
    assert listener._frames.empty()


def _frame(level: float, n: int = 1600):
    import numpy as np

    return np.full(n, level, dtype=np.float32)  # constant signal: RMS == level


def test_adaptive_threshold_catches_normal_speech_in_a_noisy_room():
    """Room at 0.05 RMS (measured 2026-10-03). The old fixed 3x-at-startup rule
    put the trigger at 0.15 and missed a 0.12 voice; the adaptive, capped
    threshold catches it."""
    from assistant.voice.listener import Listener

    listener = Listener(silence_multiplier=2.0, max_threshold=0.08)
    listener._stream = object()  # pretend the mic is open; frames are injected
    for _ in range(10):
        listener._observe(_frame(0.05))          # startup calibration: 1s of the room
    for _ in range(30):
        listener._frames.put(_frame(0.05))       # 3s of room noise
    for _ in range(8):
        listener._frames.put(_frame(0.12))       # 0.8s "Hey Alex"
    for _ in range(10):
        listener._frames.put(_frame(0.05))       # back to the room
    audio = listener.listen_for_utterance(timeout=1)
    assert audio is not None
    assert 0.8 <= len(audio) / 16000 <= 1.8
    assert abs(listener._threshold() - 0.08) < 1e-9  # min(2 x 0.05, cap 0.08)


def test_threshold_has_a_floor_in_a_silent_room():
    from assistant.voice.listener import Listener

    listener = Listener()
    for _ in range(20):
        listener._observe(_frame(0.001))
    assert listener._threshold() == listener.min_threshold


def test_utterance_ends_if_audio_stops_arriving():
    from assistant.voice.listener import Listener

    listener = Listener()
    listener._stream = object()
    for _ in range(10):
        listener._observe(_frame(0.01))
    for _ in range(5):
        listener._frames.put(_frame(0.2))  # speech, then the feed dies
    audio = listener.listen_for_utterance(timeout=1)
    assert audio is not None and len(audio) == 5 * 1600
