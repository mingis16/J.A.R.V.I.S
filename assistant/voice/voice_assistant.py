from __future__ import annotations

import re
import sys
import time

from assistant.orchestrator import Orchestrator
from assistant.voice.listener import Listener
from assistant.voice.stt import WhisperTranscriber
from assistant.voice.tts import build_speaker


GREETINGS = ("hey", "hi", "hello", "hay", "hei", "ok", "okay", "yo")
# How Whisper tends to write "Alex" when it mishears it.
NAME_VARIANTS = {"alex": ("alex", "alix", "alec", "aleks", "allex", "alexa")}
_SEP = r"[\s,.!?;:-]+"  # Whisper punctuates freely: "Hey, Alex." / "Hey Alex!"


def extract_command(transcript: str, wake_word: str) -> tuple[bool, str]:
    """Check whether `transcript` calls the assistant, and if so return the
    text spoken after the call (may be empty, meaning "just said the name").

    Accepts any greeting + the name anywhere ("Hey Alex", "hi, Alex",
    "Hello Alex") or the bare name opening the utterance ("Alex, what's...").
    Only "hey alex" used to work — "Hi Alex" was silently ignored.

    Pure/testable: no audio, no model calls.
    """
    words = wake_word.lower().split()
    names = NAME_VARIANTS.get(words[-1], (words[-1],))
    name_re = "(?:" + "|".join(re.escape(n) for n in names) + ")"
    greet_re = "(?:" + "|".join(re.escape(g) for g in sorted({*GREETINGS, *words[:-1]})) + ")"
    match = re.search(rf"\b{greet_re}{_SEP}{name_re}\b", transcript, re.IGNORECASE) or re.match(
        rf"^\W*{name_re}\b", transcript, re.IGNORECASE
    )
    if not match:
        return False, ""
    remainder = transcript[match.end() :].strip(" ,.:;-!?\"'")
    # Bare punctuation left over ("Hey Alex!" -> "!") isn't a command.
    if not any(ch.isalnum() for ch in remainder):
        remainder = ""
    return True, remainder


class VoiceAssistant:
    def __init__(self, repo_root, cfg: dict):
        voice_cfg = cfg.get("voice", {})
        self.name = voice_cfg.get("name", "Assistant")
        self.wake_word = voice_cfg.get("wake_word", self.name).lower()
        self.active_timeout_s = voice_cfg.get("active_listen_timeout_s", 12)

        self.orchestrator = Orchestrator(repo_root, cfg)
        self.listener = Listener(
            sample_rate=voice_cfg.get("sample_rate", 16000),
            silence_multiplier=voice_cfg.get("vad_silence_multiplier", 2.0),
            max_threshold=voice_cfg.get("vad_max_threshold", 0.08),
            min_speech_ms=voice_cfg.get("vad_min_speech_ms", 200),
            silence_hangover_ms=voice_cfg.get("vad_silence_hangover_ms", 800),
            max_utterance_s=voice_cfg.get("vad_max_utterance_s", 10.0),
        )
        self.transcriber = WhisperTranscriber(
            model_size=voice_cfg.get("whisper_model", "base"),
            device=voice_cfg.get("whisper_device", "cpu"),
            compute_type=voice_cfg.get("whisper_compute_type", "int8"),
        )
        self.speaker = build_speaker(cfg)

    def _say_and_print(self, text: str) -> None:
        print(f"{self.name}> {text}")
        try:
            self.speaker.speak(text)
        except Exception as exc:
            print(f"(TTS failed: {type(exc).__name__}: {exc} — reply was printed above, just not spoken)")
        # The mic stays open while speaking; drop our own voice so it isn't
        # transcribed as the user's next command.
        self.listener.flush()

    def run(self) -> int:
        print(
            f"{self.name} is listening. Say 'Hey {self.name}' (or 'Hi {self.name}', or start with "
            f"'{self.name}, ...') to wake it, Ctrl+C to quit."
        )
        print("Calibrating microphone for ambient noise...")
        try:
            self.listener.calibrate_noise_floor()
            print(f"Ready. (speech threshold {self.listener._threshold():.3f})")
            while True:
                try:
                    if self._handle_one_utterance() == "exit":
                        return 0
                except KeyboardInterrupt:
                    print()
                    return 0
                except Exception as exc:
                    # One bad utterance (mic hiccup, transcription error) must not
                    # end a session that's meant to run all day.
                    print(f"Voice loop error, continuing: {type(exc).__name__}: {exc}", file=sys.stderr)
                    time.sleep(1)
        finally:
            self.listener.close()

    def _handle_one_utterance(self) -> str | None:
        audio = self.listener.listen_for_utterance(timeout=None)
        if audio is None:
            return None

        transcript = self.transcriber.transcribe(audio, self.listener.sample_rate)
        if not transcript:
            return None
        print(f"[heard {time.strftime('%H:%M:%S')}] {transcript}")

        woke, command = extract_command(transcript, self.wake_word)
        if not woke:
            return None

        if not command:
            self._say_and_print("Yes?")
            audio = self.listener.listen_for_utterance(timeout=self.active_timeout_s)
            follow_up = self.transcriber.transcribe(audio, self.listener.sample_rate) if audio is not None else ""
            command = (follow_up or "").strip(" ,.:;-!?\"'")
            if not any(ch.isalnum() for ch in command):
                self._say_and_print(f"Didn't catch that — say '{self.wake_word}' again when you're ready.")
                return None

        print(f"you> {command}")
        if command.strip(" .!?").lower() in {"exit", "quit", "stop", "goodbye"}:
            self._say_and_print("Goodbye.")
            return "exit"

        try:
            reply = self.orchestrator.chat(command)
        except Exception as exc:  # a single bad API call must not kill a long-running voice session
            print(f"Error calling the assistant: {type(exc).__name__}: {exc}", file=sys.stderr)
            if "credit balance" in str(exc):
                self._say_and_print(
                    "My Anthropic API credit has run out, so I can hear you but can't think of a reply. "
                    "Top it up in the Anthropic console under Plans and Billing."
                )
            else:
                self._say_and_print("Sorry, I hit an error talking to my brain. Check the terminal for details.")
            return None
        self._say_and_print(reply)
        return None


def main() -> int:
    from trading_bot.config import REPO_ROOT, load_env, load_yaml_config
    import os

    # Whisper transcribes background audio in any script (it once heard a
    # Chinese character), and printing that to a cp1252 console or redirected
    # log raised UnicodeEncodeError and killed the whole session.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example).", file=sys.stderr)
        return 1

    cfg = load_yaml_config()
    assistant = VoiceAssistant(REPO_ROOT, cfg)
    return assistant.run()


if __name__ == "__main__":
    sys.exit(main())
