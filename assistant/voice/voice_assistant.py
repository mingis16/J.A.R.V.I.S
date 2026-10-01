from __future__ import annotations

import re
import sys
import time

from assistant.orchestrator import Orchestrator
from assistant.voice.listener import Listener
from assistant.voice.stt import WhisperTranscriber
from assistant.voice.tts import build_speaker


def extract_command(transcript: str, wake_word: str) -> tuple[bool, str]:
    """Check whether `transcript` contains the wake word, and if so return the
    text spoken after it (may be empty, meaning "just said the name").

    Pure/testable: no audio, no model calls.
    """
    # Whisper punctuates freely ("Hey, Alex." / "Hey Alex!"), so let any run
    # of spaces/punctuation stand in for the space between wake-word words.
    words = [re.escape(w) for w in wake_word.split()]
    pattern = re.compile(r"\b" + r"[\s,.!?;:-]+".join(words) + r"\b", re.IGNORECASE)
    match = pattern.search(transcript)
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
            silence_multiplier=voice_cfg.get("vad_silence_multiplier", 3.0),
            min_speech_ms=voice_cfg.get("vad_min_speech_ms", 200),
            silence_hangover_ms=voice_cfg.get("vad_silence_hangover_ms", 800),
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

    def run(self) -> int:
        print(f"{self.name} is listening. Say '{self.wake_word}' to wake it, Ctrl+C to quit.")
        print("Calibrating microphone for ambient noise...")
        self.listener.calibrate_noise_floor()
        print("Ready.")

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

    def _handle_one_utterance(self) -> str | None:
        audio = self.listener.listen_for_utterance(timeout=None)
        if audio is None:
            return None

        transcript = self.transcriber.transcribe(audio, self.listener.sample_rate)
        if not transcript:
            return None
        print(f"[heard] {transcript}")

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
