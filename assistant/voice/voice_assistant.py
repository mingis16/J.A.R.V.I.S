from __future__ import annotations

import re
import sys

from assistant.orchestrator import Orchestrator
from assistant.voice.listener import Listener
from assistant.voice.stt import WhisperTranscriber
from assistant.voice.tts import build_speaker


def extract_command(transcript: str, wake_word: str) -> tuple[bool, str]:
    """Check whether `transcript` contains the wake word, and if so return the
    text spoken after it (may be empty, meaning "just said the name").

    Pure/testable: no audio, no model calls.
    """
    pattern = re.compile(rf"\b{re.escape(wake_word)}\b", re.IGNORECASE)
    match = pattern.search(transcript)
    if not match:
        return False, ""
    remainder = transcript[match.end() :].strip(" ,.:;-")
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
        self.speaker.speak(text)

    def run(self) -> int:
        print(f"{self.name} is listening. Say '{self.wake_word}' to wake it, Ctrl+C to quit.")
        print("Calibrating microphone for ambient noise...")
        self.listener.calibrate_noise_floor()
        print("Ready.")

        while True:
            try:
                audio = self.listener.listen_for_utterance(timeout=None)
            except KeyboardInterrupt:
                print()
                return 0

            if audio is None:
                continue

            transcript = self.transcriber.transcribe(audio, self.listener.sample_rate)
            if not transcript:
                continue
            print(f"[heard] {transcript}")

            woke, remainder = extract_command(transcript, self.wake_word)
            if not woke:
                continue

            command = remainder
            if not command:
                self._say_and_print("Yes?")
                try:
                    audio = self.listener.listen_for_utterance(timeout=self.active_timeout_s)
                except KeyboardInterrupt:
                    print()
                    return 0
                if audio is None:
                    self._say_and_print(f"Didn't catch that — say '{self.wake_word}' again when you're ready.")
                    continue
                command = self.transcriber.transcribe(audio, self.listener.sample_rate)
                if not command:
                    self._say_and_print(f"Didn't catch that — say '{self.wake_word}' again when you're ready.")
                    continue

            print(f"you> {command}")
            if command.strip().lower() in {"exit", "quit", "stop", "goodbye"}:
                self._say_and_print("Goodbye.")
                return 0

            try:
                reply = self.orchestrator.chat(command)
            except Exception as exc:  # a single bad API call must not kill a long-running voice session
                print(f"Error calling the assistant: {type(exc).__name__}: {exc}", file=sys.stderr)
                self._say_and_print("Sorry, I hit an error talking to my brain. Check the terminal for details.")
                continue
            self._say_and_print(reply)


def main() -> int:
    from trading_bot.config import REPO_ROOT, load_env, load_yaml_config
    import os

    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example).", file=sys.stderr)
        return 1

    cfg = load_yaml_config()
    assistant = VoiceAssistant(REPO_ROOT, cfg)
    return assistant.run()


if __name__ == "__main__":
    sys.exit(main())
