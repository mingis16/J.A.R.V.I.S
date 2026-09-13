"""Text-to-speech backends.

Pyttsx3Speaker (Windows SAPI voice, offline, zero setup) is the default. To
swap in an ElevenLabs voice later, implement a class with the same
`speak(text)` method and point `voice.tts_engine` in config.yaml at it — no
other code in `voice_assistant.py` needs to change.
"""
from __future__ import annotations

from typing import Protocol


class Speaker(Protocol):
    def speak(self, text: str) -> None: ...


class Pyttsx3Speaker:
    """Offline TTS via Windows SAPI (through pyttsx3). No API key needed."""

    def __init__(self, rate: int = 180, voice_id: str | None = None):
        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        if voice_id:
            self._engine.setProperty("voice", voice_id)

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        self._engine.say(text)
        self._engine.runAndWait()


def build_speaker(cfg: dict) -> Speaker:
    voice_cfg = cfg.get("voice", {})
    engine = voice_cfg.get("tts_engine", "pyttsx3")
    if engine == "pyttsx3":
        return Pyttsx3Speaker(
            rate=voice_cfg.get("tts_rate", 180),
            voice_id=voice_cfg.get("tts_voice_id"),
        )
    raise ValueError(f"Unknown tts_engine: {engine!r}")
