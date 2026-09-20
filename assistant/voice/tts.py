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
    """Offline TTS via Windows SAPI (through pyttsx3). No API key needed.

    A fresh engine is created for every speak() call rather than reused —
    reusing one pyttsx3/SAPI5 engine instance across many calls in a
    long-running process is a well-known source of "works once, then goes
    silent" behavior on Windows.
    """

    def __init__(self, rate: int = 180, voice_id: str | None = None):
        self._rate = rate
        self._voice_id = voice_id

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        import pyttsx3

        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", self._rate)
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
            engine.say(text)
            engine.runAndWait()
        finally:
            engine.stop()


def build_speaker(cfg: dict) -> Speaker:
    voice_cfg = cfg.get("voice", {})
    engine = voice_cfg.get("tts_engine", "pyttsx3")
    if engine == "pyttsx3":
        return Pyttsx3Speaker(
            rate=voice_cfg.get("tts_rate", 180),
            voice_id=voice_cfg.get("tts_voice_id"),
        )
    raise ValueError(f"Unknown tts_engine: {engine!r}")
