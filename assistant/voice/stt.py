"""Speech-to-text via local faster-whisper. Runs on CPU, no API key, no network."""
from __future__ import annotations

import numpy as np


class WhisperTranscriber:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """audio: mono float32 samples in [-1, 1] at `sample_rate` Hz."""
        if audio.size == 0:
            return ""
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,  # we do our own VAD before this ever gets called
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
