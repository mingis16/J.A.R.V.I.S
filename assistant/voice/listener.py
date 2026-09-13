"""Microphone capture with simple energy-based voice activity detection (VAD).

No extra native VAD dependency (webrtcvad needs a C build on Windows) — an
ambient-noise-calibrated RMS threshold is good enough to segment "one
utterance" out of a live mic stream for wake-word + command capture.
"""
from __future__ import annotations

import queue
import time

import numpy as np
import sounddevice as sd


class Listener:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 100,
        min_speech_ms: int = 200,
        silence_hangover_ms: int = 800,
        max_utterance_s: float = 30.0,
        silence_multiplier: float = 3.0,
    ):
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.silence_hangover_frames = max(1, silence_hangover_ms // frame_ms)
        self.max_utterance_frames = int(max_utterance_s * 1000 // frame_ms)
        self.silence_multiplier = silence_multiplier
        self._noise_floor = 0.01

    def calibrate_noise_floor(self, duration_s: float = 1.0) -> None:
        """Sample ambient silence briefly to set a speech-detection threshold."""
        frames = int(duration_s * self.sample_rate)
        audio = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
        sd.wait()
        rms = float(np.sqrt(np.mean(np.square(audio))))
        self._noise_floor = max(rms, 0.002)

    def _threshold(self) -> float:
        return max(self._noise_floor * self.silence_multiplier, 0.02)

    def listen_for_utterance(self, timeout: float | None = None) -> np.ndarray | None:
        """Block until one speech utterance (speech, then trailing silence) is
        captured, or `timeout` seconds pass with no speech starting at all.
        Returns mono float32 samples at self.sample_rate, or None on timeout.
        """
        frame_q: queue.Queue = queue.Queue()

        def callback(indata, frames, time_info, status):
            frame_q.put(indata[:, 0].copy())

        threshold = self._threshold()
        speaking = False
        speech_frames = 0
        silence_run = 0
        collected: list[np.ndarray] = []
        start_time = time.monotonic()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=callback,
        ):
            while True:
                try:
                    frame = frame_q.get(timeout=1.0)
                except queue.Empty:
                    if not speaking and timeout is not None and time.monotonic() - start_time > timeout:
                        return None
                    continue

                rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
                is_loud = rms > threshold

                if not speaking:
                    if is_loud:
                        speech_frames += 1
                        collected.append(frame)
                        if speech_frames >= self.min_speech_frames:
                            speaking = True
                            silence_run = 0
                    else:
                        speech_frames = 0
                        collected.clear()
                        if timeout is not None and time.monotonic() - start_time > timeout:
                            return None
                    continue

                collected.append(frame)
                if is_loud:
                    silence_run = 0
                else:
                    silence_run += 1
                    if silence_run >= self.silence_hangover_frames:
                        break
                if len(collected) >= self.max_utterance_frames:
                    break

        return np.concatenate(collected) if collected else None
