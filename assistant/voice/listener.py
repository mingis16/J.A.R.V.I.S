"""Microphone capture with simple energy-based voice activity detection (VAD).

No extra native VAD dependency (webrtcvad needs a C build on Windows) — an
ambient-noise-calibrated RMS threshold is good enough to segment "one
utterance" out of a live mic stream for wake-word + command capture.

The input stream stays open for the whole session and frames queue up in the
background. Opening a fresh stream per utterance (as this used to) left the
mic closed while Whisper transcribed the previous chunk — with a TV playing
that was several seconds deaf out of every ~30, and a "Hey Alex" spoken then
was simply never recorded.
"""
from __future__ import annotations

import queue
import time
from collections import deque

import numpy as np
import sounddevice as sd

# ~60s of 100ms frames; if transcription falls that far behind, drop the
# oldest audio rather than grow without bound.
_MAX_QUEUED_FRAMES = 600


class Listener:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 100,
        min_speech_ms: int = 200,
        silence_hangover_ms: int = 800,
        max_utterance_s: float = 10.0,
        silence_multiplier: float = 2.0,
        max_threshold: float = 0.08,
        min_threshold: float = 0.015,
    ):
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.frame_ms = frame_ms
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.silence_hangover_frames = max(1, silence_hangover_ms // frame_ms)
        self.max_utterance_frames = int(max_utterance_s * 1000 // frame_ms)
        self.silence_multiplier = silence_multiplier
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self._noise_floor = 0.01
        # Recent between-utterance frame levels (~5s): the noise floor tracks
        # the room instead of being fixed by a 1-second sample at startup.
        self._recent_levels: deque[float] = deque(maxlen=max(10, 5000 // frame_ms))
        self._frames: queue.Queue = queue.Queue()
        self._stream: sd.InputStream | None = None

    # ----- stream lifecycle ---------------------------------------------------

    def start(self) -> None:
        if self._stream is not None:
            return

        def callback(indata, frames, time_info, status):
            if self._frames.qsize() >= _MAX_QUEUED_FRAMES:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
            self._frames.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def flush(self) -> None:
        """Drop queued audio — e.g. the assistant's own voice captured while it
        was speaking, which would otherwise be transcribed as the next command."""
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    # ----- VAD ------------------------------------------------------------------

    def calibrate_noise_floor(self, duration_s: float = 1.0) -> None:
        """Sample ambient sound briefly to set a speech-detection threshold."""
        self.start()
        self.flush()
        needed = max(1, int(duration_s * 1000 // self.frame_ms))
        frames = [self._frames.get(timeout=5.0) for _ in range(needed)]
        for frame in frames:
            self._observe(frame)

    def _observe(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        self._recent_levels.append(rms)
        if len(self._recent_levels) >= 10:
            # 20th percentile: the quieter moments of the room, robust to bursts
            # of speech or TV dialogue in the window.
            self._noise_floor = max(float(np.percentile(self._recent_levels, 20)), 0.002)
        return rms

    def _threshold(self) -> float:
        """Speech must be clearly above the room — but never more than
        max_threshold, so a loud room (TV, fan) can't push the trigger level
        beyond a normal speaking voice. A fixed 3x-at-startup threshold
        reached 0.117 here on 2026-10-03, above the user's voice."""
        return min(max(self._noise_floor * self.silence_multiplier, self.min_threshold), self.max_threshold)

    def listen_for_utterance(self, timeout: float | None = None) -> np.ndarray | None:
        """Block until one speech utterance (speech, then trailing silence, or
        the max length) is captured, or `timeout` seconds pass with no speech
        starting. Returns mono float32 samples at self.sample_rate, or None."""
        self.start()
        speaking = False
        speech_frames = 0
        silence_run = 0
        collected: list[np.ndarray] = []
        start_time = time.monotonic()
        threshold = self._threshold()

        last_frame_at = time.monotonic()
        while True:
            try:
                frame = self._frames.get(timeout=1.0)
                last_frame_at = time.monotonic()
            except queue.Empty:
                if not speaking and timeout is not None and time.monotonic() - start_time > timeout:
                    return None
                if speaking and time.monotonic() - last_frame_at > 2.0:
                    break  # audio stopped arriving mid-utterance (mic unplugged/stalled)
                continue

            if not speaking:
                rms = self._observe(frame)
                threshold = self._threshold()  # adapts between utterances, fixed within one
            else:
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
