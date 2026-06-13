"""Audio utilities: load, resample, segment, detect silences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SR = 16000  # Whisper + most embedding models use 16kHz mono


@dataclass(frozen=True)
class AudioSegment:
    start_sec: float
    end_sec: float
    samples: np.ndarray  # mono float32 at DEFAULT_SR

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def load_audio(path: str | Path, sr: int = DEFAULT_SR, mono: bool = True) -> np.ndarray:
    """Load an audio file to a mono float32 array at the target sample rate."""
    import librosa  # lazy — librosa pulls numba

    y, _ = librosa.load(str(path), sr=sr, mono=mono)
    return y.astype(np.float32)


def split_silences(
    samples: np.ndarray,
    sr: int = DEFAULT_SR,
    top_db: int = 30,
    min_segment_sec: float = 1.5,
) -> list[AudioSegment]:
    """Split audio at silences. Returns non-silent segments above min length."""
    import librosa

    intervals = librosa.effects.split(samples, top_db=top_db)
    segments: list[AudioSegment] = []
    for start, end in intervals:
        start_s = float(start) / sr
        end_s = float(end) / sr
        if end_s - start_s < min_segment_sec:
            continue
        segments.append(AudioSegment(start_s, end_s, samples[start:end]))
    return segments


def chunk_fixed(
    samples: np.ndarray,
    sr: int = DEFAULT_SR,
    chunk_sec: float = 30.0,
    overlap_sec: float = 1.0,
) -> list[AudioSegment]:
    """Fixed-length chunks with overlap — good for batched ASR."""
    chunk = int(chunk_sec * sr)
    step = max(1, int((chunk_sec - overlap_sec) * sr))
    segments: list[AudioSegment] = []
    for start in range(0, len(samples), step):
        end = min(start + chunk, len(samples))
        if end - start < sr:  # skip <1s tail
            break
        segments.append(AudioSegment(start / sr, end / sr, samples[start:end]))
        if end == len(samples):
            break
    return segments


def duration_sec(path: str | Path) -> float:
    import librosa

    return float(librosa.get_duration(path=str(path)))
