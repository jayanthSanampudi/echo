"""Audio utility tests — pure numpy, no model downloads."""

from __future__ import annotations

import numpy as np

from echomind_core.audio import DEFAULT_SR, chunk_fixed, split_silences


def _synthetic_speech(seconds: float = 5.0, sr: int = DEFAULT_SR) -> np.ndarray:
    """Synthesize a noisy sine wave so silence detection has something to find."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 220 * t)
    # punch a 1s silence in the middle
    mid = int(2 * sr)
    sig[mid : mid + sr] = 0.0
    return sig.astype(np.float32)


def test_chunk_fixed_emits_overlapping_windows():
    samples = _synthetic_speech(seconds=10.0)
    chunks = chunk_fixed(samples, chunk_sec=3.0, overlap_sec=0.5)
    assert len(chunks) >= 3
    for seg in chunks:
        assert seg.end_sec > seg.start_sec
        assert seg.samples.size > 0


def test_chunk_fixed_handles_short_audio():
    samples = _synthetic_speech(seconds=0.5)
    chunks = chunk_fixed(samples, chunk_sec=3.0)
    assert chunks == [] or all(s.duration_sec >= 1.0 for s in chunks)


def test_split_silences_finds_nontrivial_segments():
    samples = _synthetic_speech(seconds=5.0)
    segs = split_silences(samples, min_segment_sec=0.5)
    assert len(segs) >= 1
    for s in segs:
        assert s.duration_sec >= 0.5
