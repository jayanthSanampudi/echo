"""Lightweight speaker-turn detection.

Real production diarization uses pyannote / NeMo. Those require HF tokens and
significant compute. For an audiobook-scale demo, we use a cheap clustering
of MFCC frames into K speakers, which is sufficient for "find narrator change"
features without a 1GB model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echomind_core.audio import DEFAULT_SR
from echomind_core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str  # "spk_0", "spk_1", ...
    start_sec: float
    end_sec: float


def diarize(
    samples: np.ndarray,
    sr: int = DEFAULT_SR,
    n_speakers: int = 2,
    frame_sec: float = 1.5,
    smoothing_sec: float = 3.0,
) -> list[SpeakerTurn]:
    """Cluster audio frames into n_speakers buckets. Returns merged speaker turns."""
    import librosa
    from sklearn.cluster import KMeans

    if samples.size < sr:
        return []

    hop_length = max(1, int(sr * frame_sec))
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=20, hop_length=hop_length).T
    if mfcc.shape[0] < n_speakers:
        return [SpeakerTurn("spk_0", 0.0, samples.size / sr)]

    k = min(n_speakers, max(1, mfcc.shape[0]))
    km = KMeans(n_clusters=k, n_init=4, random_state=42).fit(mfcc)
    labels = km.labels_

    # smooth: assign each label by majority over a window
    smooth_window = max(1, int(smoothing_sec / frame_sec))
    smoothed = _majority_smooth(labels, smooth_window)

    # turn boundaries
    turns: list[SpeakerTurn] = []
    cur_label = smoothed[0]
    start_idx = 0
    for i, lbl in enumerate(smoothed[1:], start=1):
        if lbl != cur_label:
            turns.append(
                SpeakerTurn(
                    speaker=f"spk_{cur_label}",
                    start_sec=start_idx * frame_sec,
                    end_sec=i * frame_sec,
                )
            )
            cur_label = lbl
            start_idx = i
    turns.append(
        SpeakerTurn(
            speaker=f"spk_{cur_label}",
            start_sec=start_idx * frame_sec,
            end_sec=samples.size / sr,
        )
    )
    return turns


def _majority_smooth(labels: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return labels
    out = labels.copy()
    half = window // 2
    for i in range(len(labels)):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        vals, counts = np.unique(labels[lo:hi], return_counts=True)
        out[i] = vals[counts.argmax()]
    return out
