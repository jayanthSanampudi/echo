"""Chapter segmentation.

A hybrid: silence-based candidate detection from raw audio, then a textual
heuristic (transcript breaks, "Chapter N" detection) that confirms which
candidates are real chapter boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from echomind_core.audio import DEFAULT_SR
from echomind_core.logging import get_logger
from echomind_ml.asr import TranscriptSegment

logger = get_logger(__name__)

CHAPTER_RE = re.compile(r"\b(chapter|part|book)\s+(?:[ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ChapterBoundary:
    idx: int
    start_sec: float
    end_sec: float
    title: str | None = None


def detect_long_silences(
    samples: np.ndarray,
    sr: int = DEFAULT_SR,
    min_silence_sec: float = 2.0,
    top_db: int = 35,
) -> list[float]:
    """Return timestamps (sec) where long silences end — chapter boundary candidates."""
    import librosa

    non_silent = librosa.effects.split(samples, top_db=top_db)
    boundaries: list[float] = []
    prev_end = 0
    for start, end in non_silent:
        gap_sec = (start - prev_end) / sr
        if gap_sec >= min_silence_sec and prev_end > 0:
            boundaries.append(start / sr)
        prev_end = end
    return boundaries


def segment_by_transcript(
    segments: list[TranscriptSegment],
    silence_boundaries: list[float] | None = None,
    silence_window_sec: float = 5.0,
) -> list[ChapterBoundary]:
    """Walk the transcript, marking a new chapter when:
    - the text matches a "Chapter N" pattern, OR
    - a long silence boundary aligns within `silence_window_sec`.
    Always emits at least one chapter (the full book)."""
    if not segments:
        return []

    silence_boundaries = silence_boundaries or []
    breaks: list[int] = []  # indices into segments where a new chapter starts

    for i, seg in enumerate(segments):
        if CHAPTER_RE.search(seg.text):
            breaks.append(i)
            continue
        # silence-aligned break
        for b in silence_boundaries:
            if abs(seg.start_sec - b) <= silence_window_sec:
                breaks.append(i)
                break

    # always include the implicit first chapter at idx 0
    starts = sorted(set([0, *breaks]))

    chapters: list[ChapterBoundary] = []
    for ci, start_idx in enumerate(starts):
        end_idx = starts[ci + 1] if ci + 1 < len(starts) else len(segments)
        start_sec = segments[start_idx].start_sec
        end_sec = segments[end_idx - 1].end_sec
        title = _title_from_segment(segments[start_idx].text)
        chapters.append(ChapterBoundary(idx=ci, start_sec=start_sec, end_sec=end_sec, title=title))
    return chapters


def _title_from_segment(text: str) -> str | None:
    m = CHAPTER_RE.search(text)
    if not m:
        return None
    return text[m.start() : min(len(text), m.end() + 40)].strip()


def chapter_text(
    chapter: ChapterBoundary, segments: list[TranscriptSegment]
) -> str:
    """Concatenate transcript text for a chapter."""
    parts = [s.text for s in segments if s.start_sec >= chapter.start_sec and s.end_sec <= chapter.end_sec]
    return " ".join(parts)
