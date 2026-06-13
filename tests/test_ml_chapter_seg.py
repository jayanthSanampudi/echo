"""Chapter segmentation — pure-text path, no audio needed."""

from __future__ import annotations

from echomind_ml.asr import TranscriptSegment
from echomind_ml.chapter_seg import segment_by_transcript


def _seg(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start_sec=start, end_sec=end, text=text)


def test_detects_chapter_marker():
    segments = [
        _seg(0, 5, "Welcome to the story."),
        _seg(5, 10, "Chapter 1. The beginning."),
        _seg(10, 15, "Once upon a time..."),
        _seg(15, 20, "Chapter 2. The middle."),
        _seg(20, 25, "Things happen."),
    ]
    chapters = segment_by_transcript(segments)
    assert len(chapters) >= 2
    titles = [c.title or "" for c in chapters if c.title]
    assert any("Chapter 1" in t for t in titles)


def test_silence_boundary_triggers_chapter():
    segments = [
        _seg(0, 30, "Intro words."),
        _seg(60, 90, "After a long silence we continue."),
    ]
    silences = [60.0]
    chapters = segment_by_transcript(segments, silence_boundaries=silences, silence_window_sec=5.0)
    assert len(chapters) == 2


def test_always_one_chapter_even_without_breaks():
    segments = [_seg(0, 100, "All one chapter.")]
    chapters = segment_by_transcript(segments)
    assert len(chapters) == 1
    assert chapters[0].start_sec == 0
