"""Whisper ASR wrapper.

Tiny by default for CPU-friendliness. Returns segment-level transcripts with timestamps,
which downstream code uses for chunking and chapter alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from echomind_core.audio import DEFAULT_SR, load_audio
from echomind_core.config import get_settings
from echomind_core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start_sec": self.start_sec, "end_sec": self.end_sec, "text": self.text}


class WhisperASR:
    """Lazy-loading Whisper wrapper. Single instance per process recommended."""

    _model: Any = None
    _model_name: str | None = None

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().whisper_model

    def _load(self) -> Any:
        cls = type(self)
        if cls._model is None or cls._model_name != self.model_name:
            import whisper  # lazy — model load is expensive

            cache_dir = str(get_settings().model_cache_path)
            logger.info("whisper.load", model=self.model_name, cache=cache_dir)
            cls._model = whisper.load_model(self.model_name, download_root=cache_dir)
            cls._model_name = self.model_name
        return cls._model

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> list[TranscriptSegment]:
        """Transcribe a file or in-memory audio array. Returns timestamped segments."""
        model = self._load()
        if isinstance(audio, (str, Path)):
            samples = load_audio(audio, sr=DEFAULT_SR, mono=True)
        else:
            samples = audio.astype(np.float32)

        logger.debug("whisper.transcribe", samples=len(samples), sr=DEFAULT_SR)
        result = model.transcribe(
            samples,
            language=language,
            word_timestamps=word_timestamps,
            fp16=False,  # CPU
            verbose=False,
        )
        segments = result.get("segments", []) or []
        return [
            TranscriptSegment(
                start_sec=float(s["start"]),
                end_sec=float(s["end"]),
                text=str(s["text"]).strip(),
            )
            for s in segments
        ]

    def transcribe_to_text(self, audio: str | Path | np.ndarray) -> str:
        """Convenience: return the full joined transcript as one string."""
        return " ".join(seg.text for seg in self.transcribe(audio))
