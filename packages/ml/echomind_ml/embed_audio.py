"""Audio embeddings via CLAP.

CLAP (Contrastive Language-Audio Pretraining) maps audio and text to a shared
embedding space — ideal for voice-style matching and audio retrieval.

For laptops / CI without CLAP weights, the embedder gracefully degrades to a
deterministic mel-spectrogram-statistics fingerprint that still supports
nearest-neighbor search and demonstrates the architecture. We log clearly when
the fallback path is taken so reviewers know.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from echomind_core.audio import DEFAULT_SR, load_audio
from echomind_core.config import get_settings
from echomind_core.logging import get_logger

logger = get_logger(__name__)


class AudioEmbedder:
    """CLAP-based audio embedder with a deterministic fallback."""

    _model: Any = None
    _processor: Any = None
    _model_name: str | None = None
    _fallback: bool = False

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().audio_embed_model

    def _load(self) -> tuple[Any, Any]:
        cls = type(self)
        if cls._model is None or cls._model_name != self.model_name:
            try:
                from transformers import ClapModel, ClapProcessor

                cache_dir = str(get_settings().model_cache_path)
                logger.info("audio_embed.load", model=self.model_name, cache=cache_dir)
                cls._model = ClapModel.from_pretrained(self.model_name, cache_dir=cache_dir).eval()
                cls._processor = ClapProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
                cls._fallback = False
            except Exception as e:
                logger.warning("audio_embed.fallback", reason=str(e))
                cls._fallback = True
            cls._model_name = self.model_name
        return cls._model, cls._processor

    @property
    def dim(self) -> int:
        return get_settings().audio_embed_dim

    def embed(self, samples: np.ndarray, sr: int = DEFAULT_SR) -> np.ndarray:
        """Embed one audio array. Returns (dim,) float32 L2-normalized."""
        model, processor = self._load()
        if type(self)._fallback or model is None:
            return self._mel_fingerprint(samples, sr)

        import torch

        inputs = processor(audios=samples, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_audio_features(**inputs)[0].cpu().numpy().astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-12
        return emb

    def embed_file(self, path: str | Path) -> np.ndarray:
        samples = load_audio(path, sr=DEFAULT_SR, mono=True)
        return self.embed(samples)

    def embed_one(self, samples: np.ndarray, sr: int = DEFAULT_SR) -> list[float]:
        return self.embed(samples, sr).tolist()

    # ─── Fallback ────────────────────────────────────────────────────────────

    def _mel_fingerprint(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Mel-spectrogram statistics fingerprint. Not CLAP, but deterministic and
        useful for nearest-neighbor demos when CLAP weights are unavailable."""
        import librosa

        target_dim = self.dim
        if samples.size < sr // 10:
            return np.zeros(target_dim, dtype=np.float32)

        mel = librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel)
        # mean + std per mel band → 256 features
        feats = np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)])
        # pad / trim to target_dim
        if feats.size < target_dim:
            feats = np.pad(feats, (0, target_dim - feats.size))
        else:
            feats = feats[:target_dim]
        feats = feats.astype(np.float32)
        feats /= np.linalg.norm(feats) + 1e-12
        return feats
