"""Filesystem watcher — pick up new audio files dropped into a directory."""

from __future__ import annotations

import time
from pathlib import Path

from echomind_core.logging import get_logger
from echomind_worker.ingest import ingest_file

logger = get_logger(__name__)

SUPPORTED = {".mp3", ".m4a", ".m4b", ".wav", ".flac", ".ogg"}


def watch(directory: str | Path, poll_sec: float = 5.0) -> None:
    """Simple polling watcher — adequate for a single-node deployment.
    Production would use watchdog/inotify or a queue."""
    seen: set[str] = set()
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.info("watcher.start", path=str(dir_path), poll_sec=poll_sec)

    try:
        while True:
            for entry in dir_path.iterdir():
                if not entry.is_file():
                    continue
                if entry.suffix.lower() not in SUPPORTED:
                    continue
                key = str(entry.resolve())
                if key in seen:
                    continue
                seen.add(key)
                slug = entry.stem.lower().replace(" ", "-")
                try:
                    ingest_file(audio_path=entry, slug=slug, title=entry.stem)
                except Exception as e:
                    logger.exception("watcher.failed", file=str(entry), err=str(e))
            time.sleep(poll_sec)
    except KeyboardInterrupt:
        logger.info("watcher.stopped")
