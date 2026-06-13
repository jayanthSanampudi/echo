"""Typer CLI for the worker."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from echomind_core.db import create_all
from echomind_core.logging import configure_logging
from echomind_worker.ingest import ingest_file, reembed_all
from echomind_worker.watcher import watch as _watch

app = typer.Typer(add_completion=False, help="EchoMind ingestion worker")
console = Console()


@app.callback()
def _init(log_level: str = "info") -> None:
    configure_logging(log_level)
    create_all()


@app.command()
def ingest(
    audio: Path = typer.Argument(..., exists=True, readable=True, help="path to audio file"),
    slug: str = typer.Option(..., "--slug", "-s", help="URL-safe identifier for the book"),
    title: str = typer.Option(..., "--title", "-t", help="book title"),
    author: str | None = typer.Option(None, "--author", "-a"),
    narrator: str | None = typer.Option(None, "--narrator", "-n"),
    genre: str | None = typer.Option(None, "--genre", "-g"),
    language: str = typer.Option("en", "--language", "-l"),
) -> None:
    """Run the full ingestion pipeline on a single audio file."""
    result = ingest_file(
        audio_path=audio,
        slug=slug,
        title=title,
        author=author,
        narrator=narrator,
        genre=genre,
        language=language,
    )
    table = Table(title="ingestion complete", show_header=False, box=None)
    table.add_row("book_id", result.book_id)
    table.add_row("slug", result.slug)
    table.add_row("chapters", str(result.num_chapters))
    table.add_row("audio duration (s)", f"{result.duration_sec:.1f}")
    table.add_row("pipeline elapsed (s)", f"{result.elapsed_sec:.1f}")
    console.print(table)


@app.command()
def watch(
    directory: Path = typer.Argument(..., help="folder to monitor for new audio"),
    poll_sec: float = typer.Option(5.0, "--poll", help="polling interval seconds"),
) -> None:
    """Continuously ingest new audio files dropped into a directory."""
    _watch(directory=directory, poll_sec=poll_sec)


@app.command()
def reembed() -> None:
    """Re-embed every chapter (use after changing the embedding model)."""
    n = reembed_all()
    console.print(f"[green]re-embedded {n} chapters[/]")


if __name__ == "__main__":
    app()
