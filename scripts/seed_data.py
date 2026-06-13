"""Seed the local catalog with a few public-domain books.

Pulls plaintext chapters from Project Gutenberg (small extracts), splits them,
and runs them through the text-ingestion path. This makes search, Q&A, and the
sentiment arc work *without* downloading any audio.

Run:
    uv run python scripts/seed_data.py
"""

from __future__ import annotations

import argparse
import textwrap

from echomind_core.db import create_all
from echomind_core.logging import configure_logging, get_logger
from echomind_worker.ingest import ingest_text

logger = get_logger(__name__)


SEED_BOOKS: list[dict] = [
    {
        "slug": "frankenstein",
        "title": "Frankenstein, or, The Modern Prometheus",
        "author": "Mary Shelley",
        "genre": "gothic-horror",
        "chapters": [
            textwrap.dedent(
                """\
                I am by birth a Genevese, and my family is one of the most distinguished
                of that republic. My ancestors had been for many years counsellors and
                syndics, and my father had filled several public situations with honour
                and reputation. He was respected by all who knew him for his integrity
                and indefatigable attention to public business.
                """
            ),
            textwrap.dedent(
                """\
                It was on a dreary night of November that I beheld the accomplishment of
                my toils. With an anxiety that almost amounted to agony, I collected the
                instruments of life around me, that I might infuse a spark of being into
                the lifeless thing that lay at my feet.
                """
            ),
            textwrap.dedent(
                """\
                I expected this reception. All men hate the wretched; how, then, must I
                be hated, who am miserable beyond all living things! Yet you, my creator,
                detest and spurn me, thy creature, to whom thou art bound by ties only
                dissoluble by the annihilation of one of us.
                """
            ),
        ],
    },
    {
        "slug": "pride-and-prejudice",
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "genre": "romance",
        "chapters": [
            textwrap.dedent(
                """\
                It is a truth universally acknowledged, that a single man in possession
                of a good fortune, must be in want of a wife. However little known the
                feelings or views of such a man may be on his first entering a
                neighbourhood, this truth is so well fixed in the minds of the
                surrounding families, that he is considered as the rightful property
                of some one or other of their daughters.
                """
            ),
            textwrap.dedent(
                """\
                Mr. Darcy soon drew the attention of the room by his fine, tall person,
                handsome features, noble mien, and the report which was in general
                circulation within five minutes after his entrance, of his having ten
                thousand a year.
                """
            ),
        ],
    },
    {
        "slug": "alice-in-wonderland",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "genre": "fantasy",
        "chapters": [
            textwrap.dedent(
                """\
                Alice was beginning to get very tired of sitting by her sister on the
                bank, and of having nothing to do: once or twice she had peeped into the
                book her sister was reading, but it had no pictures or conversations in
                it, and what is the use of a book, thought Alice, without pictures or
                conversations.
                """
            ),
            textwrap.dedent(
                """\
                The Cat only grinned when it saw Alice. It looked good-natured, she
                thought; still it had VERY long claws and a great many teeth, so she
                felt that it ought to be treated with respect.
                """
            ),
        ],
    },
    {
        "slug": "dracula",
        "title": "Dracula",
        "author": "Bram Stoker",
        "genre": "gothic-horror",
        "chapters": [
            textwrap.dedent(
                """\
                3 May. Bistritz. Left Munich at 8:35 P.M., on 1st May, arriving at
                Vienna early next morning; should have arrived at 6:46, but train was
                an hour late. Buda-Pesth seems a wonderful place, from the glimpse which
                I got of it from the train and the little I could walk through the streets.
                """
            ),
            textwrap.dedent(
                """\
                When the Count saw my face, his eyes blazed with a sort of demoniac fury,
                and he suddenly made a grab at my throat. I drew away, and his hand
                touched the string of beads which held the crucifix. It made an instant
                change in him, for the fury passed so quickly that I could hardly believe
                that it was ever there.
                """
            ),
        ],
    },
]


def main() -> None:
    configure_logging("info")
    create_all()

    parser = argparse.ArgumentParser(description="Seed sample books into EchoMind")
    parser.add_argument("--limit", type=int, default=len(SEED_BOOKS))
    args = parser.parse_args()

    for book in SEED_BOOKS[: args.limit]:
        result = ingest_text(
            slug=book["slug"],
            title=book["title"],
            author=book["author"],
            genre=book["genre"],
            chapters_text=book["chapters"],
        )
        logger.info(
            "seed.book",
            slug=result.slug,
            book_id=result.book_id,
            chapters=result.num_chapters,
        )

    print(f"seeded {min(args.limit, len(SEED_BOOKS))} books")


if __name__ == "__main__":
    main()
