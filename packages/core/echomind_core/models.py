"""SQLAlchemy ORM models for the EchoMind catalog."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Common base for all ORM models."""


class Book(Base):
    __tablename__ = "books"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(512))
    author: Mapped[str | None] = mapped_column(String(256), nullable=True)
    narrator: Mapped[str | None] = mapped_column(String(256), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    genre: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    language: Mapped[str] = mapped_column(String(8), default="en")
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    audio_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    chapters: Mapped[list["Chapter"]] = relationship(
        back_populates="book", cascade="all, delete-orphan", order_by="Chapter.idx"
    )

    def __repr__(self) -> str:
        return f"<Book {self.slug!r} title={self.title!r}>"


class Chapter(Base):
    __tablename__ = "chapters"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    transcript: Mapped[str] = mapped_column(Text, default="")
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment_label: Mapped[str | None] = mapped_column(String(16), nullable=True)

    book: Mapped[Book] = relationship(back_populates="chapters")

    def __repr__(self) -> str:
        return f"<Chapter {self.idx} of {self.book_id} ({self.start_sec:.1f}-{self.end_sec:.1f}s)>"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    handle: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    interactions: Mapped[list["Interaction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    book_id: Mapped[str] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), index=True)
    event: Mapped[str] = mapped_column(String(16))  # listen|finish|like|skip
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    ts: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    user: Mapped[User] = relationship(back_populates="interactions")
