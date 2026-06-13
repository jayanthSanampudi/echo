"""SQLAlchemy engine + session helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from echomind_core.config import get_settings
from echomind_core.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        url = get_settings().database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, future=True, connect_args=connect_args, pool_pre_ping=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autocommit=False, autoflush=False, expire_on_commit=False
        )
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager that yields a Session, commits on success, rolls back on error."""
    sess = get_session_factory()()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


def create_all() -> None:
    """Create tables — fine for SQLite dev. Use Alembic for production migrations."""
    Base.metadata.create_all(bind=get_engine())


def drop_all() -> None:
    """Drop tables — test helper."""
    Base.metadata.drop_all(bind=get_engine())
