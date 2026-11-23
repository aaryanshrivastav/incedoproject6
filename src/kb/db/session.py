"""Database engine, session factory, and transactional session management."""

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from kb.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Return cached asynchronous SQLAlchemy engine."""
    settings = get_settings()
    return create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return cached asynchronous session factory."""
    engine = get_engine()
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Context manager providing a transactional async session scope.

    Commits upon normal completion and rolls back if an exception occurs.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
