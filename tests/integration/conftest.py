"""Pytest fixtures for integration tests against real PostgreSQL + pgvector."""

import asyncio
import os
import sys
from collections.abc import AsyncIterator

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from kb.config import get_settings

# Ensure WindowsSelectorEventLoopPolicy is set on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

settings = get_settings()
TEST_DB_URL = settings.DATABASE_URL
os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ["REDIS_URL"] = settings.REDIS_URL
os.environ["JWT_SECRET"] = settings.JWT_SECRET
os.environ["EMBEDDING_DIM"] = str(settings.EMBEDDING_DIM)


@pytest_asyncio.fixture
async def db_engine() -> AsyncIterator[AsyncEngine]:
    """Create asynchronous SQLAlchemy engine for integration tests."""
    engine = create_async_engine(TEST_DB_URL, echo=False, pool_pre_ping=True)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Provide an isolated database session with a rolled-back transaction per test."""
    session_factory = async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        # Clean existing test data across tables before test
        await session.execute(
            text(
                "TRUNCATE TABLE feedback, messages, conversations, chunks, "
                "hub_documents, documents, hubs RESTART IDENTITY CASCADE;"
            )
        )
        await session.commit()
        yield session
        await session.rollback()
