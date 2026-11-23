"""Integration tests verifying Alembic migrations and database extension / index creation."""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.integration
def test_alembic_migrations_cycle() -> None:
    """Verify that upgrade head, downgrade base, and upgrade head again all succeed."""
    alembic_cfg = Config("alembic.ini")

    # 1. Downgrade to base
    command.downgrade(alembic_cfg, "base")

    # 2. Upgrade to head
    command.upgrade(alembic_cfg, "head")

    # 3. Downgrade to base again
    command.downgrade(alembic_cfg, "base")

    # 4. Final upgrade to head
    command.upgrade(alembic_cfg, "head")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vector_extension_and_indexes_exist(db_session: AsyncSession) -> None:
    """Verify pgvector extension, HNSW index, and GIN indexes exist in PostgreSQL."""
    # 1. Verify extension
    ext_res = await db_session.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
    )
    assert ext_res.scalar_one_or_none() == "vector"

    # 2. Verify indexes on chunks table
    idx_res = await db_session.execute(
        text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'chunks';")
    )
    chunk_indexes = {row[0]: row[1] for row in idx_res.all()}

    assert "ix_chunks_embedding_hnsw" in chunk_indexes
    assert "hnsw" in chunk_indexes["ix_chunks_embedding_hnsw"].lower()
    assert "vector_cosine_ops" in chunk_indexes["ix_chunks_embedding_hnsw"].lower()

    assert "ix_chunks_tsv" in chunk_indexes
    assert "gin" in chunk_indexes["ix_chunks_tsv"].lower()

    assert "ix_chunks_document_active" in chunk_indexes
    assert "ix_chunks_content_hash" in chunk_indexes

    # 3. Verify indexes on documents table
    doc_idx_res = await db_session.execute(
        text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'documents';")
    )
    doc_indexes = {row[0]: row[1] for row in doc_idx_res.all()}

    assert "ix_documents_active_source" in doc_indexes
    assert "unique" in doc_indexes["ix_documents_active_source"].lower()
    assert "active" in doc_indexes["ix_documents_active_source"].lower()

    assert "ix_documents_acl_tags" in doc_indexes
    assert "gin" in doc_indexes["ix_documents_acl_tags"].lower()
