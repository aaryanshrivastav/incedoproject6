"""Integration tests verifying database schema constraints, partial unique index, and pgvector."""

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from kb.db.models import ChunkModel, DocumentModel, HubDocumentModel, HubModel


@pytest.mark.integration
@pytest.mark.asyncio
async def test_partial_unique_index_active_documents(db_session: AsyncSession) -> None:
    """A second active document for the same source is rejected; superseded/failed allowed."""
    org_id = uuid4()
    source_uri = "docs://finance/q4_report.pdf"

    # 1. Insert first active document
    doc1 = DocumentModel(
        org_id=org_id,
        source_uri=source_uri,
        title="Q4 Report v1",
        content_hash="hash1",
        version=1,
        status="active",
        scope="hub",
    )
    db_session.add(doc1)
    await db_session.flush()

    # 2. Attempting to insert a second active document for same source fails
    doc2_duplicate = DocumentModel(
        org_id=org_id,
        source_uri=source_uri,
        title="Q4 Report v2 Duplicate Active",
        content_hash="hash2",
        version=2,
        status="active",
        scope="hub",
    )
    db_session.add(doc2_duplicate)
    with pytest.raises(IntegrityError) as exc_info:
        await db_session.flush()
    assert "ix_documents_active_source" in str(exc_info.value)

    # Roll back the failed statement within session
    await db_session.rollback()

    # 3. Mark original as superseded, then insert a new active version and a failed version
    doc1_retry = DocumentModel(
        org_id=org_id,
        source_uri=source_uri,
        title="Q4 Report v1",
        content_hash="hash1",
        version=1,
        status="superseded",
        scope="hub",
    )
    doc2_active = DocumentModel(
        org_id=org_id,
        source_uri=source_uri,
        title="Q4 Report v2 Active",
        content_hash="hash2",
        version=2,
        status="active",
        scope="hub",
    )
    doc3_failed = DocumentModel(
        org_id=org_id,
        source_uri=source_uri,
        title="Q4 Report v3 Failed Parse",
        content_hash="",
        version=3,
        status="failed",
        failure_reason="Corrupt header",
        scope="hub",
    )
    db_session.add_all([doc1_retry, doc2_active, doc3_failed])
    await db_session.flush()

    assert doc1_retry.status == "superseded"
    assert doc2_active.status == "active"
    assert doc3_failed.status == "failed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embedding_vector_dimension_constraint(db_session: AsyncSession) -> None:
    """Inserting an embedding with dimension different from 768 is rejected by pgvector."""
    org_id = uuid4()
    doc = DocumentModel(
        org_id=org_id,
        source_uri="docs://test/dim.txt",
        title="Dimension Test",
        content_hash="dimhash",
        version=1,
        status="active",
        scope="global",
    )
    db_session.add(doc)
    await db_session.flush()

    # Wrong dimension: 3 floats instead of 768
    chunk_bad_dim = ChunkModel(
        document_id=doc.id,
        ordinal=0,
        text="Hello world test",
        token_count=3,
        content_hash="h1",
        embedding=[0.1, 0.2, 0.3],
    )
    db_session.add(chunk_bad_dim)

    with pytest.raises(DBAPIError) as exc_info:
        await db_session.flush()
    assert "dimensions" in str(exc_info.value).lower()
    await db_session.rollback()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tsv_auto_population(db_session: AsyncSession) -> None:
    """The tsv column is generated automatically from text on insert."""
    org_id = uuid4()
    doc = DocumentModel(
        org_id=org_id,
        source_uri="docs://test/tsv.txt",
        title="TSV Test",
        content_hash="tsvhash",
        version=1,
        status="active",
        scope="global",
    )
    db_session.add(doc)
    await db_session.flush()

    valid_embedding = [0.01] * 768
    chunk = ChunkModel(
        document_id=doc.id,
        ordinal=0,
        text="PostgreSQL full-text indexing with tsvector and vector search",
        token_count=8,
        content_hash="tsv_chunk_hash",
        embedding=valid_embedding,
    )
    db_session.add(chunk)
    await db_session.flush()

    # Query the generated tsv directly from database
    res = await db_session.execute(
        text("SELECT tsv::text FROM chunks WHERE id = :cid;"),
        {"cid": chunk.id},
    )
    tsv_text = res.scalar_one()
    assert "'postgresql':1" in tsv_text
    assert "'vector':" in tsv_text
    assert "'full" in tsv_text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_document_cascade_delete(db_session: AsyncSession) -> None:
    """Deleting a document cascades to chunks and hub_documents."""
    org_id = uuid4()
    hub = HubModel(org_id=org_id, name="Eng Hub")
    db_session.add(hub)
    await db_session.flush()

    doc = DocumentModel(
        org_id=org_id,
        source_uri="docs://eng/design.md",
        title="Design Doc",
        content_hash="dh",
        version=1,
        status="active",
        scope="hub",
    )
    db_session.add(doc)
    await db_session.flush()

    # Link to hub
    hd = HubDocumentModel(hub_id=hub.id, document_id=doc.id)
    # Add chunk
    chunk = ChunkModel(
        document_id=doc.id,
        ordinal=0,
        text="Design doc text content",
        token_count=4,
        content_hash="ch1",
        embedding=[0.05] * 768,
    )
    db_session.add_all([hd, chunk])
    await db_session.flush()

    # Verify rows exist
    c_count = await db_session.execute(
        text("SELECT count(*) FROM chunks WHERE document_id = :did;"), {"did": doc.id}
    )
    hd_count = await db_session.execute(
        text("SELECT count(*) FROM hub_documents WHERE document_id = :did;"), {"did": doc.id}
    )
    assert c_count.scalar_one() == 1
    assert hd_count.scalar_one() == 1

    # Delete document
    await db_session.delete(doc)
    await db_session.flush()

    # Verify cascade deletion
    c_after = await db_session.execute(
        text("SELECT count(*) FROM chunks WHERE document_id = :did;"), {"did": doc.id}
    )
    hd_after = await db_session.execute(
        text("SELECT count(*) FROM hub_documents WHERE document_id = :did;"), {"did": doc.id}
    )
    assert c_after.scalar_one() == 0
    assert hd_after.scalar_one() == 0
