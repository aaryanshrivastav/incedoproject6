"""Integration tests for HubRepository, DocumentRepository, and ChunkRepository."""

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from kb.db.repositories import ChunkRepository, DocumentRepository, HubRepository
from kb.domain.models import NewChunk


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hub_repository_get_or_create(db_session: AsyncSession) -> None:
    """HubRepository retrieves existing hub or creates a new one idempotently."""
    repo = HubRepository(db_session)
    org_id = uuid4()

    # 1. First call creates the hub
    hub1 = await repo.get_or_create(org_id, "Marketing")
    assert hub1.name == "Marketing"
    assert hub1.org_id == org_id
    assert hub1.id is not None

    # 2. Second call returns the existing hub
    hub2 = await repo.get_or_create(org_id, "Marketing")
    assert hub2.id == hub1.id

    # 3. Same name in a different org creates a separate hub
    other_org = uuid4()
    hub_other = await repo.get_or_create(other_org, "Marketing")
    assert hub_other.id != hub1.id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_document_repository_lifecycle_and_versioning(db_session: AsyncSession) -> None:
    """DocumentRepository handles creation, versions, superseded state, and failures."""
    repo = DocumentRepository(db_session)
    org_id = uuid4()
    source_uri = "s3://bucket/docs/spec.pdf"

    # 1. next_version starts at 1
    assert await repo.next_version(org_id, source_uri) == 1

    # 2. Create active document v1
    doc_v1 = await repo.create(
        org_id=org_id,
        source_uri=source_uri,
        title="Spec v1",
        content_hash="hash_v1",
        scope="hub",
        acl_tags=["engineering"],
        version=1,
    )
    assert doc_v1.version == 1
    assert doc_v1.status == "active"

    # Active lookup
    active_doc = await repo.get_active_by_source(org_id, source_uri)
    assert active_doc is not None
    assert active_doc.id == doc_v1.id

    # 3. Mark superseded
    await repo.mark_superseded(doc_v1.id)
    assert await repo.get_active_by_source(org_id, source_uri) is None

    # 4. next_version is now 2
    assert await repo.next_version(org_id, source_uri) == 2

    # 5. Create active document v2
    doc_v2 = await repo.create(
        org_id=org_id,
        source_uri=source_uri,
        title="Spec v2",
        content_hash="hash_v2",
        scope="global",
        acl_tags=["engineering", "public"],
        version=2,
    )
    assert doc_v2.version == 2
    assert doc_v2.status == "active"

    # 6. Record failure creates a failed record with incremented version
    doc_failed = await repo.record_failure(
        org_id=org_id,
        source_uri=source_uri,
        title="Spec v3 (Corrupt)",
        scope="hub",
        acl_tags=["engineering"],
        reason="Corrupt PDF bytes",
    )
    assert doc_failed.version == 3
    assert doc_failed.status == "failed"
    assert doc_failed.failure_reason == "Corrupt PDF bytes"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_document_repository_metadata_hub_and_list(db_session: AsyncSession) -> None:
    """DocumentRepository updates metadata, manages hub junction, and lists documents."""
    hub_repo = HubRepository(db_session)
    doc_repo = DocumentRepository(db_session)
    org_id = uuid4()

    hub_a = await hub_repo.get_or_create(org_id, "Hub A")
    hub_b = await hub_repo.get_or_create(org_id, "Hub B")

    # Create document
    doc = await doc_repo.create(
        org_id=org_id,
        source_uri="file://a.txt",
        title="Doc A",
        content_hash="ha",
        scope="hub",
        acl_tags=["confidential"],
    )

    # 1. Update metadata
    updated_doc = await doc_repo.update_metadata(
        document_id=doc.id,
        scope="global",
        acl_tags=["confidential", "reviewed"],
    )
    assert updated_doc.scope == "global"
    assert set(updated_doc.acl_tags) == {"confidential", "reviewed"}

    # 2. Add to hubs
    await doc_repo.add_to_hub(doc.id, hub_a.id)
    await doc_repo.add_to_hub(doc.id, hub_b.id)

    # 3. List with filters
    all_docs = await doc_repo.list(org_id)
    assert len(all_docs) == 1

    hub_a_docs = await doc_repo.list(org_id, hub_id=hub_a.id)
    assert len(hub_a_docs) == 1
    assert hub_a_docs[0].id == doc.id

    other_hub = await hub_repo.get_or_create(org_id, "Empty Hub")
    empty_docs = await doc_repo.list(org_id, hub_id=other_hub.id)
    assert len(empty_docs) == 0

    active_only = await doc_repo.list(org_id, status="active")
    assert len(active_only) == 1
    failed_only = await doc_repo.list(org_id, status="failed")
    assert len(failed_only) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunk_repository_add_deactivate_and_embedding_cache(
    db_session: AsyncSession,
) -> None:
    """ChunkRepository adds chunks, deactivates them, and caches embeddings."""
    doc_repo = DocumentRepository(db_session)
    chunk_repo = ChunkRepository(db_session)
    org_id = uuid4()

    doc = await doc_repo.create(
        org_id=org_id,
        source_uri="file://chunks.txt",
        title="Chunks Doc",
        content_hash="dhash",
        scope="hub",
    )

    # 1. Add chunks
    dummy_vec1 = [0.1] * 768
    dummy_vec2 = [0.2] * 768
    chunks_data = [
        NewChunk(
            ordinal=0,
            text="First chunk text",
            page=1,
            section="Intro",
            token_count=3,
            content_hash="chunk_hash_1",
            metadata={"embedding_model": "BAAI/bge-base-en-v1.5"},
            embedding=dummy_vec1,
        ),
        NewChunk(
            ordinal=1,
            text="Second chunk text",
            page=2,
            section="Body",
            token_count=3,
            content_hash="chunk_hash_2",
            metadata={"embedding_model": "different-embedder"},
            embedding=dummy_vec2,
        ),
    ]

    added = await chunk_repo.add_many(doc.id, chunks_data)
    assert len(added) == 2

    # 2. Count active chunks
    assert await chunk_repo.count(doc.id, active_only=True) == 2
    assert await chunk_repo.count(doc.id, active_only=False) == 2

    # 3. Test embeddings_by_hash filtering
    # Must retrieve chunk_hash_1 matching "BAAI/bge-base-en-v1.5"
    cached = await chunk_repo.embeddings_by_hash(
        org_id=org_id,
        hashes=["chunk_hash_1", "chunk_hash_2", "non_existent_hash"],
        embedding_model="BAAI/bge-base-en-v1.5",
    )
    assert "chunk_hash_1" in cached
    assert len(cached["chunk_hash_1"]) == 768
    assert cached["chunk_hash_1"][0] == pytest.approx(0.1, rel=1e-4)

    # chunk_hash_2 used "different-embedder", so it must NOT be returned
    assert "chunk_hash_2" not in cached

    # Different org_id must NOT see these cached embeddings
    other_org = uuid4()
    other_cached = await chunk_repo.embeddings_by_hash(
        org_id=other_org,
        hashes=["chunk_hash_1"],
        embedding_model="BAAI/bge-base-en-v1.5",
    )
    assert len(other_cached) == 0

    # 4. Deactivate chunks
    await chunk_repo.deactivate_for_document(doc.id)
    assert await chunk_repo.count(doc.id, active_only=True) == 0
    assert await chunk_repo.count(doc.id, active_only=False) == 2
