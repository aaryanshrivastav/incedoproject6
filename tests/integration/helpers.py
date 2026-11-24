"""Test helpers for building hubs, documents, and chunks with controlled data."""

import hashlib
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from kb.config import get_settings
from kb.db.models import ChunkModel, DocumentModel, HubModel
from kb.db.repositories import ChunkRepository, DocumentRepository, HubRepository
from kb.domain.models import NewChunk


def make_deterministic_embedding(seed_text: str, dimension: int | None = None) -> list[float]:
    """Generate a normalized, deterministic embedding vector from a seed text."""
    if dimension is None:
        dimension = get_settings().EMBEDDING_DIM or 768

    # Generate pseudo-random numbers from MD5 hash of seed_text
    h = hashlib.sha256(seed_text.encode("utf-8")).digest()
    vals = [float((b % 100) - 50) for b in h]
    # Tile up to dimension
    while len(vals) < dimension:
        vals.extend(vals[: dimension - len(vals)])
    vals = vals[:dimension]

    # Normalize to unit length for cosine similarity
    norm = sum(v * v for v in vals) ** 0.5
    if norm == 0:
        return [1.0] + [0.0] * (dimension - 1)
    return [round(v / norm, 6) for v in vals]


async def create_test_hub(
    session: AsyncSession,
    org_id: UUID,
    name: str = "test-hub",
) -> HubModel:
    """Create or fetch a test hub."""
    repo = HubRepository(session)
    return await repo.get_or_create(org_id, name)


async def create_test_document(
    session: AsyncSession,
    org_id: UUID,
    title: str = "Test Document",
    source_uri: str | None = None,
    scope: str = "hub",
    acl_tags: list[str] | None = None,
    status: str = "active",
    hub_id: UUID | None = None,
    content_hash: str = "test-content-hash",
) -> DocumentModel:
    """Create a document with controlled metadata and optional hub membership."""
    doc_repo = DocumentRepository(session)
    actual_source_uri = source_uri or f"file:///test/doc-{uuid4()}.txt"
    doc = await doc_repo.create(
        org_id=org_id,
        source_uri=actual_source_uri,
        title=title,
        content_hash=content_hash,
        scope=scope,
        acl_tags=acl_tags or [],
        version=1,
        owner_hub_id=hub_id,
    )
    if status != "active":
        doc.status = status
        await session.flush()

    if hub_id is not None:
        await doc_repo.add_to_hub(doc.id, hub_id)

    return doc


async def create_test_chunk(
    session: AsyncSession,
    document_id: UUID,
    text: str,
    ordinal: int = 1,
    embedding: list[float] | None = None,
    is_active: bool = True,
    page: int | None = 1,
    section: str | None = "Section 1",
) -> ChunkModel:
    """Create a chunk with controlled text and embedding."""
    if embedding is None:
        embedding = make_deterministic_embedding(text)

    new_chunk = NewChunk(
        ordinal=ordinal,
        text=text,
        page=page,
        section=section,
        token_count=len(text.split()),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        metadata={"seed": text[:20]},
        embedding=embedding,
    )
    chunk_repo = ChunkRepository(session)
    chunks = await chunk_repo.add_many(document_id, [new_chunk])
    chunk = chunks[0]
    if not is_active:
        chunk.is_active = False
        await session.flush()
    return chunk
