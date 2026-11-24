"""Asynchronous repositories for database entity management."""

from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from kb.db.models import ChunkModel, DocumentModel, HubDocumentModel, HubModel
from kb.domain.models import NewChunk


class HubRepository:
    """Repository managing hub entities."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with an active asynchronous session."""
        self.session = session

    async def get_or_create(self, org_id: UUID, name: str) -> HubModel:
        """Retrieve an existing hub by name within an org or create a new one."""
        stmt = select(HubModel).where(HubModel.org_id == org_id, HubModel.name == name)
        res = await self.session.execute(stmt)
        hub = res.scalar_one_or_none()
        if hub is None:
            hub = HubModel(org_id=org_id, name=name)
            self.session.add(hub)
            await self.session.flush()
        return hub


class DocumentRepository:
    """Repository managing document entities, versions, and hub membership."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with an active asynchronous session."""
        self.session = session

    async def get_active_by_source(self, org_id: UUID, source_uri: str) -> DocumentModel | None:
        """Retrieve the currently active document for a source URI."""
        stmt = select(DocumentModel).where(
            DocumentModel.org_id == org_id,
            DocumentModel.source_uri == source_uri,
            DocumentModel.status == "active",
        )
        res = await self.session.execute(stmt)
        return res.scalar_one_or_none()

    async def next_version(self, org_id: UUID, source_uri: str) -> int:
        """Calculate the next integer version for a document source across all statuses."""
        stmt = select(func.coalesce(func.max(DocumentModel.version), 0)).where(
            DocumentModel.org_id == org_id,
            DocumentModel.source_uri == source_uri,
        )
        res = await self.session.execute(stmt)
        max_version = res.scalar_one()
        return int(max_version) + 1

    async def create(
        self,
        org_id: UUID,
        source_uri: str,
        title: str,
        content_hash: str,
        scope: str = "hub",
        acl_tags: list[str] | None = None,
        version: int = 1,
        owner_hub_id: UUID | None = None,
    ) -> DocumentModel:
        """Insert and return a new active document version."""
        doc = DocumentModel(
            org_id=org_id,
            source_uri=source_uri,
            title=title,
            content_hash=content_hash,
            version=version,
            status="active",
            scope=scope,
            acl_tags=acl_tags or [],
            owner_hub_id=owner_hub_id,
        )
        self.session.add(doc)
        await self.session.flush()
        return doc

    async def mark_superseded(self, document_id: UUID) -> None:
        """Mark a document version as superseded."""
        stmt = (
            update(DocumentModel).where(DocumentModel.id == document_id).values(status="superseded")
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def record_failure(
        self,
        org_id: UUID,
        source_uri: str,
        title: str,
        scope: str,
        acl_tags: list[str] | None,
        reason: str,
    ) -> DocumentModel:
        """Record a failed ingestion attempt as a failed document version."""
        version = await self.next_version(org_id, source_uri)
        doc = DocumentModel(
            org_id=org_id,
            source_uri=source_uri,
            title=title,
            content_hash="",
            version=version,
            status="failed",
            failure_reason=reason,
            scope=scope,
            acl_tags=acl_tags or [],
        )
        self.session.add(doc)
        await self.session.flush()
        return doc

    async def update_metadata(
        self,
        document_id: UUID,
        scope: str | None = None,
        acl_tags: list[str] | None = None,
    ) -> DocumentModel:
        """Update scope and/or ACL tags for an existing document."""
        values: dict[str, Any] = {}
        if scope is not None:
            values["scope"] = scope
        if acl_tags is not None:
            values["acl_tags"] = acl_tags

        if values:
            stmt = (
                update(DocumentModel)
                .where(DocumentModel.id == document_id)
                .values(**values)
                .returning(DocumentModel)
            )
            res = await self.session.execute(stmt)
            await self.session.flush()
            return res.scalar_one()

        select_stmt = select(DocumentModel).where(DocumentModel.id == document_id)
        select_res = await self.session.execute(select_stmt)
        return select_res.scalar_one()

    async def add_to_hub(self, document_id: UUID, hub_id: UUID) -> None:
        """Associate a document with a hub in hub_documents if not already associated."""
        stmt = select(HubDocumentModel).where(
            HubDocumentModel.hub_id == hub_id,
            HubDocumentModel.document_id == document_id,
        )
        res = await self.session.execute(stmt)
        if res.scalar_one_or_none() is None:
            hd = HubDocumentModel(hub_id=hub_id, document_id=document_id)
            self.session.add(hd)
            await self.session.flush()

    async def list(
        self,
        org_id: UUID,
        hub_id: UUID | None = None,
        status: str | None = None,
    ) -> list[DocumentModel]:
        """List documents matching org, optional hub, and optional status."""
        stmt = select(DocumentModel).where(DocumentModel.org_id == org_id)
        if status is not None:
            stmt = stmt.where(DocumentModel.status == status)
        if hub_id is not None:
            stmt = stmt.join(
                HubDocumentModel,
                HubDocumentModel.document_id == DocumentModel.id,
            ).where(HubDocumentModel.hub_id == hub_id)

        res = await self.session.execute(stmt.order_by(DocumentModel.created_at.desc()))
        return list(res.scalars().all())


class ChunkRepository:
    """Repository managing document text chunks and vector embeddings."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with an active asynchronous session."""
        self.session = session

    async def add_many(self, document_id: UUID, chunks: list[NewChunk]) -> list[ChunkModel]:
        """Insert multiple chunk records for a document."""
        chunk_models: list[ChunkModel] = []
        for c in chunks:
            m = ChunkModel(
                document_id=document_id,
                ordinal=c.ordinal,
                text=c.text,
                page=c.page,
                section=c.section,
                token_count=c.token_count,
                content_hash=c.content_hash,
                parent_chunk_id=c.parent_chunk_id,
                metadata_=c.metadata,
                is_active=True,
                embedding=c.embedding,
            )
            chunk_models.append(m)
            self.session.add(m)

        await self.session.flush()
        return chunk_models

    async def deactivate_for_document(self, document_id: UUID) -> None:
        """Deactivate all active chunks associated with a document ID."""
        stmt = (
            update(ChunkModel).where(ChunkModel.document_id == document_id).values(is_active=False)
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def count(self, document_id: UUID, active_only: bool = True) -> int:
        """Count chunks for a given document."""
        stmt = select(func.count(ChunkModel.id)).where(ChunkModel.document_id == document_id)
        if active_only:
            stmt = stmt.where(ChunkModel.is_active.is_(True))
        res = await self.session.execute(stmt)
        return int(res.scalar_one())

    async def embeddings_by_hash(
        self,
        org_id: UUID,
        hashes: list[str],
        embedding_model: str,
    ) -> dict[str, list[float]]:
        """Retrieve cached embeddings for matching content hashes and embedding model."""
        if not hashes:
            return {}

        stmt = (
            select(ChunkModel.content_hash, ChunkModel.embedding)
            .join(DocumentModel, DocumentModel.id == ChunkModel.document_id)
            .where(
                DocumentModel.org_id == org_id,
                ChunkModel.content_hash.in_(hashes),
                ChunkModel.metadata_["embedding_model"].astext == embedding_model,
            )
        )
        res = await self.session.execute(stmt)
        results: dict[str, list[float]] = {}
        for row in res.all():
            content_hash = str(row[0])
            raw_embedding = row[1]
            results[content_hash] = [float(x) for x in raw_embedding]
        return results
