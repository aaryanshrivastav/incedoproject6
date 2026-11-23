"""End-to-end document ingestion pipeline with idempotent versioning and caching."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kb.db.repositories import ChunkRepository, DocumentRepository, HubRepository
from kb.db.session import get_session_factory
from kb.domain.documents import FailureReason, ParseError
from kb.domain.models import DocumentScope, NewChunk
from kb.domain.protocols import EmbedderProtocol
from kb.ingestion.chunkers.recursive import ChunkingConfig, chunk
from kb.ingestion.normalize import document_content_hash
from kb.ingestion.parsers.registry import get_parser
from kb.ingestion.tokens import ApproxTokenCounter, TokenCounter


class IngestStatus(StrEnum):
    """Result status of ingesting a single document file."""

    CREATED = "created"
    UPDATED = "updated"
    SKIPPED = "skipped"
    METADATA_UPDATED = "metadata_updated"
    FAILED = "failed"


class IngestRequest(BaseModel):
    """Parameters specifying organizational context and document attributes for ingestion."""

    model_config = ConfigDict(frozen=True)

    org_id: UUID
    hub_name: str = "default"
    scope: DocumentScope = DocumentScope.HUB
    acl_tags: list[str] = Field(default_factory=list)
    source_root: Path | None = None


class IngestResult(BaseModel):
    """Detailed result of an ingestion operation for one source file."""

    model_config = ConfigDict(frozen=True)

    status: IngestStatus
    source_uri: str
    document_id: UUID | None = None
    version: int = 0
    chunks_created: int = 0
    reason: str | None = None


class IngestSummary(BaseModel):
    """Aggregate summary of batch ingestion across a file or directory."""

    model_config = ConfigDict(frozen=True)

    total_files: int = 0
    created: int = 0
    updated: int = 0
    skipped: int = 0
    metadata_updated: int = 0
    failed: int = 0
    chunks_created: int = 0
    failures: list[tuple[str, str]] = Field(default_factory=list)


@asynccontextmanager
async def _scoped_session(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Helper providing atomic transaction scope for a given session factory."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


class IngestionPipeline:
    """Orchestrates parsing, normalization, chunking, caching, embedding, and atomic storage."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        token_counter: TokenCounter | None = None,
        chunk_config: ChunkingConfig | None = None,
        batch_size: int = 32,
    ) -> None:
        """Initialize pipeline with embedder and transactional dependencies."""
        self.embedder = embedder
        self.session_factory = (
            session_factory if session_factory is not None else get_session_factory()
        )
        self.token_counter = token_counter if token_counter is not None else ApproxTokenCounter()
        self.chunk_config = chunk_config if chunk_config is not None else ChunkingConfig()
        self.batch_size = max(1, batch_size)

    def resolve_source_uri(self, path: Path, source_root: Path | None = None) -> str:
        """Resolve a consistent, canonical source URI for a file path."""
        resolved = path.resolve()
        if source_root is not None:
            root_resolved = source_root.resolve()
            try:
                rel = resolved.relative_to(root_resolved)
                return rel.as_posix()
            except ValueError:
                pass
        return resolved.as_uri()

    async def ingest_file(self, path: Path, request: IngestRequest) -> IngestResult:
        """Ingest a single document file idempotently into the knowledge repository."""
        source_uri = self.resolve_source_uri(path, request.source_root)

        # 1. Parsing and Normalization
        try:
            parser = get_parser(path)
            parsed = parser.parse(path)
        except ParseError as pe:
            # Record failed document in its own transaction (Invariant: failure is persisted)
            async with _scoped_session(self.session_factory) as session:
                hub_repo = HubRepository(session)
                doc_repo = DocumentRepository(session)
                hub = await hub_repo.get_or_create(request.org_id, request.hub_name)
                failed_doc = await doc_repo.record_failure(
                    org_id=request.org_id,
                    source_uri=source_uri,
                    title=path.name,
                    scope=request.scope.value,
                    acl_tags=request.acl_tags,
                    reason=pe.reason.value,
                )
                await doc_repo.add_to_hub(failed_doc.id, hub.id)

            return IngestResult(
                status=IngestStatus.FAILED,
                source_uri=source_uri,
                document_id=failed_doc.id,
                version=failed_doc.version,
                chunks_created=0,
                reason=pe.reason.value,
            )
        except Exception as e:
            async with _scoped_session(self.session_factory) as session:
                hub_repo = HubRepository(session)
                doc_repo = DocumentRepository(session)
                hub = await hub_repo.get_or_create(request.org_id, request.hub_name)
                failed_doc = await doc_repo.record_failure(
                    org_id=request.org_id,
                    source_uri=source_uri,
                    title=path.name,
                    scope=request.scope.value,
                    acl_tags=request.acl_tags,
                    reason=FailureReason.PARSE_ERROR.value,
                )
                await doc_repo.add_to_hub(failed_doc.id, hub.id)

            return IngestResult(
                status=IngestStatus.FAILED,
                source_uri=source_uri,
                document_id=failed_doc.id,
                version=failed_doc.version,
                chunks_created=0,
                reason=f"parse_error: {e}",
            )

        doc_hash = document_content_hash(parsed)

        # 2. Check for existing active document
        async with _scoped_session(self.session_factory) as session:
            doc_repo = DocumentRepository(session)
            hub_repo = HubRepository(session)
            active_doc = await doc_repo.get_active_by_source(request.org_id, source_uri)
            hub = await hub_repo.get_or_create(request.org_id, request.hub_name)

        # 3. Idempotent check: exact content match
        if active_doc is not None and active_doc.content_hash == doc_hash:
            # Check if metadata or hub membership changed
            scope_changed = active_doc.scope != request.scope.value
            acl_changed = sorted(active_doc.acl_tags) != sorted(request.acl_tags)

            if scope_changed or acl_changed:
                async with _scoped_session(self.session_factory) as session:
                    d_repo = DocumentRepository(session)
                    await d_repo.update_metadata(
                        document_id=active_doc.id,
                        scope=request.scope.value if scope_changed else None,
                        acl_tags=request.acl_tags if acl_changed else None,
                    )
                    await d_repo.add_to_hub(active_doc.id, hub.id)

                return IngestResult(
                    status=IngestStatus.METADATA_UPDATED,
                    source_uri=source_uri,
                    document_id=active_doc.id,
                    version=active_doc.version,
                    chunks_created=0,
                )

            # Re-ensure hub membership even if metadata identical
            async with _scoped_session(self.session_factory) as session:
                d_repo = DocumentRepository(session)
                await d_repo.add_to_hub(active_doc.id, hub.id)

            return IngestResult(
                status=IngestStatus.SKIPPED,
                source_uri=source_uri,
                document_id=active_doc.id,
                version=active_doc.version,
                chunks_created=0,
            )

        # 4. Generate Chunk Drafts
        chunk_drafts = chunk(parsed, self.chunk_config, self.token_counter)
        if not chunk_drafts:
            async with _scoped_session(self.session_factory) as session:
                doc_repo = DocumentRepository(session)
                failed_doc = await doc_repo.record_failure(
                    org_id=request.org_id,
                    source_uri=source_uri,
                    title=parsed.title,
                    scope=request.scope.value,
                    acl_tags=request.acl_tags,
                    reason=FailureReason.EMPTY.value,
                )
            return IngestResult(
                status=IngestStatus.FAILED,
                source_uri=source_uri,
                document_id=failed_doc.id,
                version=failed_doc.version,
                chunks_created=0,
                reason=FailureReason.EMPTY.value,
            )

        # 5. Embedding with Hash-Based Reuse (Happens BEFORE the write transaction opens)
        all_chunk_hashes = [d.content_hash for d in chunk_drafts]
        cached_embeddings: dict[str, list[float]] = {}

        async with _scoped_session(self.session_factory) as session:
            chunk_repo = ChunkRepository(session)
            cached_embeddings = await chunk_repo.embeddings_by_hash(
                org_id=request.org_id,
                hashes=all_chunk_hashes,
                embedding_model=self.embedder.model_name,
            )

        # Group drafts needing new embeddings
        texts_to_embed: list[str] = []
        hashes_to_embed: list[str] = []

        for draft in chunk_drafts:
            if (
                draft.content_hash not in cached_embeddings
                and draft.content_hash not in hashes_to_embed
            ):
                texts_to_embed.append(draft.text)
                hashes_to_embed.append(draft.content_hash)

        newly_computed_embeddings: dict[str, list[float]] = {}
        if texts_to_embed:
            # Batch embedding
            for i in range(0, len(texts_to_embed), self.batch_size):
                batch = texts_to_embed[i : i + self.batch_size]
                batch_hashes = hashes_to_embed[i : i + self.batch_size]

                try:
                    vectors = await self.embedder.embed(batch)
                except Exception as e:
                    async with _scoped_session(self.session_factory) as session:
                        doc_repo = DocumentRepository(session)
                        failed_doc = await doc_repo.record_failure(
                            org_id=request.org_id,
                            source_uri=source_uri,
                            title=parsed.title,
                            scope=request.scope.value,
                            acl_tags=request.acl_tags,
                            reason="embed_error",
                        )
                    return IngestResult(
                        status=IngestStatus.FAILED,
                        source_uri=source_uri,
                        document_id=failed_doc.id,
                        version=failed_doc.version,
                        chunks_created=0,
                        reason=f"embed_error: {e}",
                    )

                # Validate vector dimension (Invariant: correct vector size)
                for h, vec in zip(batch_hashes, vectors, strict=True):
                    if len(vec) != self.embedder.dimension:
                        async with _scoped_session(self.session_factory) as session:
                            doc_repo = DocumentRepository(session)
                            failed_doc = await doc_repo.record_failure(
                                org_id=request.org_id,
                                source_uri=source_uri,
                                title=parsed.title,
                                scope=request.scope.value,
                                acl_tags=request.acl_tags,
                                reason="embed_error",
                            )
                        return IngestResult(
                            status=IngestStatus.FAILED,
                            source_uri=source_uri,
                            document_id=failed_doc.id,
                            version=failed_doc.version,
                            chunks_created=0,
                            reason=(
                                f"embed_error: vector dimension {len(vec)} "
                                f"!= expected {self.embedder.dimension}"
                            ),
                        )
                    newly_computed_embeddings[h] = vec

        # Assemble NewChunk domain objects
        new_chunks: list[NewChunk] = []
        for draft in chunk_drafts:
            vec = (
                cached_embeddings.get(draft.content_hash)
                or newly_computed_embeddings[draft.content_hash]
            )
            meta = {**draft.metadata, "embedding_model": self.embedder.model_name}
            new_chunks.append(
                NewChunk(
                    ordinal=draft.ordinal,
                    text=draft.text,
                    page=draft.page,
                    section=draft.section,
                    token_count=draft.token_count,
                    content_hash=draft.content_hash,
                    metadata=meta,
                    embedding=vec,
                )
            )

        # 6. Single Atomic Write Transaction
        async with _scoped_session(self.session_factory) as session:
            hub_repo = HubRepository(session)
            doc_repo = DocumentRepository(session)
            chunk_repo = ChunkRepository(session)

            hub = await hub_repo.get_or_create(request.org_id, request.hub_name)

            if active_doc is not None:
                new_version = await doc_repo.next_version(request.org_id, source_uri)
                await doc_repo.mark_superseded(active_doc.id)
                await chunk_repo.deactivate_for_document(active_doc.id)
                status = IngestStatus.UPDATED
            else:
                new_version = 1
                status = IngestStatus.CREATED

            new_doc = await doc_repo.create(
                org_id=request.org_id,
                source_uri=source_uri,
                title=parsed.title,
                content_hash=doc_hash,
                scope=request.scope.value,
                acl_tags=request.acl_tags,
                version=new_version,
                owner_hub_id=hub.id,
            )

            await doc_repo.add_to_hub(new_doc.id, hub.id)
            await chunk_repo.add_many(new_doc.id, new_chunks)

        return IngestResult(
            status=status,
            source_uri=source_uri,
            document_id=new_doc.id,
            version=new_version,
            chunks_created=len(new_chunks),
        )

    async def ingest_path(self, path: Path, request: IngestRequest) -> IngestSummary:
        """Recursively walk and ingest all supported files from a file or directory."""
        if not path.exists():
            return IngestSummary(
                total_files=0,
                failures=[(str(path), f"Path not found: {path}")],
            )

        # Collect files
        files_to_process: list[Path] = []
        if path.is_file():
            files_to_process.append(path)
        else:
            supported_suffixes = {".pdf", ".docx", ".html", ".htm", ".csv", ".xlsx", ".txt"}
            for p in sorted(path.rglob("*")):
                if p.is_file() and p.suffix.lower() in supported_suffixes:
                    files_to_process.append(p)

        created = 0
        updated = 0
        skipped = 0
        metadata_updated = 0
        failed = 0
        chunks_created = 0
        failures: list[tuple[str, str]] = []

        source_root = (
            request.source_root
            if request.source_root is not None
            else (path if path.is_dir() else path.parent)
        )
        req_with_root = IngestRequest(
            org_id=request.org_id,
            hub_name=request.hub_name,
            scope=request.scope,
            acl_tags=request.acl_tags,
            source_root=source_root,
        )

        for file_path in files_to_process:
            try:
                res = await self.ingest_file(file_path, req_with_root)
                if res.status == IngestStatus.CREATED:
                    created += 1
                    chunks_created += res.chunks_created
                elif res.status == IngestStatus.UPDATED:
                    updated += 1
                    chunks_created += res.chunks_created
                elif res.status == IngestStatus.SKIPPED:
                    skipped += 1
                elif res.status == IngestStatus.METADATA_UPDATED:
                    metadata_updated += 1
                elif res.status == IngestStatus.FAILED:
                    failed += 1
                    failures.append((res.source_uri, res.reason or "Unknown failure"))
            except Exception as e:
                failed += 1
                uri = self.resolve_source_uri(file_path, req_with_root.source_root)
                failures.append((uri, f"Unexpected error: {e}"))

        return IngestSummary(
            total_files=len(files_to_process),
            created=created,
            updated=updated,
            skipped=skipped,
            metadata_updated=metadata_updated,
            failed=failed,
            chunks_created=chunks_created,
            failures=failures,
        )
