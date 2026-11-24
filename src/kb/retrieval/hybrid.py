"""Access-filtered hybrid retrieval fusing vector search and full-text search with RRF."""

import logging
import time
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kb.config import get_settings
from kb.db.models import ChunkModel, DocumentModel
from kb.domain.protocols import EmbedderProtocol
from kb.domain.retrieval import AccessContext, RetrievalConfig, RetrievalError, RetrievedChunk
from kb.retrieval.filters import eligible_chunk_conditions

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Retrieves access-filtered candidate chunks by fusing vector and full-text search."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: EmbedderProtocol,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self._version_verified = False

    async def verify_pgvector_version(self, session: AsyncSession) -> str:
        """Verify that the database has pgvector extension version >= 0.8.0."""
        res = await session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        )
        version_str = res.scalar_one_or_none()
        if not version_str:
            raise RetrievalError("pgvector extension is not installed in PostgreSQL.")

        # Parse semver (e.g. '0.8.2' -> (0, 8, 2))
        parts: list[int] = []
        for p in version_str.split("."):
            try:
                parts.append(int(p))
            except ValueError:
                break
        if tuple(parts) < (0, 8, 0):
            raise RetrievalError(
                f"pgvector version {version_str} is below the required 0.8.0 for iterative scan."
            )
        self._version_verified = True
        return str(version_str)

    async def retrieve(self, query: str, access: AccessContext) -> list[RetrievedChunk]:
        """Retrieve top candidate chunks for a query under strict access constraints.

        Args:
            query: User search query string.
            access: Verified AccessContext (org_id, hub_id, user_tags). Required.

        Returns:
            List of candidate chunks sorted by descending RRF fused score.
        """
        start_time = time.perf_counter()

        # 1. Embed query and validate dimension
        settings = get_settings()
        expected_dim = settings.EMBEDDING_DIM
        embeddings = await self.embedder.embed([query])
        if not embeddings or len(embeddings[0]) != expected_dim:
            actual_dim = len(embeddings[0]) if embeddings else 0
            raise RetrievalError(
                f"Query vector dimension {actual_dim} does not match expected "
                f"EMBEDDING_DIM {expected_dim}."
            )
        query_vector = embeddings[0]

        async with self.session_factory() as session:
            # 2. Check pgvector version on first call
            if not self._version_verified:
                await self.verify_pgvector_version(session)

            # 3. Vector search inside transaction with relaxed_order iterative scan
            ef_search = self.config.effective_ef_search
            await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order;"))
            await session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search};"))

            # Inner subquery fetches candidate vectors using HNSW index with relaxed order
            dist_expr = ChunkModel.embedding.cosine_distance(query_vector).label("distance")
            vector_subquery = (
                select(
                    ChunkModel.id.label("chunk_id"),
                    ChunkModel.document_id.label("document_id"),
                    DocumentModel.title.label("document_title"),
                    DocumentModel.source_uri.label("source_uri"),
                    ChunkModel.text.label("text"),
                    ChunkModel.page.label("page"),
                    ChunkModel.section.label("section"),
                    ChunkModel.ordinal.label("ordinal"),
                    ChunkModel.content_hash.label("content_hash"),
                    ChunkModel.parent_chunk_id.label("parent_chunk_id"),
                    dist_expr,
                )
                .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
                .where(*eligible_chunk_conditions(access))
                .order_by(dist_expr.asc())
                .limit(self.config.vector_k)
                .subquery("sub_vector")
            )

            # Outer query guarantees exact distance re-sorting
            vector_stmt = select(vector_subquery).order_by(vector_subquery.c.distance.asc())
            vector_rows = (await session.execute(vector_stmt)).all()

            # 4. Full-text search (evaluate websearch_to_tsquery first)
            fts_rows: list[Any] = []
            tsq_check = await session.execute(select(func.websearch_to_tsquery("english", query)))
            raw_tsq = tsq_check.scalar()
            has_fts_query = raw_tsq is not None and str(raw_tsq).strip() != ""

            if has_fts_query:
                ts_query = func.websearch_to_tsquery("english", query)
                rank_expr = func.ts_rank_cd(ChunkModel.tsv, ts_query).label("rank_score")
                fts_stmt = (
                    select(
                        ChunkModel.id.label("chunk_id"),
                        ChunkModel.document_id.label("document_id"),
                        DocumentModel.title.label("document_title"),
                        DocumentModel.source_uri.label("source_uri"),
                        ChunkModel.text.label("text"),
                        ChunkModel.page.label("page"),
                        ChunkModel.section.label("section"),
                        ChunkModel.ordinal.label("ordinal"),
                        ChunkModel.content_hash.label("content_hash"),
                        ChunkModel.parent_chunk_id.label("parent_chunk_id"),
                        rank_expr,
                    )
                    .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
                    .where(
                        *eligible_chunk_conditions(access),
                        ChunkModel.tsv.op("@@")(ts_query),
                    )
                    .order_by(rank_expr.desc())
                    .limit(self.config.fts_k)
                )
                fts_rows = [
                    (
                        row.chunk_id,
                        row.document_id,
                        row.document_title,
                        row.source_uri,
                        row.text,
                        row.page,
                        row.section,
                        row.ordinal,
                        row.content_hash,
                        row.parent_chunk_id,
                        float(row.rank_score),
                    )
                    for row in (await session.execute(fts_stmt)).all()
                ]

        # 5. Build candidate maps and calculate Reciprocal Rank Fusion (RRF)
        vector_ranks: dict[UUID, int] = {}
        vector_sims: dict[UUID, float] = {}
        fts_ranks: dict[UUID, int] = {}
        fts_scores: dict[UUID, float] = {}
        payloads: dict[UUID, dict[str, Any]] = {}

        for rank_idx, row in enumerate(vector_rows, start=1):
            cid: UUID = row.chunk_id
            vector_ranks[cid] = rank_idx
            # Cosine distance to similarity: 1 - distance
            vector_sims[cid] = max(0.0, 1.0 - float(row.distance))
            if cid not in payloads:
                payloads[cid] = {
                    "document_id": row.document_id,
                    "document_title": row.document_title,
                    "source_uri": row.source_uri,
                    "text": row.text,
                    "page": row.page,
                    "section": row.section,
                    "ordinal": row.ordinal,
                    "content_hash": row.content_hash,
                    "parent_chunk_id": row.parent_chunk_id,
                }

        for rank_idx, fts_item in enumerate(fts_rows, start=1):
            cid = fts_item[0]
            fts_ranks[cid] = rank_idx
            fts_scores[cid] = fts_item[10]
            if cid not in payloads:
                payloads[cid] = {
                    "document_id": fts_item[1],
                    "document_title": fts_item[2],
                    "source_uri": fts_item[3],
                    "text": fts_item[4],
                    "page": fts_item[5],
                    "section": fts_item[6],
                    "ordinal": fts_item[7],
                    "content_hash": fts_item[8],
                    "parent_chunk_id": fts_item[9],
                }

        all_ids = set(vector_ranks.keys()) | set(fts_ranks.keys())
        rrf_k = self.config.rrf_k

        # Calculate fused score for each chunk
        fused_items: list[tuple[float, int, str, UUID]] = []
        for cid in all_ids:
            score = 0.0
            vr = vector_ranks.get(cid)
            fr = fts_ranks.get(cid)
            if vr is not None:
                score += 1.0 / (rrf_k + vr)
            if fr is not None:
                score += 1.0 / (rrf_k + fr)

            best_rank = min(vr or 999999, fr or 999999)
            # Sorting tuple: (-fused_score, best_rank, str(chunk_id))
            fused_items.append((-score, best_rank, str(cid), cid))

        # Sort deterministically
        fused_items.sort()

        # Build final RetrievedChunk list up to fused_k
        results: list[RetrievedChunk] = []
        for neg_score, _best_rank, _id_str, cid in fused_items[: self.config.fused_k]:
            data = payloads[cid]
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_id=data["document_id"],
                    document_title=str(data["document_title"]),
                    source_uri=str(data["source_uri"]),
                    text=str(data["text"]),
                    page=data["page"],
                    section=data["section"],
                    ordinal=int(data["ordinal"]),
                    content_hash=str(data["content_hash"]),
                    parent_chunk_id=data["parent_chunk_id"],
                    vector_similarity=vector_sims.get(cid),
                    fts_rank_score=fts_scores.get(cid),
                    vector_rank=vector_ranks.get(cid),
                    fts_rank=fts_ranks.get(cid),
                    fused_score=-neg_score,
                )
            )

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        # Invariant I4: Never log raw query text or chunk text at INFO or above
        logger.info(
            "Hybrid retrieval: vector_hits=%d, fts_hits=%d, fused_hits=%d, duration_ms=%.2f, "
            "top_ids=%s",
            len(vector_rows),
            len(fts_rows),
            len(results),
            duration_ms,
            [str(c.chunk_id) for c in results[:5]],
        )

        return results
