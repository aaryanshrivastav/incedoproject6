"""Context assembly module for deduplication, budgeting, and deterministic ordering."""

import logging
from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kb.domain.retrieval import (
    AccessContext,
    ContextBundle,
    ContextConfig,
    ContextItem,
    ScoredChunk,
)
from kb.ingestion.tokens import ApproxTokenCounter, TokenCounter
from kb.retrieval.parents import get_eligible_parent_chunks

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Assembles retrieved and scored chunks into a bounded, deduplicated context bundle."""

    def __init__(
        self,
        counter: TokenCounter | None = None,
        config: ContextConfig | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        """Initialize assembler with token counter, config, and optional session factory."""
        self.counter = counter or ApproxTokenCounter()
        self.config = config or ContextConfig()
        self.session_factory = session_factory

    async def assemble(
        self,
        chunks: list[ScoredChunk],
        access: AccessContext | None = None,
    ) -> ContextBundle:
        """Assemble candidate chunks into a deterministic ContextBundle.

        Args:
            chunks: Scored candidate chunks from retriever.
            access: Verified AccessContext (required when expand_parents=True).

        Returns:
            ContextBundle satisfying budget, deduplication, and ordering constraints.
        """
        if not chunks:
            return ContextBundle(
                items=[],
                total_tokens=0,
                dropped=[],
                top_score=None,
                reranked=False,
            )

        # 1. Determine reranked state and top_score across original candidate inputs
        reranked = any(c.rerank_score is not None for c in chunks)
        top_score: float | None = None
        if reranked:
            rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
            top_score = max(rerank_scores) if rerank_scores else None
        else:
            sims = [c.vector_similarity for c in chunks if c.vector_similarity is not None]
            top_score = max(sims) if sims else None

        current_chunks = list(chunks)
        dropped: list[dict[str, Any]] = []

        # 2. Parent expansion (when enabled and DB session available)
        if self.config.expand_parents and self.session_factory is not None and access is not None:
            parent_ids = [
                c.parent_chunk_id for c in current_chunks if c.parent_chunk_id is not None
            ]
            if parent_ids:
                async with self.session_factory() as session:
                    eligible_parents = await get_eligible_parent_chunks(session, parent_ids, access)

                expanded_list: list[ScoredChunk] = []
                for c in current_chunks:
                    if c.parent_chunk_id is not None and c.parent_chunk_id in eligible_parents:
                        p_model = eligible_parents[c.parent_chunk_id]
                        expanded_list.append(
                            ScoredChunk(
                                chunk_id=p_model.id,
                                document_id=p_model.document_id,
                                document_title=c.document_title,
                                source_uri=c.source_uri,
                                text=p_model.text,
                                page=p_model.page,
                                section=p_model.section,
                                ordinal=p_model.ordinal,
                                content_hash=p_model.content_hash,
                                parent_chunk_id=p_model.parent_chunk_id,
                                vector_similarity=c.vector_similarity,
                                fts_rank_score=c.fts_rank_score,
                                vector_rank=c.vector_rank,
                                fts_rank=c.fts_rank,
                                fused_score=c.fused_score,
                                rerank_score=c.rerank_score,
                                final_score=c.final_score,
                            )
                        )
                    else:
                        expanded_list.append(c)
                current_chunks = expanded_list

        # 3. Deduplicate by content_hash across documents
        # Sort descending by final_score, breaking ties deterministically by chunk_id string
        sorted_for_dedupe = sorted(
            current_chunks,
            key=lambda c: (-c.final_score, str(c.chunk_id)),
        )

        seen_hashes: set[str] = set()
        deduped_chunks: list[ScoredChunk] = []
        for c in sorted_for_dedupe:
            if c.content_hash in seen_hashes:
                dropped.append({"chunk_id": str(c.chunk_id), "reason": "duplicate"})
            else:
                seen_hashes.add(c.content_hash)
                deduped_chunks.append(c)

        # 4. Greedy budget fill in descending final_score order
        # Chunks are tested in descending final_score order (tie-break chunk_id)
        sorted_for_budget = sorted(
            deduped_chunks,
            key=lambda c: (-c.final_score, str(c.chunk_id)),
        )

        budget_chunks: list[ScoredChunk] = []
        total_tokens = 0
        budget = self.config.budget_tokens

        for c in sorted_for_budget:
            tokens = self.counter.count_tokens(c.text)
            if total_tokens + tokens <= budget:
                total_tokens += tokens
                budget_chunks.append(c)
            else:
                dropped.append({"chunk_id": str(c.chunk_id), "reason": "budget"})

        # 5. Final presentation order:
        # - Documents ordered by their best chunk's final_score descending (tie-break document_id)
        # - Chunks within a document by ordinal ascending (tie-break chunk_id)
        # Invariant D-021: Adjacent chunks keep their natural overlap text; do NOT trim overlap.
        doc_chunks: dict[UUID, list[ScoredChunk]] = defaultdict(list)
        for c in budget_chunks:
            doc_chunks[c.document_id].append(c)

        doc_best_scores: list[tuple[float, str, UUID]] = []
        for doc_id, c_list in doc_chunks.items():
            best_score = max(c.final_score for c in c_list)
            doc_best_scores.append((-best_score, str(doc_id), doc_id))

        doc_best_scores.sort()

        final_items: list[ContextItem] = []
        for _neg_score, _doc_id_str, doc_id in doc_best_scores:
            c_list = doc_chunks[doc_id]
            # Order within document by ordinal ascending, then chunk_id string
            c_list.sort(key=lambda c: (c.ordinal, str(c.chunk_id)))
            for c in c_list:
                final_items.append(
                    ContextItem(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        document_title=c.document_title,
                        source_uri=c.source_uri,
                        page=c.page,
                        section=c.section,
                        ordinal=c.ordinal,
                        text=c.text,
                        content_hash=c.content_hash,
                        final_score=c.final_score,
                    )
                )

        return ContextBundle(
            items=final_items,
            total_tokens=total_tokens,
            dropped=dropped,
            top_score=top_score,
            reranked=reranked,
        )
