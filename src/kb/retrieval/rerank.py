"""Candidate reranking stage using RerankerProtocol."""

import logging
import time

from kb.domain.protocols import RerankerProtocol
from kb.domain.retrieval import RerankConfig, RerankError, RetrievedChunk, ScoredChunk

logger = logging.getLogger(__name__)


class RerankStage:
    """Reranks candidate chunks using a cross-encoder or reranker model."""

    def __init__(
        self,
        reranker: RerankerProtocol,
        config: RerankConfig | None = None,
    ) -> None:
        """Initialize reranker stage with protocol implementation and configuration."""
        self.reranker = reranker
        self.config = config or RerankConfig()

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[ScoredChunk]:
        """Rerank candidates based on semantic relevance to query.

        Args:
            query: User search query.
            candidates: Retrieved candidate chunks from hybrid retrieval.

        Returns:
            List of ScoredChunks sorted by relevance, truncated to config.top_n.

        Raises:
            RerankError: If the reranker invocation fails.
        """
        start_time = time.perf_counter()

        if not candidates:
            return []

        # 1. Ablation mode: reranking disabled -> preserve fused order and use fused_score
        if not self.config.enabled:
            top_candidates = candidates[: self.config.top_n]
            return [
                ScoredChunk(
                    **c.model_dump(),
                    rerank_score=None,
                    final_score=c.fused_score,
                )
                for c in top_candidates
            ]

        # 2. Reranking enabled -> batched call through RerankerProtocol
        candidate_texts = [c.text for c in candidates]
        try:
            scored_candidates = await self.reranker.rerank(query, candidate_texts)
        except Exception as e:
            logger.error("Reranker invocation failed: %s", type(e).__name__)
            raise RerankError(f"Reranker failed to score candidates: {e}") from e

        # Match scored results back to candidates by index
        scored_by_index = {sc.index: sc.score for sc in scored_candidates}

        scored_chunks: list[ScoredChunk] = []
        for idx, candidate in enumerate(candidates):
            rerank_score = scored_by_index.get(idx, 0.0)
            scored_chunks.append(
                ScoredChunk(
                    **candidate.model_dump(),
                    rerank_score=rerank_score,
                    final_score=rerank_score,
                )
            )

        # Sort descending by rerank_score, tie-break by fused_score descending then chunk_id
        scored_chunks.sort(
            key=lambda c: (
                -(c.rerank_score if c.rerank_score is not None else 0.0),
                -c.fused_score,
                str(c.chunk_id),
            )
        )

        final_chunks = scored_chunks[: self.config.top_n]
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Invariant I4: Never log raw query text or chunk text at INFO
        logger.info(
            "Rerank stage completed: input_count=%d, output_count=%d, duration_ms=%.2f, top_ids=%s",
            len(candidates),
            len(final_chunks),
            duration_ms,
            [str(c.chunk_id) for c in final_chunks[:5]],
        )

        return final_chunks
