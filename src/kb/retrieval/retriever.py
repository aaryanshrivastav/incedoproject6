"""End-to-end retriever orchestrating hybrid retrieval and reranking."""

from kb.domain.retrieval import AccessContext, ScoredChunk
from kb.retrieval.hybrid import HybridRetriever
from kb.retrieval.rerank import RerankStage


class Retriever:
    """Orchestrates hybrid candidate retrieval and subsequent candidate reranking."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        rerank_stage: RerankStage,
    ) -> None:
        """Initialize retriever with hybrid retrieval and reranking components."""
        self.hybrid_retriever = hybrid_retriever
        self.rerank_stage = rerank_stage

    async def retrieve(
        self,
        query: str,
        access: AccessContext,
    ) -> list[ScoredChunk]:
        """Execute hybrid search and candidate reranking under access controls.

        Args:
            query: User search query.
            access: Verified AccessContext (org_id, hub_id, user_tags). Required.

        Returns:
            List of ScoredChunk candidates sorted by final relevance score.
        """
        # 1. Fetch candidate chunks from vector and full-text search fused by RRF
        candidates = await self.hybrid_retriever.retrieve(query, access)

        # 2. Score and reorder candidates via reranking stage
        return await self.rerank_stage.rerank(query, candidates)
