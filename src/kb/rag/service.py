"""High-level RAG service coordinating retrieval, context assembly, and answer generation."""

import logging

from kb.domain.answers import AnswerResult, RefusalReason
from kb.domain.retrieval import AccessContext
from kb.generation.generator import Generator
from kb.retrieval.context import ContextAssembler
from kb.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class RagService:
    """Service orchestrating end-to-end question answering pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        assembler: ContextAssembler,
        generator: Generator,
        refusal_min_score: float = 0.0,
    ) -> None:
        """Initialize RAG service with retrieval, assembly, and generation components."""
        self.retriever = retriever
        self.assembler = assembler
        self.generator = generator
        self.refusal_min_score = refusal_min_score

        if self.refusal_min_score == 0.0:
            logger.warning("REFUSAL_MIN_SCORE is 0.0: relevance-based refusal gating is disabled.")

    async def ask(
        self,
        question: str,
        access: AccessContext,
    ) -> AnswerResult:
        """Execute end-to-end RAG answer flow under access control constraints.

        Args:
            question: User inquiry string.
            access: Verified AccessContext (org_id, hub_id, user_tags).

        Returns:
            AnswerResult model with answer, citations, or refusal reason.
        """
        # 1. Access-controlled hybrid retrieval + reranking
        candidates = await self.retriever.retrieve(question, access)

        # 2. Context assembly (deduplication, budget, ordering)
        bundle = await self.assembler.assemble(candidates, access)

        model_name = getattr(self.generator.client._llm, "model_name", "unknown")

        # 3. Pre-LLM short-circuit refusals
        # Empty context check
        if not bundle.items:
            logger.info("Pre-check refusal: no eligible context found for query.")
            return AnswerResult(
                answer=(
                    "I cannot answer this question because no relevant documents "
                    "were found in your accessible knowledge base."
                ),
                citations=[],
                refusal=True,
                refusal_reason=RefusalReason.no_context,
                invalid_citations=[],
                prompt_version=self.generator.prompt_version,
                model=model_name,
                context_chunk_ids=[],
            )

        # Low relevance score check
        if self.refusal_min_score > 0.0:
            if bundle.top_score is None or bundle.top_score < self.refusal_min_score:
                logger.info(
                    "Pre-check refusal: top score %.4f is below threshold %.4f.",
                    bundle.top_score or 0.0,
                    self.refusal_min_score,
                )
                return AnswerResult(
                    answer=(
                        "I cannot answer this question because the retrieved evidence "
                        "is below the required relevance confidence threshold."
                    ),
                    citations=[],
                    refusal=True,
                    refusal_reason=RefusalReason.low_relevance,
                    invalid_citations=[],
                    prompt_version=self.generator.prompt_version,
                    model=model_name,
                    context_chunk_ids=[item.chunk_id for item in bundle.items],
                )

        # 4. Generate cited answer via LLM
        return await self.generator.generate(question, bundle)
