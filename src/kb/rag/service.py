"""High-level RAG service coordinating retrieval, context assembly, and answer generation."""

import logging

from pydantic import BaseModel, Field

from kb.domain.answers import AnswerResult, RefusalReason
from kb.domain.retrieval import AccessContext, ContextBundle, ScoredChunk
from kb.generation.generator import Generator
from kb.retrieval.context import ContextAssembler
from kb.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class RagTrace(BaseModel):
    """Detailed intermediate retrieval and context artifacts for a single ask call."""

    candidates: list[ScoredChunk] = Field(default_factory=list)
    bundle: ContextBundle
    top_score: float | None = None


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

    async def ask_detailed(
        self,
        question: str,
        access: AccessContext,
    ) -> tuple[AnswerResult, RagTrace]:
        """Execute end-to-end RAG flow and return both AnswerResult and RagTrace."""
        candidates = await self.retriever.retrieve(question, access)
        bundle = await self.assembler.assemble(candidates, access)
        trace = RagTrace(
            candidates=candidates,
            bundle=bundle,
            top_score=bundle.top_score,
        )

        model_name = getattr(self.generator.client._llm, "model_name", "unknown")

        # Empty context check
        if not bundle.items:
            logger.info("Pre-check refusal: no eligible context found for query.")
            return (
                AnswerResult(
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
                ),
                trace,
            )

        # Low relevance score check
        if self.refusal_min_score > 0.0:
            if bundle.top_score is None or bundle.top_score < self.refusal_min_score:
                logger.info(
                    "Pre-check refusal: top score %.4f is below threshold %.4f.",
                    bundle.top_score or 0.0,
                    self.refusal_min_score,
                )
                return (
                    AnswerResult(
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
                    ),
                    trace,
                )

        # Generate cited answer via LLM
        result = await self.generator.generate(question, bundle)
        return result, trace

    async def ask(
        self,
        question: str,
        access: AccessContext,
    ) -> AnswerResult:
        """Execute end-to-end RAG answer flow under access control constraints."""
        result, _ = await self.ask_detailed(question, access)
        return result
