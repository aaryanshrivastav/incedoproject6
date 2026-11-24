"""Unit tests for RagService pre-checks and orchestration."""

from uuid import uuid4

import pytest

from kb.domain.answers import RefusalReason
from kb.domain.retrieval import AccessContext, ContextBundle, ContextItem, ScoredChunk
from kb.generation.generator import Generator
from kb.llm.client import LLMClient
from kb.llm.fakes import FakeLLM
from kb.rag.service import RagService


class StubRetriever:
    """Stub retriever returning predefined ScoredChunks."""

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self.chunks = chunks

    async def retrieve(self, query: str, access: AccessContext) -> list[ScoredChunk]:
        return self.chunks


class StubContextAssembler:
    """Stub assembler returning a predefined ContextBundle."""

    def __init__(self, bundle: ContextBundle) -> None:
        self.bundle = bundle

    async def assemble(
        self, chunks: list[ScoredChunk], access: AccessContext | None = None
    ) -> ContextBundle:
        return self.bundle


@pytest.mark.asyncio
async def test_empty_context_refuses_without_calling_llm() -> None:
    """Verify empty eligible context triggers no_context refusal without calling LLM."""
    fake_llm = FakeLLM()
    client = LLMClient(fake_llm)
    generator = Generator(client)

    empty_bundle = ContextBundle(
        items=[], total_tokens=0, dropped=[], top_score=None, reranked=True
    )
    service = RagService(
        retriever=StubRetriever([]),  # type: ignore[arg-type]
        assembler=StubContextAssembler(empty_bundle),  # type: ignore[arg-type]
        generator=generator,
    )

    access = AccessContext(org_id=uuid4(), hub_id=uuid4(), user_tags=frozenset())
    result = await service.ask("What is the quarterly revenue?", access)

    # 1. LLM must NOT be called
    assert len(fake_llm.history) == 0
    # 2. Refusal must be declared with reason no_context
    assert result.refusal is True
    assert result.refusal_reason == RefusalReason.no_context
    assert result.citations == []


@pytest.mark.asyncio
async def test_low_relevance_refuses_without_calling_llm() -> None:
    """Verify score below REFUSAL_MIN_SCORE triggers low_relevance without calling LLM."""
    fake_llm = FakeLLM()
    client = LLMClient(fake_llm)
    generator = Generator(client)

    low_score_item = ContextItem(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_title="Peripheral Doc",
        source_uri="file:///doc.txt",
        ordinal=0,
        text="Peripheral unrelated notes.",
        content_hash="h1",
        final_score=0.35,
    )
    bundle = ContextBundle(
        items=[low_score_item],
        total_tokens=10,
        dropped=[],
        top_score=0.35,
        reranked=True,
    )

    # Threshold set to 0.70 (> 0.35)
    service = RagService(
        retriever=StubRetriever([]),  # type: ignore[arg-type]
        assembler=StubContextAssembler(bundle),  # type: ignore[arg-type]
        generator=generator,
        refusal_min_score=0.70,
    )

    access = AccessContext(org_id=uuid4(), hub_id=uuid4(), user_tags=frozenset())
    result = await service.ask("Specific query?", access)

    # 1. LLM must NOT be called
    assert len(fake_llm.history) == 0
    # 2. Refusal must be declared with reason low_relevance
    assert result.refusal is True
    assert result.refusal_reason == RefusalReason.low_relevance
    assert result.citations == []
