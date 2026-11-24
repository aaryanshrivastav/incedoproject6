"""Unit tests for rerank stage, context assembly, and retriever factory."""

import random
from uuid import UUID, uuid4

import pytest

from kb.config import Settings
from kb.domain.protocols import RerankerProtocol, ScoredCandidate
from kb.domain.retrieval import (
    ContextConfig,
    RerankConfig,
    RerankError,
    RetrievedChunk,
    ScoredChunk,
)
from kb.ingestion.tokens import ApproxTokenCounter
from kb.llm.factory import ConfigError, build_reranker
from kb.llm.fakes import FakeReranker
from kb.retrieval.context import ContextAssembler
from kb.retrieval.rerank import RerankStage


def _make_retrieved_chunk(
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
    title: str = "Doc 1",
    text: str = "text",
    ordinal: int = 0,
    content_hash: str | None = None,
    fused_score: float = 0.5,
    vector_sim: float | None = 0.8,
    parent_chunk_id: UUID | None = None,
) -> RetrievedChunk:
    cid = chunk_id or uuid4()
    did = document_id or uuid4()
    chash = content_hash or f"hash-{cid}"
    return RetrievedChunk(
        chunk_id=cid,
        document_id=did,
        document_title=title,
        source_uri=f"file:///test/{did}.txt",
        text=text,
        ordinal=ordinal,
        content_hash=chash,
        parent_chunk_id=parent_chunk_id,
        fused_score=fused_score,
        vector_similarity=vector_sim,
    )


class FailingReranker(RerankerProtocol):
    """Stub reranker that raises an exception during rerank."""

    async def rerank(self, query: str, candidates: list[str]) -> list[ScoredCandidate]:
        raise RuntimeError("Reranker model inference failure")


@pytest.mark.asyncio
async def test_reranker_reorders_candidates() -> None:
    """Verify reranker reorders candidates based on query token overlap."""
    fake_reranker = FakeReranker()
    stage = RerankStage(fake_reranker, RerankConfig(enabled=True, top_n=2))

    # Candidate 1: low keyword match
    c1 = _make_retrieved_chunk(text="unrelated document notes", fused_score=0.9)
    # Candidate 2: high keyword match to query
    c2 = _make_retrieved_chunk(text="quantum computing algorithms", fused_score=0.3)
    # Candidate 3: moderate match
    c3 = _make_retrieved_chunk(text="general computing overview", fused_score=0.2)

    query = "quantum computing"
    results = await stage.rerank(query, [c1, c2, c3])

    assert len(results) == 2
    # c2 matches 2 tokens (score 1.0) -> rank 1
    assert results[0].chunk_id == c2.chunk_id
    assert results[0].final_score == pytest.approx(1.0)
    assert results[0].rerank_score == pytest.approx(1.0)

    # c3 matches 1 token (score 0.5) -> rank 2
    assert results[1].chunk_id == c3.chunk_id
    assert results[1].final_score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_reranker_ablation_disabled() -> None:
    """Verify ablation mode (enabled=False) preserves fused order and sets final_score."""
    fake_reranker = FakeReranker()
    stage = RerankStage(fake_reranker, RerankConfig(enabled=False, top_n=2))

    c1 = _make_retrieved_chunk(text="high fused score", fused_score=0.8)
    c2 = _make_retrieved_chunk(text="medium fused score", fused_score=0.6)
    c3 = _make_retrieved_chunk(text="low fused score", fused_score=0.2)

    results = await stage.rerank("quantum", [c1, c2, c3])

    assert len(results) == 2
    assert results[0].chunk_id == c1.chunk_id
    assert results[0].rerank_score is None
    assert results[0].final_score == 0.8

    assert results[1].chunk_id == c2.chunk_id
    assert results[1].rerank_score is None
    assert results[1].final_score == 0.6


@pytest.mark.asyncio
async def test_rerank_failure_raises_rerank_error() -> None:
    """Verify that failure in the reranker model raises RerankError and is not swallowed."""
    stage = RerankStage(FailingReranker(), RerankConfig(enabled=True))
    c1 = _make_retrieved_chunk(text="text sample")

    with pytest.raises(RerankError) as exc_info:
        await stage.rerank("test query", [c1])

    assert "Reranker failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_context_deduplication() -> None:
    """Verify deduplication keeps the highest-scoring chunk and records duplicates in dropped."""
    assembler = ContextAssembler(config=ContextConfig(budget_tokens=1000))

    cid_high = uuid4()
    cid_low = uuid4()
    shared_hash = "shared-content-hash-123"

    chunk_high = ScoredChunk(
        **_make_retrieved_chunk(
            chunk_id=cid_high,
            text="Duplicate content block",
            content_hash=shared_hash,
            fused_score=0.9,
        ).model_dump(),
        rerank_score=0.9,
        final_score=0.9,
    )
    chunk_low = ScoredChunk(
        **_make_retrieved_chunk(
            chunk_id=cid_low,
            text="Duplicate content block",
            content_hash=shared_hash,
            fused_score=0.4,
        ).model_dump(),
        rerank_score=0.4,
        final_score=0.4,
    )

    bundle = await assembler.assemble([chunk_low, chunk_high])

    assert len(bundle.items) == 1
    assert bundle.items[0].chunk_id == cid_high
    assert bundle.items[0].final_score == 0.9

    assert len(bundle.dropped) == 1
    assert bundle.dropped[0]["chunk_id"] == str(cid_low)
    assert bundle.dropped[0]["reason"] == "duplicate"


@pytest.mark.asyncio
async def test_context_budget_greedy_fill_no_truncation() -> None:
    """Verify budget fitting skips an oversize chunk, packs smaller chunks, and never truncates."""
    counter = ApproxTokenCounter()
    # ApproxTokenCounter counts ceil(len(text) / 4)
    # text with 20 chars = 5 tokens
    # text with 400 chars = 100 tokens
    # text with 12 chars = 3 tokens

    # Set budget to 20 tokens
    assembler = ContextAssembler(counter=counter, config=ContextConfig(budget_tokens=20))

    chunk_1 = ScoredChunk(
        **_make_retrieved_chunk(
            text="A" * 40,  # 10 tokens
            fused_score=0.9,
        ).model_dump(),
        rerank_score=0.9,
        final_score=0.9,
    )
    # Oversize chunk: 100 tokens (will exceed remaining 10 budget)
    chunk_oversize = ScoredChunk(
        **_make_retrieved_chunk(
            text="B" * 400,  # 100 tokens
            fused_score=0.8,
        ).model_dump(),
        rerank_score=0.8,
        final_score=0.8,
    )
    # Smaller chunk that fits in remaining 10 budget: 20 chars = 5 tokens
    chunk_small = ScoredChunk(
        **_make_retrieved_chunk(
            text="C" * 20,  # 5 tokens
            fused_score=0.7,
        ).model_dump(),
        rerank_score=0.7,
        final_score=0.7,
    )

    bundle = await assembler.assemble([chunk_1, chunk_oversize, chunk_small])

    # Should contain chunk_1 and chunk_small, skipping chunk_oversize
    assert len(bundle.items) == 2
    included_texts = [item.text for item in bundle.items]
    assert "A" * 40 in included_texts
    assert "C" * 20 in included_texts
    assert "B" * 400 not in included_texts

    # Total tokens = 10 + 5 = 15 <= 20
    assert bundle.total_tokens == 15
    assert bundle.total_tokens <= 20

    # Never truncate text
    assert len(bundle.items[0].text) in (40, 20)
    assert len(bundle.items[1].text) in (40, 20)

    # Oversize recorded in dropped
    assert any(
        d["chunk_id"] == str(chunk_oversize.chunk_id) and d["reason"] == "budget"
        for d in bundle.dropped
    )


@pytest.mark.asyncio
async def test_context_deterministic_ordering_shuffled() -> None:
    """Verify document-grouped ordering is deterministic across repeated runs and shuffles."""
    assembler = ContextAssembler(config=ContextConfig(budget_tokens=5000))

    doc_a = uuid4()
    doc_b = uuid4()

    # Doc A has chunks with max score 0.8, ordinals 0 and 1
    c_a0 = ScoredChunk(
        **_make_retrieved_chunk(
            document_id=doc_a, title="Doc A", ordinal=0, fused_score=0.5
        ).model_dump(),
        rerank_score=0.5,
        final_score=0.5,
    )
    c_a1 = ScoredChunk(
        **_make_retrieved_chunk(
            document_id=doc_a, title="Doc A", ordinal=1, fused_score=0.8
        ).model_dump(),
        rerank_score=0.8,
        final_score=0.8,
    )

    # Doc B has chunks with max score 0.9 (better than Doc A), ordinals 0 and 1
    c_b0 = ScoredChunk(
        **_make_retrieved_chunk(
            document_id=doc_b, title="Doc B", ordinal=0, fused_score=0.9
        ).model_dump(),
        rerank_score=0.9,
        final_score=0.9,
    )
    c_b1 = ScoredChunk(
        **_make_retrieved_chunk(
            document_id=doc_b, title="Doc B", ordinal=1, fused_score=0.2
        ).model_dump(),
        rerank_score=0.2,
        final_score=0.2,
    )

    initial_list = [c_a0, c_a1, c_b0, c_b1]

    # Run multiple times with random shuffling
    reference_bundle = await assembler.assemble(initial_list)
    ref_ids = [item.chunk_id for item in reference_bundle.items]

    # Doc B has higher best score (0.9 vs 0.8), so Doc B chunks come first, ordered by ordinal
    assert reference_bundle.items[0].chunk_id == c_b0.chunk_id  # Doc B ord 0
    assert reference_bundle.items[1].chunk_id == c_b1.chunk_id  # Doc B ord 1
    assert reference_bundle.items[2].chunk_id == c_a0.chunk_id  # Doc A ord 0
    assert reference_bundle.items[3].chunk_id == c_a1.chunk_id  # Doc A ord 1

    for seed in range(10):
        shuffled = list(initial_list)
        random.Random(seed).shuffle(shuffled)
        bundle = await assembler.assemble(shuffled)
        assert [item.chunk_id for item in bundle.items] == ref_ids


@pytest.mark.asyncio
async def test_top_score_calculation() -> None:
    """Verify top_score calculation for reranked, unreranked, and empty states."""
    assembler = ContextAssembler()

    # 1. Empty candidates
    empty_bundle = await assembler.assemble([])
    assert empty_bundle.top_score is None
    assert not empty_bundle.reranked

    # 2. Reranked candidates -> top_score is highest rerank_score
    r_chunk_1 = ScoredChunk(
        **_make_retrieved_chunk(vector_sim=0.9).model_dump(),
        rerank_score=0.75,
        final_score=0.75,
    )
    r_chunk_2 = ScoredChunk(
        **_make_retrieved_chunk(vector_sim=0.95).model_dump(),
        rerank_score=0.85,
        final_score=0.85,
    )
    r_bundle = await assembler.assemble([r_chunk_1, r_chunk_2])
    assert r_bundle.reranked is True
    assert r_bundle.top_score == 0.85

    # 3. Unreranked candidates -> top_score is highest vector_similarity
    un_chunk_1 = ScoredChunk(
        **_make_retrieved_chunk(vector_sim=0.72).model_dump(),
        rerank_score=None,
        final_score=0.5,
    )
    un_chunk_2 = ScoredChunk(
        **_make_retrieved_chunk(vector_sim=0.88).model_dump(),
        rerank_score=None,
        final_score=0.3,
    )
    un_bundle = await assembler.assemble([un_chunk_1, un_chunk_2])
    assert un_bundle.reranked is False
    assert un_bundle.top_score == 0.88


@pytest.mark.asyncio
async def test_parent_expansion_disabled_leaves_untouched() -> None:
    """Verify that when expand_parents=False, chunks with parent_chunk_id are not touched."""
    assembler = ContextAssembler(config=ContextConfig(expand_parents=False))
    parent_id = uuid4()
    child = ScoredChunk(
        **_make_retrieved_chunk(
            text="child leaf text",
            parent_chunk_id=parent_id,
        ).model_dump(),
        rerank_score=0.8,
        final_score=0.8,
    )
    bundle = await assembler.assemble([child])
    assert len(bundle.items) == 1
    assert bundle.items[0].text == "child leaf text"
    assert bundle.items[0].chunk_id == child.chunk_id


def test_build_reranker_factory() -> None:
    """Verify build_reranker factory returns fake or raises ConfigError."""
    # 1. fake provider
    settings_fake = Settings(
        DATABASE_URL="postgresql+psycopg://postgres:secret@localhost:5432/postgres",
        REDIS_URL="redis://localhost:6379/0",
        JWT_SECRET="secret",
        EMBEDDING_DIM=768,
        RERANKER_PROVIDER="fake",
    )
    reranker = build_reranker(settings_fake)
    assert isinstance(reranker, FakeReranker)

    # 2. unset provider raises ConfigError
    settings_unset = Settings(
        DATABASE_URL="postgresql+psycopg://postgres:secret@localhost:5432/postgres",
        REDIS_URL="redis://localhost:6379/0",
        JWT_SECRET="secret",
        EMBEDDING_DIM=768,
        RERANKER_PROVIDER="",
    )
    with pytest.raises(ConfigError) as exc_info:
        build_reranker(settings_unset)
    assert "RERANKER_PROVIDER is unset" in str(exc_info.value)

    # 3. unsupported provider raises ConfigError
    settings_unsupported = Settings(
        DATABASE_URL="postgresql+psycopg://postgres:secret@localhost:5432/postgres",
        REDIS_URL="redis://localhost:6379/0",
        JWT_SECRET="secret",
        EMBEDDING_DIM=768,
        RERANKER_PROVIDER="unsupported-provider",
    )
    with pytest.raises(ConfigError) as exc_info_unsupported:
        build_reranker(settings_unsupported)
    assert "Unsupported RERANKER_PROVIDER" in str(exc_info_unsupported.value)
