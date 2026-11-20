"""Unit tests for deterministic fake implementations."""

import math

import pytest

from kb.domain.protocols import EmbedderProtocol, LLMProtocol, RerankerProtocol
from kb.llm.fakes import FakeEmbedder, FakeLLM, FakeReranker


@pytest.mark.asyncio
async def test_fake_embedder_conforms_to_protocol() -> None:
    """FakeEmbedder satisfies EmbedderProtocol and produces deterministic vectors."""
    embedder = FakeEmbedder(dimension=384)
    assert isinstance(embedder, EmbedderProtocol)
    assert embedder.dimension == 384

    texts = ["hello world", "rag knowledge assistant"]
    vectors1 = await embedder.embed(texts)
    vectors2 = await embedder.embed(texts)

    assert len(vectors1) == 2
    assert len(vectors1[0]) == 384
    assert len(vectors1[1]) == 384

    # Determinism
    assert vectors1 == vectors2

    # Normalization (unit vector)
    norm = math.sqrt(sum(x * x for x in vectors1[0]))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)

    # Different texts yield different vectors
    assert vectors1[0] != vectors1[1]


@pytest.mark.asyncio
async def test_fake_llm_conforms_to_protocol() -> None:
    """FakeLLM satisfies LLMProtocol and generates deterministic output and usage."""
    llm = FakeLLM(
        default_response="Default answer",
        responses={"What is 2+2?": "4"},
        model_name="test-llm",
    )
    assert isinstance(llm, LLMProtocol)

    # Predefined response
    res1 = await llm.generate("What is 2+2?")
    assert res1.content == "4"
    assert res1.model == "test-llm"
    assert res1.usage.prompt_tokens == 3
    assert res1.usage.completion_tokens == 1

    # Default fallback response
    res2 = await llm.generate("Tell me a story", system_prompt="Be concise")
    assert res2.content == "Default answer"
    assert len(llm.history) == 2

    # Streaming
    chunks: list[str] = []
    async for chunk in llm.stream("What is 2+2?"):
        chunks.append(chunk)
    assert "".join(chunks) == "4"


@pytest.mark.asyncio
async def test_fake_reranker_conforms_to_protocol() -> None:
    """FakeReranker satisfies RerankerProtocol and scores by token overlap."""
    reranker = FakeReranker()
    assert isinstance(reranker, RerankerProtocol)

    query = "enterprise rag assistant"
    candidates = [
        "recipe for chocolate cake",
        "enterprise document search and assistant",
        "unrelated text string",
        "enterprise assistant",
    ]

    ranked = await reranker.rerank(query, candidates)

    assert len(ranked) == 4
    # Highest overlap should be candidate index 1 ("enterprise", "assistant")
    assert ranked[0].index == 1
    assert ranked[0].score > 0.0

    # Scores should be in descending order
    scores = [c.score for c in ranked]
    assert scores == sorted(scores, reverse=True)
