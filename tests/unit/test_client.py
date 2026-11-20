"""Unit tests for the unified LLMClient wrapper."""

import pytest

from kb.llm.client import LLMClient
from kb.llm.fakes import FakeLLM


@pytest.mark.asyncio
async def test_llm_client_delegates_and_records_latency() -> None:
    """LLMClient delegates generation to protocol and records execution latency."""
    fake_llm = FakeLLM(default_response="Delegated response", model_name="stub-model")
    client = LLMClient(fake_llm)

    result = await client.generate("test prompt", system_prompt="system")

    assert result.response.content == "Delegated response"
    assert result.response.model == "stub-model"
    assert result.latency_ms >= 0.0

    # Underlying fake recorded the call
    assert len(fake_llm.history) == 1
    assert fake_llm.history[0]["prompt"] == "test prompt"


@pytest.mark.asyncio
async def test_llm_client_streaming_delegation() -> None:
    """LLMClient delegates streaming to protocol."""
    fake_llm = FakeLLM(default_response="word1 word2 word3")
    client = LLMClient(fake_llm)

    chunks: list[str] = []
    async for chunk in client.stream("stream prompt"):
        chunks.append(chunk)

    assert "".join(chunks) == "word1 word2 word3"
