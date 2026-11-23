"""Domain protocols and pydantic models for LLM, Embedder, and Reranker interfaces."""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    """Token usage details for LLM execution."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0, description="Tokens in the prompt")
    completion_tokens: int = Field(default=0, ge=0, description="Tokens in the completion")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed")


class LLMResponse(BaseModel):
    """Standardized response from an LLM generation call."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(description="Generated text content")
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Token usage details")
    model: str = Field(default="", description="Model identifier used for generation")


class ScoredCandidate(BaseModel):
    """Ranked and scored text candidate returned by a reranker."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="Original index of candidate in input list")
    text: str = Field(description="Candidate text content")
    score: float = Field(description="Relevance score assigned by reranker")


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for language model text generation and streaming."""

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a complete text response from the given prompt."""
        ...

    def stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream chunks of generated text from the given prompt."""
        ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol for generating dense vector embeddings."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings into float vectors."""
        ...

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality produced by this embedder."""
        ...

    @property
    def model_name(self) -> str:
        """Return the name or identifier of the underlying embedding model."""
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol for reranking candidate texts against a query."""

    async def rerank(self, query: str, candidates: list[str]) -> list[ScoredCandidate]:
        """Rerank candidates based on relevance to query, returned in sorted order."""
        ...
