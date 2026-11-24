"""Domain models for generation outputs, citations, and refusal states."""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RefusalReason(StrEnum):
    """Enumeration of possible refusal causes."""

    no_context = "no_context"
    low_relevance = "low_relevance"
    model_declined = "model_declined"
    invalid_model_output = "invalid_model_output"
    uncited_answer = "uncited_answer"


class Citation(BaseModel):
    """Resolved citation mapping an inline claim to verifiable stored chunk provenance."""

    model_config = ConfigDict(frozen=True)

    chunk_id: UUID
    document_id: UUID
    document_title: str
    source_uri: str
    page: int | None = None
    section: str | None = None
    snippet: str
    snippet_truncated: bool


class Usage(BaseModel):
    """Token consumption counts for LLM generation."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class AnswerResult(BaseModel):
    """Complete grounded answer result including citations, refusal status, and telemetry."""

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refusal: bool = False
    refusal_reason: RefusalReason | None = None
    invalid_citations: list[str] = Field(default_factory=list)
    prompt_version: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    context_chunk_ids: list[UUID] = Field(default_factory=list)
