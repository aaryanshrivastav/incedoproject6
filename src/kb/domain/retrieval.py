"""Domain models and configuration for hybrid retrieval."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AccessContext(BaseModel):
    """Verified authorization context for access-controlled retrieval (Invariant I1)."""

    model_config = ConfigDict(frozen=True)

    org_id: UUID = Field(..., description="Organization UUID from auth token")
    hub_id: UUID = Field(..., description="Hub UUID from verified route/token")
    user_tags: frozenset[str] = Field(
        ..., description="User access control tags from auth token (empty frozenset for untagged)"
    )


class RetrievalConfig(BaseModel):
    """Configuration parameters for hybrid search and fusion."""

    model_config = ConfigDict(frozen=True)

    vector_k: int = Field(default=50, ge=1, description="Number of vector candidates to fetch")
    fts_k: int = Field(default=50, ge=1, description="Number of full-text candidates to fetch")
    fused_k: int = Field(default=30, ge=1, description="Top-k candidates after RRF fusion")
    rrf_k: int = Field(default=60, ge=1, description="RRF rank smoothing constant")
    hnsw_ef_search: int | None = Field(
        default=None,
        description="HNSW ef_search parameter (when None, max(40, vector_k) is used)",
    )

    @property
    def effective_ef_search(self) -> int:
        """Return configured or dynamic ef_search."""
        if self.hnsw_ef_search is not None:
            return self.hnsw_ef_search
        return max(40, self.vector_k)


class RetrievedChunk(BaseModel):
    """Single candidate chunk returned by hybrid retrieval with scores and ranks."""

    model_config = ConfigDict(frozen=True)

    chunk_id: UUID
    document_id: UUID
    document_title: str
    source_uri: str
    text: str
    page: int | None = None
    section: str | None = None
    ordinal: int
    content_hash: str
    parent_chunk_id: UUID | None = Field(
        default=None,
        description="ID of parent chunk when hierarchical chunking is used",
    )
    vector_similarity: float | None = Field(
        default=None,
        description="1 - cosine_distance (None if chunk originated only from full-text)",
    )
    fts_rank_score: float | None = Field(
        default=None,
        description="ts_rank_cd score (None if chunk originated only from vector search)",
    )
    vector_rank: int | None = Field(
        default=None,
        description="1-based rank in vector results list",
    )
    fts_rank: int | None = Field(
        default=None,
        description="1-based rank in full-text results list",
    )
    fused_score: float = Field(
        ...,
        description="Reciprocal rank fusion score",
    )


class RetrievalError(Exception):
    """Base exception for retrieval failures."""


class RerankError(RetrievalError):
    """Raised when the reranker call or processing fails."""


class ScoredChunk(RetrievedChunk):
    """Candidate chunk after reranking stage with rerank_score and final_score."""

    rerank_score: float | None = Field(
        default=None,
        description="Relevance score assigned by reranker (None when reranking is disabled)",
    )
    final_score: float = Field(
        ...,
        description="Final relevance score (rerank_score when reranking ran, else fused_score)",
    )


class RerankConfig(BaseModel):
    """Configuration for candidate reranking."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Whether to execute reranker stage")
    top_n: int = Field(
        default=8, ge=1, description="Number of candidate chunks kept after reranking"
    )


class ContextConfig(BaseModel):
    """Configuration for context assembly."""

    model_config = ConfigDict(frozen=True)

    budget_tokens: int = Field(
        default=3000, ge=1, description="Maximum total tokens allowed in assembled context"
    )
    expand_parents: bool = Field(
        default=False, description="Whether to expand child chunks to parent chunks"
    )


class ContextItem(BaseModel):
    """Single deduplicated and budgeted chunk item included in the assembled context."""

    model_config = ConfigDict(frozen=True)

    chunk_id: UUID
    document_id: UUID
    document_title: str
    source_uri: str
    page: int | None = None
    section: str | None = None
    ordinal: int
    text: str
    content_hash: str
    final_score: float


class ContextBundle(BaseModel):
    """Assembled context bundle ready for prompt injection and generation."""

    model_config = ConfigDict(frozen=True)

    items: list[ContextItem] = Field(
        default_factory=list, description="Ordered context items fitting within token budget"
    )
    total_tokens: int = Field(
        default=0, ge=0, description="Sum of tokens across all included items"
    )
    dropped: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Records of chunks dropped during assembly (chunk_id, reason)",
    )
    top_score: float | None = Field(
        default=None,
        description="Best relevance score: best rerank_score if reranked, else vector_similarity",
    )
    reranked: bool = Field(
        default=False, description="Whether reranking was executed on candidate chunks"
    )
