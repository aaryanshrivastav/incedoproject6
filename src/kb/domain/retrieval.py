"""Domain models and configuration for hybrid retrieval."""

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
