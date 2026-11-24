"""Application configuration single source of truth."""

from functools import lru_cache
from uuid import UUID

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # Required settings (fail fast if missing or empty)
    DATABASE_URL: str = Field(default="", description="PostgreSQL database URL")
    REDIS_URL: str = Field(default="", description="Redis connection URL")
    JWT_SECRET: str = Field(default="", description="Secret key for JWT validation")
    EMBEDDING_DIM: int = Field(default=0, description="Dimension of embedding vectors")

    # App settings (optional in this phase)
    APP_ENV: str = Field(default="local", description="Environment: local, test, production")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    DEFAULT_ORG_ID: UUID = Field(
        default=UUID("00000000-0000-0000-0000-000000000001"),
        description="Default organization UUID for single-org MVP",
    )

    # LLM provider settings
    LLM_PROVIDER: str = Field(default="ollama", description="LLM provider name")
    LLM_MODEL: str = Field(default="llama3.2:3b", description="LLM model identifier")
    LLM_API_KEY: str = Field(default="", description="LLM provider API key")

    # Generation & Refusal settings (Phase 2.3)
    REFUSAL_MIN_SCORE: float = Field(
        default=0.0,
        description="Minimum relevance score for answering (0.0 disables refusal gating)",
    )
    ON_UNCITED: str = Field(
        default="retry",
        description="Behavior when answer is uncited: 'retry' or 'refuse'",
    )
    SNIPPET_MAX_CHARS: int = Field(
        default=240,
        description="Maximum characters for resolved citation snippet",
    )

    # Embedding settings (optional in this phase)
    EMBEDDING_PROVIDER: str = Field(default="", description="Embedding provider name")
    EMBEDDING_MODEL: str = Field(default="", description="Embedding model identifier")

    # Reranker settings (optional in this phase)
    RERANKER_PROVIDER: str = Field(default="", description="Reranker provider name")
    RERANKER_MODEL: str = Field(default="", description="Reranker model identifier")
    RERANK_ENABLED: bool = Field(default=True, description="Enable reranker stage")
    RERANK_TOP_N: int = Field(default=8, description="Chunks kept after reranking")

    # Context assembly settings
    CONTEXT_BUDGET_TOKENS: int = Field(
        default=3000, description="Token budget for context assembly"
    )
    CONTEXT_EXPAND_PARENTS: bool = Field(
        default=False, description="Expand child chunks to parents"
    )

    # Ingestion / OCR settings
    OCR_LANG: str = Field(
        default="eng", description="Language code for OCR fallback (default: eng)"
    )
    OCR_CHAR_THRESHOLD: int = Field(
        default=25,
        description="Minimum character count per page before triggering OCR fallback",
    )

    # Auth settings (optional in this phase)
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")

    # Retrieval settings
    RETRIEVAL_VECTOR_K: int = Field(default=50, description="Default vector candidate count")
    RETRIEVAL_FTS_K: int = Field(default=50, description="Default full-text candidate count")
    RETRIEVAL_FUSED_K: int = Field(default=30, description="Default fused top-k count")
    RETRIEVAL_RRF_K: int = Field(default=60, description="Default RRF constant")
    RETRIEVAL_HNSW_EF_SEARCH: int | None = Field(
        default=None, description="HNSW ef_search parameter"
    )

    # Ollama settings
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434", description="Ollama API base URL"
    )

    # Observability settings (optional in this phase)
    LANGFUSE_HOST: str = Field(default="", description="Langfuse host URL")
    LANGFUSE_PUBLIC_KEY: str = Field(default="", description="Langfuse public API key")
    LANGFUSE_SECRET_KEY: str = Field(default="", description="Langfuse secret API key")

    @field_validator("DATABASE_URL", "REDIS_URL", "JWT_SECRET")
    @classmethod
    def validate_required_string(cls, v: str, info: ValidationInfo) -> str:
        """Ensure required string settings are set and not empty."""
        val = v.strip()
        if not val:
            field_name = info.field_name or "Required field"
            raise ValueError(f"{field_name} is required and cannot be empty")
        return val

    @field_validator("EMBEDDING_DIM")
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        """Ensure embedding dimension is set and strictly positive."""
        if v <= 0:
            raise ValueError("EMBEDDING_DIM is required and must be greater than 0")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
