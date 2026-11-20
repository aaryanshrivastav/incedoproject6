"""Application configuration single source of truth."""

from functools import lru_cache

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

    # LLM provider settings (optional in this phase)
    LLM_PROVIDER: str = Field(default="", description="LLM provider name")
    LLM_MODEL: str = Field(default="", description="LLM model identifier")
    LLM_API_KEY: str = Field(default="", description="LLM provider API key")

    # Embedding settings (optional in this phase)
    EMBEDDING_PROVIDER: str = Field(default="", description="Embedding provider name")
    EMBEDDING_MODEL: str = Field(default="", description="Embedding model identifier")

    # Reranker settings (optional in this phase)
    RERANKER_PROVIDER: str = Field(default="", description="Reranker provider name")
    RERANKER_MODEL: str = Field(default="", description="Reranker model identifier")

    # Auth settings (optional in this phase)
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")

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
