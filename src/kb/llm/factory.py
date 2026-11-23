"""Factory for building embedder instances based on application settings."""

from kb.config import Settings, get_settings
from kb.domain.protocols import EmbedderProtocol
from kb.llm.fakes import FakeEmbedder


class ConfigError(Exception):
    """Exception raised when configuration is invalid or missing required provider."""


def build_embedder(settings: Settings | None = None) -> EmbedderProtocol:
    """Select and instantiate an EmbedderProtocol implementation from configuration.

    'fake' is always available for deterministic local testing.
    'sentence-transformers' uses local models.
    """
    if settings is None:
        settings = get_settings()

    provider = settings.EMBEDDING_PROVIDER.strip().lower()
    if not provider:
        raise ConfigError(
            "EMBEDDING_PROVIDER is unset. Set EMBEDDING_PROVIDER='fake' for testing "
            "or 'sentence-transformers' for local model embeddings."
        )

    if provider == "fake":
        return FakeEmbedder(
            dimension=settings.EMBEDDING_DIM,
            model_name=settings.EMBEDDING_MODEL or "fake-embedder",
        )

    if provider == "sentence-transformers":
        from kb.llm.sentence_transformers import SentenceTransformersEmbedder

        return SentenceTransformersEmbedder(
            model_name=settings.EMBEDDING_MODEL,
            dimension=settings.EMBEDDING_DIM,
        )

    raise ConfigError(
        f"Unsupported EMBEDDING_PROVIDER '{provider}'. "
        "Supported providers: 'fake', 'sentence-transformers'."
    )
