"""Factory for building embedder instances based on application settings."""

from kb.config import Settings, get_settings
from kb.domain.protocols import EmbedderProtocol, LLMProtocol, RerankerProtocol
from kb.llm.fakes import FakeEmbedder, FakeLLM, FakeReranker
from kb.llm.ollama import OllamaLLM


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


def build_reranker(settings: Settings | None = None) -> RerankerProtocol:
    """Select and instantiate a RerankerProtocol implementation from configuration.

    'fake' is always available for deterministic local testing.
    """
    if settings is None:
        settings = get_settings()

    provider = settings.RERANKER_PROVIDER.strip().lower()
    if not provider:
        raise ConfigError("RERANKER_PROVIDER is unset. Set RERANKER_PROVIDER='fake' for testing.")

    if provider == "fake":
        return FakeReranker()

    raise ConfigError(f"Unsupported RERANKER_PROVIDER '{provider}'. Supported providers: 'fake'.")


def build_llm(settings: Settings | None = None) -> LLMProtocol:
    """Select and instantiate an LLMProtocol implementation from configuration.

    'fake' is always available for deterministic local testing.
    'ollama' connects to local Ollama daemon using configured model.
    """
    if settings is None:
        settings = get_settings()

    provider = settings.LLM_PROVIDER.strip().lower()
    if not provider:
        raise ConfigError(
            "LLM_PROVIDER is unset. Set LLM_PROVIDER='fake' for testing "
            "or 'ollama' for local inference."
        )

    if provider == "fake":
        return FakeLLM(model_name=settings.LLM_MODEL or "fake-model")

    if provider == "ollama":
        return OllamaLLM(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL or "llama3.2:3b",
        )

    raise ConfigError(
        f"Unsupported LLM_PROVIDER '{provider}'. Supported providers: 'fake', 'ollama'."
    )
