"""Local embedding adapter using sentence-transformers (vendor-isolated)."""

import asyncio

from kb.domain.protocols import EmbedderProtocol


class SentenceTransformersEmbedder(EmbedderProtocol):
    """Local embedding generator using sentence-transformers models."""

    def __init__(self, model_name: str, dimension: int = 768) -> None:
        """Initialize sentence-transformers model."""
        # Lazy import of vendor SDK strictly within src/kb/llm/ (Invariant I6)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers package is required for local embeddings. "
                "Install it via `uv add sentence-transformers`."
            ) from e

        self._model_name = model_name
        self._dimension = dimension
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings asynchronously in thread pool."""
        if not texts:
            return []

        def _encode() -> list[list[float]]:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [[float(x) for x in vec] for vec in embeddings]

        return await asyncio.to_thread(_encode)
