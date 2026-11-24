"""Deterministic fake implementations of domain protocols for tests."""

import hashlib
import json
import math
import re
import struct
from collections.abc import AsyncIterator, Callable
from typing import Any

from kb.domain.protocols import (
    EmbedderProtocol,
    LLMProtocol,
    LLMResponse,
    RerankerProtocol,
    ScoredCandidate,
    TokenUsage,
)


class FakeEmbedder(EmbedderProtocol):
    """Deterministic hash-based dense vector embedder for testing."""

    def __init__(self, dimension: int = 768, model_name: str = "fake-embedder") -> None:
        """Initialize embedder with fixed vector dimension and model identifier."""
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed list of texts into deterministic unit-normalized float vectors."""
        embeddings: list[list[float]] = []
        for text in texts:
            vec: list[float] = []
            for i in range(self._dimension):
                digest = hashlib.sha256(f"{text}::{i}".encode()).digest()
                uint_val = struct.unpack(">I", digest[:4])[0]
                val = (uint_val / 0xFFFFFFFF) * 2.0 - 1.0
                vec.append(val)

            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0.0:
                vec = [v / norm for v in vec]
            embeddings.append(vec)
        return embeddings


class FakeLLM(LLMProtocol):
    """Deterministic stub LLM for testing text generation and streaming."""

    def __init__(
        self,
        default_response: str = "Fake generated response",
        responses: dict[str, str] | list[str] | Callable[[str], str] | None = None,
        model_name: str = "fake-model",
    ) -> None:
        """Initialize stub LLM with predefined responses, scripted sequence, or callable."""
        self.default_response = default_response
        self.responses = responses
        self.model_name = model_name
        self.history: list[dict[str, Any]] = []
        self._call_index: int = 0

    def _next_response(self, prompt: str) -> str:
        """Resolve the next response based on responses configuration."""
        if callable(self.responses):
            return self.responses(prompt)
        if isinstance(self.responses, list):
            if self._call_index < len(self.responses):
                resp = self.responses[self._call_index]
                self._call_index += 1
                return resp
            return self.default_response
        if isinstance(self.responses, dict):
            return self.responses.get(prompt, self.default_response)

        # Smart default for RAG cited generation when responses are not predefined
        if self.default_response == "Fake generated response" and "<chunk id=" in prompt:
            m = re.search(r'<chunk id="([a-zA-Z0-9_\-]+)"', prompt)
            if m:
                cid = m.group(1)
                return json.dumps(
                    {
                        "answer": (
                            "Based on the provided documentation, this claim is verified "
                            f"[[chunk:{cid}]]."
                        ),
                        "citations": [cid],
                        "refusal": False,
                        "refusal_reason": None,
                    }
                )

        return self.default_response

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate response based on scripted responses, callable, exact match, or fallback."""
        content = self._next_response(prompt)
        prompt_tokens = len(prompt.split()) + (len(system_prompt.split()) if system_prompt else 0)
        completion_tokens = len(content.split())
        total_tokens = prompt_tokens + completion_tokens

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        self.history.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response": content,
            }
        )

        return LLMResponse(content=content, usage=usage, model=self.model_name)

    async def stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream response words sequentially."""
        content = self._next_response(prompt)
        self.history.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "streamed": True,
            }
        )
        words = content.split()
        for idx, word in enumerate(words):
            yield word if idx == len(words) - 1 else word + " "


class FakeReranker(RerankerProtocol):
    """Deterministic reranker based on case-insensitive token overlap."""

    async def rerank(self, query: str, candidates: list[str]) -> list[ScoredCandidate]:
        """Rank candidates by token overlap against the query."""
        query_tokens = set(query.lower().split())
        denom = max(len(query_tokens), 1)

        scored: list[ScoredCandidate] = []
        for index, text in enumerate(candidates):
            candidate_tokens = set(text.lower().split())
            overlap = len(query_tokens & candidate_tokens)
            score = float(overlap) / denom
            scored.append(ScoredCandidate(index=index, text=text, score=score))

        # Sort descending by score; break ties by original index
        scored.sort(key=lambda c: (-c.score, c.index))
        return scored
