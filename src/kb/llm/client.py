"""Single LLM call wrapper enforcing Invariant I7."""

import time
from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from kb.domain.protocols import LLMProtocol, LLMResponse


class LLMClientResult(BaseModel):
    """Result of an LLM generation call wrapped with telemetry."""

    model_config = ConfigDict(frozen=True)

    response: LLMResponse = Field(description="Underlying response from the LLM")
    latency_ms: float = Field(ge=0.0, description="Call duration in milliseconds")


class LLMClient:
    """Unified wrapper around LLMProtocol implementations.

    Enforces Invariant I7: All LLM calls pass through this single client,
    exposing hook points for guardrails, tracing, retries, and metrics.
    """

    def __init__(self, llm: LLMProtocol) -> None:
        """Initialize client with a concrete LLMProtocol implementation."""
        self._llm = llm

    async def _before_call(self, prompt: str, system_prompt: str | None) -> None:
        """Hook point for input guardrails, request redaction, and trace creation."""
        # Future phases attach input guardrails and trace initialization here.

    async def _after_call(self, result: LLMResponse, latency_ms: float) -> None:
        """Hook point for output guardrails, metrics recording, and trace completion."""
        # Future phases attach output guardrails, cost logging, and telemetry here.

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMClientResult:
        """Delegate generation to the underlying LLM, measuring duration and invoking hooks."""
        await self._before_call(prompt, system_prompt)

        start_time = time.perf_counter()
        response = await self._llm.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        await self._after_call(response, latency_ms)

        return LLMClientResult(response=response, latency_ms=latency_ms)

    async def stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Delegate streaming to the underlying LLM."""
        await self._before_call(prompt, system_prompt)

        async for chunk in self._llm.stream(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield chunk
