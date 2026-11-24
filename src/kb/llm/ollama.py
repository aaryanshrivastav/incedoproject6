"""Ollama provider adapter implementing LLMProtocol.

Enforces Invariant I6: Vendor and transport details for Ollama are strictly
isolated to this module in src/kb/llm/.
"""

import asyncio
import json
import logging
import urllib.error
import urllib.request
from collections.abc import AsyncIterator

from kb.domain.protocols import LLMProtocol, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class OllamaLLM(LLMProtocol):
    """Local LLM adapter connecting to an Ollama daemon via HTTP API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
    ) -> None:
        """Initialize Ollama adapter with API endpoint and model tag."""
        self.base_url = base_url.rstrip("/")
        self.model = model

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self.model

    def _sync_post(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        """Execute synchronous HTTP POST request to Ollama endpoint."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)  # type: ignore[no-any-return]
        except urllib.error.URLError as e:
            logger.error("Failed to connect to Ollama at %s: %s", url, e)
            raise RuntimeError(f"Ollama connection error at {self.base_url}: {e}") from e

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate response from Ollama endpoint."""
        url = f"{self.base_url}/api/generate"
        options: dict[str, object] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system_prompt:
            payload["system"] = system_prompt

        data = await asyncio.to_thread(self._sync_post, url, payload)

        content = str(data.get("response", ""))
        p_eval = data.get("prompt_eval_count")
        prompt_tokens = int(p_eval) if isinstance(p_eval, (int, str)) else len(prompt.split())
        c_eval = data.get("eval_count")
        completion_tokens = int(c_eval) if isinstance(c_eval, (int, str)) else len(content.split())
        total_tokens = prompt_tokens + completion_tokens

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return LLMResponse(content=content, usage=usage, model=self.model)

    async def stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream generated text chunks from Ollama endpoint."""
        url = f"{self.base_url}/api/generate"
        options: dict[str, object] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": options,
        }
        if system_prompt:
            payload["system"] = system_prompt

        def _get_stream_lines() -> list[str]:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            lines: list[str] = []
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for line in resp:
                        lines.append(line.decode("utf-8"))
            except urllib.error.URLError as e:
                logger.error("Failed to stream from Ollama at %s: %s", url, e)
                raise RuntimeError(
                    f"Ollama streaming connection error at {self.base_url}: {e}"
                ) from e
            return lines

        lines = await asyncio.to_thread(_get_stream_lines)
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            chunk_data = json.loads(line_str)
            chunk_text = chunk_data.get("response", "")
            if chunk_text:
                yield chunk_text
