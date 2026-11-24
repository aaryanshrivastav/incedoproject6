"""Generator invoking LLMClient with prompt versioning and citation enforcement."""

import logging
from typing import Any

from kb.domain.answers import AnswerResult, RefusalReason, Usage
from kb.domain.retrieval import ContextBundle
from kb.generation.citations import (
    ModelOutputParseError,
    extract_and_validate_citations,
    parse_model_output,
    resolve_citations,
)
from kb.generation.context_prompt import render_context
from kb.generation.prompts import load_prompt
from kb.llm.client import LLMClient

logger = logging.getLogger(__name__)


class Generator:
    """Orchestrates prompt preparation, LLM generation, validation, and citation resolution."""

    def __init__(
        self,
        client: LLMClient,
        prompt_name: str = "answer",
        prompt_version: str = "1.0.0",
        on_uncited: str = "retry",
        snippet_max_chars: int = 240,
    ) -> None:
        """Initialize generator with LLM client, prompt specs, and validation policies."""
        self.client = client
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version
        self.on_uncited = on_uncited.lower()
        self.snippet_max_chars = snippet_max_chars

        # Pre-load prompt template to fail fast if missing
        self.metadata, self.system_prompt = load_prompt(prompt_name, prompt_version)

    async def generate(
        self,
        question: str,
        bundle: ContextBundle,
    ) -> AnswerResult:
        """Generate cited answer from question and assembled context bundle.

        Args:
            question: User inquiry string.
            bundle: ContextBundle containing retrieved reference chunks.

        Returns:
            AnswerResult model with sanitized text, validated citations, or refusal state.
        """
        valid_chunk_ids = {item.chunk_id for item in bundle.items}
        context_block = render_context(bundle)
        user_prompt = f"{context_block}\n\nUser Question:\n{question}"

        total_prompt_tokens = 0
        total_completion_tokens = 0

        # 1. First LLM generation attempt (Invariant I7: single client entry point)
        llm_res = await self.client.generate(
            user_prompt,
            system_prompt=self.system_prompt,
            temperature=0.0,
        )
        total_prompt_tokens += llm_res.response.usage.prompt_tokens
        total_completion_tokens += llm_res.response.usage.completion_tokens
        model_name = llm_res.response.model

        # 2. Parse model output JSON (with 1 retry on malformed JSON)
        parsed: dict[str, Any] | None = None
        try:
            parsed = parse_model_output(llm_res.response.content)
        except ModelOutputParseError:
            logger.warning("Model JSON parsing failed on initial attempt. Retrying once.")
            retry_prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT: Your previous response was not valid JSON. "
                "You must output ONLY a valid JSON object matching the required schema."
            )
            retry_res = await self.client.generate(
                retry_prompt,
                system_prompt=self.system_prompt,
                temperature=0.0,
            )
            total_prompt_tokens += retry_res.response.usage.prompt_tokens
            total_completion_tokens += retry_res.response.usage.completion_tokens

            try:
                parsed = parse_model_output(retry_res.response.content)
            except ModelOutputParseError:
                logger.error("Model JSON parsing failed on retry. Refusing.")
                return AnswerResult(
                    answer="Failed to generate a valid structured response.",
                    citations=[],
                    refusal=True,
                    refusal_reason=RefusalReason.invalid_model_output,
                    invalid_citations=[],
                    prompt_version=self.prompt_version,
                    model=model_name,
                    usage=Usage(
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                    ),
                    context_chunk_ids=list(valid_chunk_ids),
                )

        assert parsed is not None

        # 3. Check for model-declared refusal
        if parsed.get("refusal") is True:
            refusal_msg = str(parsed.get("answer") or "The model could not answer from context.")
            return AnswerResult(
                answer=refusal_msg,
                citations=[],
                refusal=True,
                refusal_reason=RefusalReason.model_declined,
                invalid_citations=[],
                prompt_version=self.prompt_version,
                model=model_name,
                usage=Usage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                ),
                context_chunk_ids=list(valid_chunk_ids),
            )

        # 4. Extract and validate citations
        raw_answer = str(parsed.get("answer") or "")
        raw_citations = [str(c) for c in (parsed.get("citations") or [])]
        sanitized_answer, valid_ids, invalid_ids = extract_and_validate_citations(
            raw_answer, raw_citations, valid_chunk_ids
        )

        # 5. Handle non-refusal with zero valid citations
        if not valid_ids:
            if self.on_uncited == "retry":
                logger.warning("Non-refusal answer had zero valid citations. Retrying once.")
                uncited_retry_prompt = (
                    f"{user_prompt}\n\n"
                    "CRITICAL: Previous answer contained zero valid citations from <context>.\n"
                    "Every claim must cite [[chunk:<id>]]. If context lacks evidence, "
                    "declare a refusal."
                )
                uncited_res = await self.client.generate(
                    uncited_retry_prompt,
                    system_prompt=self.system_prompt,
                    temperature=0.0,
                )
                total_prompt_tokens += uncited_res.response.usage.prompt_tokens
                total_completion_tokens += uncited_res.response.usage.completion_tokens

                try:
                    retry_parsed = parse_model_output(uncited_res.response.content)
                    if retry_parsed.get("refusal") is True:
                        return AnswerResult(
                            answer=str(retry_parsed.get("answer") or "Refused after retry."),
                            citations=[],
                            refusal=True,
                            refusal_reason=RefusalReason.model_declined,
                            invalid_citations=[],
                            prompt_version=self.prompt_version,
                            model=model_name,
                            usage=Usage(
                                prompt_tokens=total_prompt_tokens,
                                completion_tokens=total_completion_tokens,
                                total_tokens=total_prompt_tokens + total_completion_tokens,
                            ),
                            context_chunk_ids=list(valid_chunk_ids),
                        )

                    retry_raw_answer = str(retry_parsed.get("answer") or "")
                    retry_raw_citations = [str(c) for c in (retry_parsed.get("citations") or [])]
                    sanitized_answer, valid_ids, invalid_ids = extract_and_validate_citations(
                        retry_raw_answer, retry_raw_citations, valid_chunk_ids
                    )
                except ModelOutputParseError:
                    valid_ids = []

            # If still zero valid citations, refuse
            if not valid_ids:
                return AnswerResult(
                    answer="Refused: insufficient citation grounding in reference documents.",
                    citations=[],
                    refusal=True,
                    refusal_reason=RefusalReason.uncited_answer,
                    invalid_citations=invalid_ids,
                    prompt_version=self.prompt_version,
                    model=model_name,
                    usage=Usage(
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                    ),
                    context_chunk_ids=list(valid_chunk_ids),
                )

        # 6. Resolve citations to stored chunk text
        citations = resolve_citations(valid_ids, bundle, max_chars=self.snippet_max_chars)

        return AnswerResult(
            answer=sanitized_answer,
            citations=citations,
            refusal=False,
            refusal_reason=None,
            invalid_citations=invalid_ids,
            prompt_version=self.prompt_version,
            model=model_name,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
            context_chunk_ids=list(valid_chunk_ids),
        )
