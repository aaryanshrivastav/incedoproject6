"""JudgeMetric Protocol and native LLM-as-a-judge evaluation metrics."""

import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from kb.generation.citations import ModelOutputParseError, parse_model_output
from kb.generation.prompts import load_prompt
from kb.llm.client import LLMClient

logger = logging.getLogger(__name__)

JUDGE_METRIC_NAMES = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
)


def _escape_delimiters(text: str) -> str:
    """Neutralize closing XML-style delimiter tags in untrusted data (Invariant I3)."""
    return (
        text.replace("</context>", "&lt;/context&gt;")
        .replace("</answer>", "&lt;/answer&gt;")
        .replace("</reference_answer>", "&lt;/reference_answer&gt;")
        .replace("</chunks>", "&lt;/chunks&gt;")
        .replace("</question>", "&lt;/question&gt;")
    )


class JudgeMetricScore(BaseModel):
    """Result of evaluating one metric on one item."""

    score: float | None = None
    reason: str = ""
    failed: bool = False


class ItemJudgeResults(BaseModel):
    """All judge metric results for a single evaluation item."""

    faithfulness: JudgeMetricScore | None = None
    answer_relevance: JudgeMetricScore | None = None
    context_precision: JudgeMetricScore | None = None
    context_recall: JudgeMetricScore | None = None


@runtime_checkable
class JudgeMetric(Protocol):
    """Protocol for judge-based generation and context evaluation."""

    @property
    def prompt_versions(self) -> dict[str, str]:
        """Return mapping of judge prompt names to version strings."""
        ...

    async def evaluate_item(
        self,
        *,
        question: str,
        reference_answer: str,
        answer: str,
        context_chunks: list[str],
        is_answerable: bool,
        was_refused: bool,
    ) -> ItemJudgeResults:
        """Score applicable judge metrics for one item."""
        ...


class NativeJudgeEvaluator:
    """Native LLM-as-a-judge evaluator routed through LLMClient (I7) and file prompts (I9)."""

    def __init__(self, client: LLMClient, prompt_version: str = "1.0.0") -> None:
        self.client = client
        self.prompt_version = prompt_version
        self._prompts: dict[str, str] = {}
        self._versions: dict[str, str] = {}
        for name in (
            "judge_faithfulness",
            "judge_answer_relevance",
            "judge_context_relevance",
            "judge_context_recall",
        ):
            meta, text = load_prompt(name, prompt_version)
            self._prompts[name] = text
            self._versions[name] = meta.version

    @property
    def prompt_versions(self) -> dict[str, str]:
        """Return mapping of judge prompt names to version strings."""
        return dict(self._versions)

    async def _call_json_with_retry(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Call judge LLM via LLMClient with one retry on JSON parse failure."""
        res = await self.client.generate(user_prompt, system_prompt=system_prompt, temperature=0.0)
        try:
            return parse_model_output(res.response.content)
        except ModelOutputParseError:
            retry_prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT: Your previous output was not valid JSON. "
                "Output ONLY a valid JSON object matching the schema."
            )
            retry_res = await self.client.generate(
                retry_prompt, system_prompt=system_prompt, temperature=0.0
            )
            return parse_model_output(retry_res.response.content)

    async def score_faithfulness(self, answer: str, context_chunks: list[str]) -> JudgeMetricScore:
        """Compute faithfulness = supported_claims / total_claims (None if no claims)."""
        ctx = "\n\n".join(_escape_delimiters(c) for c in context_chunks)
        user_prompt = (
            f"<context>\n{ctx}\n</context>\n\n<answer>\n{_escape_delimiters(answer)}\n</answer>"
        )
        try:
            data = await self._call_json_with_retry(
                self._prompts["judge_faithfulness"], user_prompt
            )
            claims = data.get("claims")
            if not isinstance(claims, list):
                raise ValueError("Missing 'claims' list")
            if len(claims) == 0:
                return JudgeMetricScore(score=None, reason=str(data.get("reason", "")))
            supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported") is True)
            return JudgeMetricScore(
                score=supported / len(claims),
                reason=str(data.get("reason", "")),
            )
        except Exception as exc:
            logger.warning("Judge faithfulness failed: %s", exc)
            return JudgeMetricScore(score=None, reason=f"judge_failure: {exc}", failed=True)

    async def score_answer_relevance(self, question: str, answer: str) -> JudgeMetricScore:
        """Compute normalized answer relevance = (score - 1) / 4 for score in [1, 5]."""
        user_prompt = (
            f"<question>\n{_escape_delimiters(question)}\n</question>\n\n"
            f"<answer>\n{_escape_delimiters(answer)}\n</answer>"
        )
        try:
            data = await self._call_json_with_retry(
                self._prompts["judge_answer_relevance"], user_prompt
            )
            raw_score = float(data["score"])
            if not (1.0 <= raw_score <= 5.0):
                raise ValueError(f"Score {raw_score} out of range [1, 5]")
            return JudgeMetricScore(
                score=(raw_score - 1.0) / 4.0,
                reason=str(data.get("reason", "")),
            )
        except Exception as exc:
            logger.warning("Judge answer_relevance failed: %s", exc)
            return JudgeMetricScore(score=None, reason=f"judge_failure: {exc}", failed=True)

    async def score_context_precision(
        self, question: str, reference_answer: str, context_chunks: list[str]
    ) -> JudgeMetricScore:
        """Compute average precision over ranked context chunks (0.0 if none relevant)."""
        if not context_chunks:
            return JudgeMetricScore(score=0.0, reason="No context chunks retrieved")
        rendered_chunks = "\n".join(
            f'<chunk rank="{idx}">{_escape_delimiters(c)}</chunk>'
            for idx, c in enumerate(context_chunks, start=1)
        )
        user_prompt = (
            f"<question>\n{_escape_delimiters(question)}\n</question>\n\n"
            f"<reference_answer>\n{_escape_delimiters(reference_answer)}\n</reference_answer>\n\n"
            f"<chunks>\n{rendered_chunks}\n</chunks>"
        )
        try:
            data = await self._call_json_with_retry(
                self._prompts["judge_context_relevance"], user_prompt
            )
            chunk_evals = data.get("chunks")
            if not isinstance(chunk_evals, list):
                raise ValueError("Missing 'chunks' list")
            rel_by_rank: dict[int, bool] = {}
            for entry in chunk_evals:
                if isinstance(entry, dict) and "rank" in entry:
                    rel_by_rank[int(entry["rank"])] = bool(entry.get("relevant"))
            total_rel = 0
            precision_sum = 0.0
            for rank in range(1, len(context_chunks) + 1):
                if rel_by_rank.get(rank, False):
                    total_rel += 1
                    precision_sum += total_rel / rank
            ap = (precision_sum / total_rel) if total_rel > 0 else 0.0
            return JudgeMetricScore(score=ap, reason=str(data.get("reason", "")))
        except Exception as exc:
            logger.warning("Judge context_precision failed: %s", exc)
            return JudgeMetricScore(score=None, reason=f"judge_failure: {exc}", failed=True)

    async def score_context_recall(
        self, reference_answer: str, context_chunks: list[str]
    ) -> JudgeMetricScore:
        """Compute context_recall = supported_statements / total_statements."""
        if not context_chunks:
            return JudgeMetricScore(score=0.0, reason="No context chunks retrieved")
        ctx = "\n\n".join(_escape_delimiters(c) for c in context_chunks)
        user_prompt = (
            f"<reference_answer>\n{_escape_delimiters(reference_answer)}\n</reference_answer>\n\n"
            f"<context>\n{ctx}\n</context>"
        )
        try:
            data = await self._call_json_with_retry(
                self._prompts["judge_context_recall"], user_prompt
            )
            stmts = data.get("statements")
            if not isinstance(stmts, list):
                raise ValueError("Missing 'statements' list")
            if len(stmts) == 0:
                return JudgeMetricScore(score=0.0, reason=str(data.get("reason", "")))
            supported = sum(1 for s in stmts if isinstance(s, dict) and s.get("supported") is True)
            return JudgeMetricScore(
                score=supported / len(stmts),
                reason=str(data.get("reason", "")),
            )
        except Exception as exc:
            logger.warning("Judge context_recall failed: %s", exc)
            return JudgeMetricScore(score=None, reason=f"judge_failure: {exc}", failed=True)

    async def evaluate_item(
        self,
        *,
        question: str,
        reference_answer: str,
        answer: str,
        context_chunks: list[str],
        is_answerable: bool,
        was_refused: bool,
    ) -> ItemJudgeResults:
        """Evaluate applicable judge metrics per Phase 3.2 applicability rules."""
        if not is_answerable:
            return ItemJudgeResults()

        ctx_prec = await self.score_context_precision(question, reference_answer, context_chunks)
        ctx_rec = await self.score_context_recall(reference_answer, context_chunks)

        faith: JudgeMetricScore | None = None
        ans_rel: JudgeMetricScore | None = None
        if not was_refused:
            faith = await self.score_faithfulness(answer, context_chunks)
            ans_rel = await self.score_answer_relevance(question, answer)

        return ItemJudgeResults(
            faithfulness=faith,
            answer_relevance=ans_rel,
            context_precision=ctx_prec,
            context_recall=ctx_rec,
        )
