"""Evaluation runner orchestrating preflight, ask_detailed, oracle checks, metrics, and reports."""

import asyncio
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from kb.config import Settings
from kb.db.repositories import HubRepository
from kb.domain.evaluation import GoldenItem, GoldenType
from kb.domain.retrieval import AccessContext, ContextConfig, RerankConfig, RetrievalConfig
from kb.evaluation.access_oracle import leaks as oracle_leaks
from kb.evaluation.corpus_plan import CorpusPlan, check_corpus_against_golden
from kb.evaluation.golden import golden_file_hash, load_golden, span_in_text
from kb.evaluation.metrics.behavior import BehaviorObservation, compute_behavior_metrics
from kb.evaluation.metrics.judge import ItemJudgeResults, JudgeMetric
from kb.evaluation.metrics.retrieval import compute_item_retrieval_metrics
from kb.evaluation.resolve import resolve_item
from kb.rag.service import RagService


class MetricSummary(BaseModel):
    """Aggregate summary statistics for a single metric."""

    mean: float | None = None
    n: int = 0
    std_error: float | None = None
    judge_failures: int = 0
    invalid: bool = False
    skipped: bool = False


class ItemEvalRecord(BaseModel):
    """Detailed per-item evaluation trace and scores."""

    id: str
    type: GoldenType
    question: str
    expected_refusal: bool
    refused: bool
    refusal_reason: str | None = None
    answer: str
    citations: list[dict[str, str]] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    context_chunk_ids: list[str] = Field(default_factory=list)
    top_score: float | None = None
    oracle_leaks: list[str] = Field(default_factory=list)
    answer_leak: bool = False
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    judge: ItemJudgeResults | None = None


class EvalReport(BaseModel):
    """Complete evaluation run report saved as report.json and rendered to report.md."""

    timestamp: str
    git_commit: str
    config_hash: str
    golden_hash: str
    exploratory: bool = False
    no_judge: bool = False
    warnings: list[str] = Field(default_factory=list)
    models: dict[str, str] = Field(default_factory=dict)
    prompts: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    item_count: int = 0
    summary_metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    per_type_metrics: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)
    invalid_metrics: list[str] = Field(default_factory=list)
    items: list[ItemEvalRecord] = Field(default_factory=list)


def parse_set_overrides(set_args: list[str]) -> dict[str, Any]:
    """Parse --set key=value overrides into typed configuration values."""
    supported = {
        "retrieval.vector_k": int,
        "retrieval.fts_k": int,
        "retrieval.fused_k": int,
        "rerank.enabled": lambda s: s.strip().lower() in ("true", "1", "yes"),
        "rerank.top_n": int,
        "context.budget_tokens": int,
        "refusal.min_score": float,
    }
    parsed: dict[str, Any] = {}
    for raw in set_args:
        if "=" not in raw:
            raise ValueError(f"Invalid --set argument '{raw}': expected key=value")
        key, val = raw.split("=", 1)
        key = key.strip()
        if key not in supported:
            raise ValueError(
                f"Unsupported --set key '{key}'. Supported: {', '.join(sorted(supported))}"
            )
        caster = supported[key]
        parsed[key] = caster(val.strip())  # type: ignore[operator]
    return parsed


def apply_config_overrides(
    settings: Settings,
    overrides: dict[str, Any],
) -> tuple[RetrievalConfig, RerankConfig, ContextConfig, float]:
    """Build in-memory RetrievalConfig, RerankConfig, ContextConfig, and refusal_min_score."""
    ret = RetrievalConfig(
        vector_k=int(overrides.get("retrieval.vector_k", settings.RETRIEVAL_VECTOR_K)),
        fts_k=int(overrides.get("retrieval.fts_k", settings.RETRIEVAL_FTS_K)),
        fused_k=int(overrides.get("retrieval.fused_k", settings.RETRIEVAL_FUSED_K)),
        rrf_k=settings.RETRIEVAL_RRF_K,
        hnsw_ef_search=settings.RETRIEVAL_HNSW_EF_SEARCH,
    )
    rerank = RerankConfig(
        enabled=bool(overrides.get("rerank.enabled", settings.RERANK_ENABLED)),
        top_n=int(overrides.get("rerank.top_n", settings.RERANK_TOP_N)),
    )
    ctx = ContextConfig(
        budget_tokens=int(overrides.get("context.budget_tokens", settings.CONTEXT_BUDGET_TOKENS)),
        expand_parents=settings.CONTEXT_EXPAND_PARENTS,
    )
    min_score = float(overrides.get("refusal.min_score", settings.REFUSAL_MIN_SCORE))
    return ret, rerank, ctx, min_score


def compute_eval_config_hash(
    *,
    retrieval_cfg: RetrievalConfig,
    rerank_cfg: RerankConfig,
    context_cfg: ContextConfig,
    refusal_min_score: float,
    on_uncited: str,
    answer_prompt: tuple[str, str],
    models: dict[str, str],
    judge_prompt_versions: dict[str, str],
    golden_hash: str,
) -> str:
    """Compute deterministic SHA-256 hash over all evaluation configuration inputs."""
    payload = {
        "retrieval": retrieval_cfg.model_dump(mode="json"),
        "rerank": rerank_cfg.model_dump(mode="json"),
        "context": context_cfg.model_dump(mode="json"),
        "refusal_min_score": refusal_min_score,
        "on_uncited": on_uncited,
        "answer_prompt": {"name": answer_prompt[0], "version": answer_prompt[1]},
        "models": models,
        "judge_prompts": judge_prompt_versions,
        "golden_hash": golden_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _get_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _summarize_values(values: list[float]) -> MetricSummary:
    n = len(values)
    if n == 0:
        return MetricSummary(mean=None, n=0, std_error=None)
    mean = sum(values) / n
    if n < 2:
        return MetricSummary(mean=mean, n=n, std_error=0.0)
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(variance / n)
    return MetricSummary(mean=mean, n=n, std_error=se)


async def run_evaluation(
    *,
    golden_path: Path,
    out_dir: Path,
    session_factory: async_sessionmaker[AsyncSession],
    rag_service: RagService,
    settings: Settings,
    retrieval_cfg: RetrievalConfig,
    rerank_cfg: RerankConfig,
    context_cfg: ContextConfig,
    refusal_min_score: float,
    models: dict[str, str],
    judge: JudgeMetric | None = None,
    plan: CorpusPlan | None = None,
    corpus_root: Path | None = None,
    include_unreviewed: bool = False,
    limit: int | None = None,
    concurrency: int = 4,
    ks: tuple[int, ...] = (1, 3, 5, 8),
    run_preflight: bool = True,
) -> tuple[EvalReport, Path]:
    """Execute full evaluation suite and persist report.json and report.md."""
    items = load_golden(golden_path, reviewed_only=not include_unreviewed)
    items = sorted(items, key=lambda x: x.id)
    if limit is not None and limit > 0:
        items = items[:limit]

    if not items:
        raise ValueError(
            f"No {'eligible' if include_unreviewed else 'reviewed'} items found in {golden_path}. "
            "Review items first or pass --include-unreviewed for an exploratory run."
        )

    if run_preflight:
        async with session_factory() as session:
            problems = await check_corpus_against_golden(
                session=session,
                org_id=settings.DEFAULT_ORG_ID,
                items=items,
                plan=plan,
                root=corpus_root,
            )
            if problems:
                joined = "\n".join(f"  - {p}" for p in problems)
                raise ValueError(f"Corpus preflight failed ({len(problems)} problem(s)):\n{joined}")

    warnings: list[str] = []
    if include_unreviewed:
        warnings.append("EXPLORATORY RUN: includes unreviewed items; cannot be used for baselines.")
    if judge is not None and models.get("judge") and models.get("judge") == models.get("llm"):
        warnings.append(
            f"SAME-MODEL JUDGE WARNING: JUDGE_MODEL ({models['judge']}) equals "
            f"LLM_MODEL ({models['llm']}); see D-011."
        )

    g_hash = golden_file_hash(golden_path)
    judge_prompt_versions = judge.prompt_versions if judge is not None else {}
    cfg_hash = compute_eval_config_hash(
        retrieval_cfg=retrieval_cfg,
        rerank_cfg=rerank_cfg,
        context_cfg=context_cfg,
        refusal_min_score=refusal_min_score,
        on_uncited=settings.ON_UNCITED,
        answer_prompt=(rag_service.generator.prompt_name, rag_service.generator.prompt_version),
        models=models,
        judge_prompt_versions=judge_prompt_versions,
        golden_hash=g_hash,
    )

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _eval_one(item: GoldenItem) -> tuple[ItemEvalRecord, BehaviorObservation]:
        async with sem:
            async with session_factory() as session:
                hub_repo = HubRepository(session)
                hub_model = await hub_repo.get_by_name(settings.DEFAULT_ORG_ID, item.hub)
                if hub_model is None:
                    raise ValueError(f"Hub '{item.hub}' not found for item {item.id}")

                resolved = await resolve_item(session, settings.DEFAULT_ORG_ID, item)
                access = AccessContext(
                    org_id=settings.DEFAULT_ORG_ID,
                    hub_id=hub_model.id,
                    user_tags=frozenset(item.user_tags),
                )

                answer_res, trace = await rag_service.ask_detailed(item.question, access)

                ctx_ids = [c.chunk_id for c in trace.bundle.items]
                leaked_ids = await oracle_leaks(
                    access, trace.candidates, session, context_chunk_ids=ctx_ids
                )

            retrieved_list = [
                {
                    "chunk_id": str(c.chunk_id),
                    "score": c.final_score,
                    "relevant": c.chunk_id in resolved.relevant_chunk_ids,
                }
                for c in trace.candidates
            ]

            ret_metrics: dict[str, float] = {}
            is_answerable = not item.expected_refusal
            if is_answerable:
                ret_metrics = compute_item_retrieval_metrics(
                    retrieved_ids=[c.chunk_id for c in trace.candidates],
                    retrieved_texts=[c.text for c in trace.candidates],
                    context_texts=[c.text for c in trace.bundle.items],
                    relevant_ids=resolved.relevant_chunk_ids,
                    evidence_spans=item.evidence_spans,
                    ks=ks,
                )

            judge_res: ItemJudgeResults | None = None
            if judge is not None and is_answerable:
                judge_res = await judge.evaluate_item(
                    question=item.question,
                    reference_answer=item.reference_answer,
                    answer=answer_res.answer,
                    context_chunks=[c.text for c in trace.bundle.items],
                    is_answerable=is_answerable,
                    was_refused=answer_res.refusal,
                )

            ans_leak = item.type == GoldenType.access_restricted and any(
                span_in_text(span, answer_res.answer) for span in item.evidence_spans
            )

            rec = ItemEvalRecord(
                id=item.id,
                type=item.type,
                question=item.question,
                expected_refusal=item.expected_refusal,
                refused=answer_res.refusal,
                refusal_reason=(
                    answer_res.refusal_reason.value if answer_res.refusal_reason else None
                ),
                answer=answer_res.answer,
                citations=[
                    {"chunk_id": str(c.chunk_id), "document_title": c.document_title}
                    for c in answer_res.citations
                ],
                invalid_citations=list(answer_res.invalid_citations),
                retrieved=retrieved_list,
                context_chunk_ids=[str(cid) for cid in ctx_ids],
                top_score=trace.top_score,
                oracle_leaks=[str(lid) for lid in leaked_ids],
                answer_leak=ans_leak,
                retrieval_metrics=ret_metrics,
                judge=judge_res,
            )

            obs = BehaviorObservation(
                item_id=item.id,
                item_type=item.type,
                expected_refusal=item.expected_refusal,
                refused=answer_res.refusal,
                valid_citations_count=len(answer_res.citations),
                invalid_citations_count=len(answer_res.invalid_citations),
                oracle_leak_count=len(leaked_ids),
                answer_text=answer_res.answer,
                evidence_spans=list(item.evidence_spans),
            )
            return rec, obs

    pairs = await asyncio.gather(*[_eval_one(item) for item in items])
    pairs_sorted = sorted(pairs, key=lambda pair: pair[0].id)
    records = [p[0] for p in pairs_sorted]
    observations = [p[1] for p in pairs_sorted]

    # Aggregate summary metrics
    summary: dict[str, MetricSummary] = {}

    # 1. Retrieval metrics
    ret_buckets: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        for k_name, v_val in rec.retrieval_metrics.items():
            ret_buckets[k_name].append(v_val)
    for k_name, vals in sorted(ret_buckets.items()):
        summary[k_name] = _summarize_values(vals)

    # 2. Behavior metrics
    beh = compute_behavior_metrics(observations)
    for b_name, b_val in beh.items():
        if b_val is None:
            summary[b_name] = MetricSummary(mean=None, n=len(observations), std_error=None)
        else:
            summary[b_name] = MetricSummary(mean=float(b_val), n=len(observations), std_error=0.0)

    # 3. Judge metrics
    invalid_metrics: list[str] = []
    judge_metric_names = (
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
    )
    for jm in judge_metric_names:
        if judge is None:
            summary[jm] = MetricSummary(mean=None, n=0, std_error=None, skipped=True)
            continue

        scores: list[float] = []
        eligible = 0
        failures = 0
        for rec in records:
            if rec.judge is None:
                continue
            metric_obj = getattr(rec.judge, jm)
            if metric_obj is None:
                continue
            eligible += 1
            if metric_obj.failed:
                failures += 1
            elif metric_obj.score is not None:
                scores.append(metric_obj.score)

        fail_rate = (failures / eligible) if eligible > 0 else 0.0
        is_invalid = fail_rate > 0.10
        if is_invalid:
            invalid_metrics.append(jm)
            warnings.append(
                f"INVALID JUDGE METRIC '{jm}': failure rate {fail_rate:.1%} exceeds 10%."
            )

        ms = _summarize_values(scores)
        summary[jm] = MetricSummary(
            mean=ms.mean,
            n=ms.n,
            std_error=ms.std_error,
            judge_failures=failures,
            invalid=is_invalid,
            skipped=False,
        )

    # Per-type breakdown
    per_type: dict[str, dict[str, float | int | None]] = {}
    for gtype in GoldenType:
        type_recs = [r for r in records if r.type == gtype]
        if not type_recs:
            continue
        type_obs = [o for o in observations if o.item_type == gtype]
        type_beh = compute_behavior_metrics(type_obs)
        type_entry: dict[str, float | int | None] = {
            "count": len(type_recs),
            "refusal_accuracy": type_beh["refusal_accuracy"],
        }
        if not type_recs[0].expected_refusal:
            for mk in ("recall@5", "mrr", "ndcg@5"):
                vals = [r.retrieval_metrics[mk] for r in type_recs if mk in r.retrieval_metrics]
                type_entry[mk] = (sum(vals) / len(vals)) if vals else None
        per_type[gtype.value] = type_entry

    now_utc = datetime.now(UTC)
    ts_str = now_utc.strftime("%Y%m%dT%H%M%SZ")
    report = EvalReport(
        timestamp=now_utc.isoformat(),
        git_commit=_get_git_commit(),
        config_hash=cfg_hash,
        golden_hash=g_hash,
        exploratory=include_unreviewed,
        no_judge=(judge is None),
        warnings=warnings,
        models=models,
        prompts={
            "answer": {
                "name": rag_service.generator.prompt_name,
                "version": rag_service.generator.prompt_version,
            },
            "judge": judge_prompt_versions,
        },
        config={
            "retrieval": retrieval_cfg.model_dump(mode="json"),
            "rerank": rerank_cfg.model_dump(mode="json"),
            "context": context_cfg.model_dump(mode="json"),
            "refusal_min_score": refusal_min_score,
            "on_uncited": settings.ON_UNCITED,
        },
        item_count=len(records),
        summary_metrics=summary,
        per_type_metrics=per_type,
        invalid_metrics=invalid_metrics,
        items=records,
    )

    report_dir = out_dir / f"{ts_str}-{cfg_hash[:8]}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (report_dir / "report.md").write_text(render_report_markdown(report), encoding="utf-8")

    return report, report_dir


def render_report_markdown(report: EvalReport) -> str:
    """Render human-readable Markdown report with summary, per-type breakdown, and worst items."""
    lines: list[str] = [
        "# Evaluation Report",
        "",
        f"- **Timestamp**: `{report.timestamp}`",
        f"- **Git Commit**: `{report.git_commit}`",
        f"- **Config Hash**: `{report.config_hash}`",
        f"- **Golden Hash**: `{report.golden_hash}`",
        f"- **Exploratory**: `{report.exploratory}`",
        f"- **Items Evaluated**: `{report.item_count}`",
        f"- **Models**: `{json.dumps(report.models, sort_keys=True)}`",
        "",
    ]

    if report.warnings:
        lines.append("## Warnings")
        for w in report.warnings:
            lines.append(f"- **WARNING**: {w}")
        lines.append("")

    lines.extend(
        [
            "## Summary Metrics",
            "",
            "| Metric | Mean | N | Std Error | Failures | Status |",
            "|---|---|---|---|---|---|",
        ]
    )
    for name, ms in sorted(report.summary_metrics.items()):
        mean_str = f"{ms.mean:.4f}" if ms.mean is not None else "N/A"
        se_str = f"{ms.std_error:.4f}" if ms.std_error is not None else "N/A"
        status = "SKIPPED" if ms.skipped else ("INVALID" if ms.invalid else "OK")
        lines.append(
            f"| `{name}` | {mean_str} | {ms.n} | {se_str} | {ms.judge_failures} | {status} |"
        )

    lines.extend(["", "## Per-Type Breakdown", ""])
    for t_name, t_vals in sorted(report.per_type_metrics.items()):
        lines.append(f"- **{t_name}**: `{json.dumps(t_vals, sort_keys=True)}`")

    # Access and answer leaks
    leaks_list = [r for r in report.items if r.oracle_leaks or r.answer_leak]
    lines.extend(["", f"## Access & Answer Leaks ({len(leaks_list)})", ""])
    for lr in leaks_list:
        lines.append(f"- `{lr.id}`: oracle_leaks={lr.oracle_leaks}, answer_leak={lr.answer_leak}")

    # Worst items per metric (top 10 lowest recall@5)
    worst_ret = sorted(
        [r for r in report.items if "recall@5" in r.retrieval_metrics],
        key=lambda r: r.retrieval_metrics["recall@5"],
    )[:10]
    lines.extend(["", "## Worst Items (recall@5)", ""])
    for wr in worst_ret:
        lines.append(
            f"- `{wr.id}` (`{wr.type.value}`, recall@5={wr.retrieval_metrics['recall@5']:.2f}): "
            f"{wr.question}"
        )

    lines.extend(
        ["", "## Configuration", "", "```json", json.dumps(report.config, indent=2), "```", ""]
    )
    return "\n".join(lines)
