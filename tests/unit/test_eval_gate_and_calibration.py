"""Unit tests for regression gate, baseline promotion, calibration, and config hash."""

import json
from pathlib import Path

import pytest

from kb.domain.evaluation import GoldenType
from kb.domain.retrieval import ContextConfig, RerankConfig, RetrievalConfig
from kb.evaluation.calibrate import calibrate_report
from kb.evaluation.gate import evaluate_gate, save_baseline
from kb.evaluation.golden import golden_file_hash
from kb.evaluation.runner import (
    EvalReport,
    ItemEvalRecord,
    MetricSummary,
    compute_eval_config_hash,
)


def _make_report(
    tmp_path: Path,
    *,
    recall5: float = 0.90,
    faith: float = 0.85,
    false_ref: float = 0.05,
    access_leaks: int = 0,
    answer_leaks: int = 0,
    golden_hash: str = "ghash1",
    config_hash: str = "chash1",
    exploratory: bool = False,
    no_judge: bool = False,
    invalid_judge: bool = False,
    omit_metric: str | None = None,
) -> Path:
    summary = {
        "recall@5": MetricSummary(mean=recall5, n=10, std_error=0.02),
        "mrr": MetricSummary(mean=0.85, n=10, std_error=0.02),
        "ndcg@5": MetricSummary(mean=0.88, n=10, std_error=0.02),
        "refusal_accuracy": MetricSummary(mean=0.95, n=12, std_error=0.01),
        "citation_validity_rate": MetricSummary(mean=0.98, n=10, std_error=0.01),
        "false_refusal_rate": MetricSummary(mean=false_ref, n=10, std_error=0.01),
        "missed_refusal_rate": MetricSummary(mean=0.05, n=2, std_error=0.01),
        "access_leaks": MetricSummary(mean=float(access_leaks), n=12, std_error=0.0),
        "answer_leaks": MetricSummary(mean=float(answer_leaks), n=12, std_error=0.0),
        "faithfulness": MetricSummary(
            mean=None if no_judge else faith,
            n=0 if no_judge else 10,
            skipped=no_judge,
            invalid=invalid_judge,
        ),
        "answer_relevance": MetricSummary(mean=None if no_judge else 0.85, n=10, skipped=no_judge),
        "context_precision": MetricSummary(mean=None if no_judge else 0.80, n=10, skipped=no_judge),
        "context_recall": MetricSummary(mean=None if no_judge else 0.85, n=10, skipped=no_judge),
    }
    if omit_metric and omit_metric in summary:
        del summary[omit_metric]

    report = EvalReport(
        timestamp="2026-11-26T00:00:00Z",
        git_commit="abc1234",
        config_hash=config_hash,
        golden_hash=golden_hash,
        exploratory=exploratory,
        no_judge=no_judge,
        invalid_metrics=["faithfulness"] if invalid_judge else [],
        item_count=12,
        summary_metrics=summary,
    )
    rdir = tmp_path / f"rep-{config_hash}-{recall5}-{faith}-{false_ref}-{access_leaks}"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return rdir


def test_gate_thresholds_hard_gates_and_edge_cases(tmp_path: Path) -> None:
    """Gate handles pass, boundary, drop/rise fail, hard gates, hash mismatch, SKIPPED, invalid."""
    thresholds_path = Path("eval/thresholds.json")
    baselines_dir = tmp_path / "baselines"

    base_dir = _make_report(tmp_path, recall5=0.90, faith=0.85, false_ref=0.05)
    baseline_path = save_baseline(base_dir, name="default", baselines_dir=baselines_dir)

    # 1. Exact boundary passes (recall@5 drop == 0.03, false_refusal_rate rise == 0.05)
    boundary_dir = _make_report(tmp_path, recall5=0.87, faith=0.85, false_ref=0.10)
    res_bound = evaluate_gate(
        report_path=boundary_dir, baseline_path=baseline_path, thresholds_path=thresholds_path
    )
    assert res_bound.passed is True

    # 2. Beyond threshold fails (recall@5 drop 0.04 > 0.03)
    drop_dir = _make_report(tmp_path, recall5=0.86, faith=0.85, false_ref=0.05)
    res_drop = evaluate_gate(
        report_path=drop_dir, baseline_path=baseline_path, thresholds_path=thresholds_path
    )
    assert res_drop.passed is False
    assert any(r.metric == "recall@5" and r.status == "FAIL" for r in res_drop.rows)

    # 3. max_rise fails (false_refusal_rate rise 0.06 > 0.05)
    rise_dir = _make_report(tmp_path, recall5=0.90, faith=0.85, false_ref=0.11)
    res_rise = evaluate_gate(
        report_path=rise_dir, baseline_path=baseline_path, thresholds_path=thresholds_path
    )
    assert res_rise.passed is False
    assert any(r.metric == "false_refusal_rate" and r.status == "FAIL" for r in res_rise.rows)

    # 4. Hard gate fails even with nonexistent baseline
    leak_dir = _make_report(tmp_path, access_leaks=1)
    res_leak = evaluate_gate(
        report_path=leak_dir,
        baseline_path=tmp_path / "missing.json",
        thresholds_path=thresholds_path,
    )
    assert res_leak.passed is False
    assert any("access_leaks" in hf for hf in res_leak.hard_failures)

    # 5. Golden hash mismatch fails unless allow_golden_change=True
    ghash_dir = _make_report(tmp_path, golden_hash="different_ghash")
    assert (
        evaluate_gate(
            report_path=ghash_dir, baseline_path=baseline_path, thresholds_path=thresholds_path
        ).passed
        is False
    )
    assert (
        evaluate_gate(
            report_path=ghash_dir,
            baseline_path=baseline_path,
            thresholds_path=thresholds_path,
            allow_golden_change=True,
        ).passed
        is True
    )

    # 6. --no-judge marks judge metrics SKIPPED (never PASS)
    no_judge_dir = _make_report(tmp_path, no_judge=True)
    res_nj = evaluate_gate(
        report_path=no_judge_dir, baseline_path=baseline_path, thresholds_path=thresholds_path
    )
    assert res_nj.passed is True
    assert all(
        r.status == "SKIPPED"
        for r in res_nj.rows
        if r.metric in ("faithfulness", "answer_relevance", "context_precision", "context_recall")
    )

    # 7. Invalid judge metric fails gate & baseline promotion
    inv_dir = _make_report(tmp_path, invalid_judge=True)
    assert (
        evaluate_gate(
            report_path=inv_dir, baseline_path=baseline_path, thresholds_path=thresholds_path
        ).passed
        is False
    )
    with pytest.raises(ValueError, match="invalid judge metrics"):
        save_baseline(inv_dir, baselines_dir=baselines_dir)

    # 8. Exploratory report cannot be baselined
    exp_dir = _make_report(tmp_path, exploratory=True)
    with pytest.raises(ValueError, match="exploratory"):
        save_baseline(exp_dir, baselines_dir=baselines_dir)

    # 9. Missing metric in report fails clearly
    miss_dir = _make_report(tmp_path, omit_metric="mrr")
    res_miss = evaluate_gate(
        report_path=miss_dir, baseline_path=baseline_path, thresholds_path=thresholds_path
    )
    assert res_miss.passed is False
    assert any("MISSING METRIC" in hf for hf in res_miss.hard_failures)


def test_calibrate_recommends_threshold_and_writes_calibration_md(tmp_path: Path) -> None:
    """Calibration finds optimal threshold under false_refusal_rate <= 0.10."""
    items = [
        # Answerable items with high scores (0.70, 0.80, 0.90)
        ItemEvalRecord(
            id="q-0001",
            type=GoldenType.factual,
            question="q1",
            expected_refusal=False,
            refused=False,
            answer="a1",
            top_score=0.70,
        ),
        ItemEvalRecord(
            id="q-0002",
            type=GoldenType.factual,
            question="q2",
            expected_refusal=False,
            refused=False,
            answer="a2",
            top_score=0.80,
        ),
        ItemEvalRecord(
            id="q-0003",
            type=GoldenType.factual,
            question="q3",
            expected_refusal=False,
            refused=False,
            answer="a3",
            top_score=0.90,
        ),
        # Unanswerable items with low scores (0.20, 0.30)
        ItemEvalRecord(
            id="q-0004",
            type=GoldenType.unanswerable,
            question="q4",
            expected_refusal=True,
            refused=False,
            answer="a4",
            top_score=0.20,
        ),
        ItemEvalRecord(
            id="q-0005",
            type=GoldenType.unanswerable,
            question="q5",
            expected_refusal=True,
            refused=False,
            answer="a5",
            top_score=0.30,
        ),
    ]
    report = EvalReport(
        timestamp="2026-11-26T00:00:00Z",
        git_commit="abc",
        config_hash="cfg",
        golden_hash="gld",
        config={"rerank": {"enabled": True}},
        item_count=5,
        items=items,
    )
    rdir = tmp_path / "cal_rep"
    rdir.mkdir()
    (rdir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    res = calibrate_report(rdir, max_false_refusal_rate=0.10)
    # At 0.70: false_refusal=0.0, missed_refusal=0.0 -> balanced_accuracy=1.0
    assert res.recommended_threshold == pytest.approx(0.70)
    assert res.recommended_balanced_accuracy == pytest.approx(1.0)
    assert (rdir / "calibration.md").exists()
    assert {p.name for p in rdir.iterdir()} == {"report.json", "calibration.md"}


def test_config_hash_sensitivity(tmp_path: Path) -> None:
    """Config hash is deterministic and changes whenever any single component changes."""
    gpath = tmp_path / "g.jsonl"
    gpath.write_text(
        json.dumps(
            {
                "id": "q-0001",
                "question": "Q1?",
                "reference_answer": "A1.",
                "type": "factual",
                "source_uris": ["d.pdf"],
                "evidence_spans": ["A1."],
                "provenance": "handwritten",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gh1 = golden_file_hash(gpath)
    base = compute_eval_config_hash(
        retrieval_cfg=RetrievalConfig(),
        rerank_cfg=RerankConfig(),
        context_cfg=ContextConfig(),
        refusal_min_score=0.0,
        on_uncited="retry",
        answer_prompt=("answer", "1.0.0"),
        models={"llm": "m1", "embedder": "e1", "reranker": "r1", "judge": "j1"},
        judge_prompt_versions={"judge_faithfulness": "1.0.0"},
        golden_hash=gh1,
    )
    # Identical call -> identical hash
    assert base == compute_eval_config_hash(
        retrieval_cfg=RetrievalConfig(),
        rerank_cfg=RerankConfig(),
        context_cfg=ContextConfig(),
        refusal_min_score=0.0,
        on_uncited="retry",
        answer_prompt=("answer", "1.0.0"),
        models={"llm": "m1", "embedder": "e1", "reranker": "r1", "judge": "j1"},
        judge_prompt_versions={"judge_faithfulness": "1.0.0"},
        golden_hash=gh1,
    )
    # Change --set override
    assert base != compute_eval_config_hash(
        retrieval_cfg=RetrievalConfig(vector_k=10),
        rerank_cfg=RerankConfig(),
        context_cfg=ContextConfig(),
        refusal_min_score=0.0,
        on_uncited="retry",
        answer_prompt=("answer", "1.0.0"),
        models={"llm": "m1", "embedder": "e1", "reranker": "r1", "judge": "j1"},
        judge_prompt_versions={"judge_faithfulness": "1.0.0"},
        golden_hash=gh1,
    )
    # Change prompt version
    assert base != compute_eval_config_hash(
        retrieval_cfg=RetrievalConfig(),
        rerank_cfg=RerankConfig(),
        context_cfg=ContextConfig(),
        refusal_min_score=0.0,
        on_uncited="retry",
        answer_prompt=("answer", "2.0.0"),
        models={"llm": "m1", "embedder": "e1", "reranker": "r1", "judge": "j1"},
        judge_prompt_versions={"judge_faithfulness": "1.0.0"},
        golden_hash=gh1,
    )
    # Change model id
    assert base != compute_eval_config_hash(
        retrieval_cfg=RetrievalConfig(),
        rerank_cfg=RerankConfig(),
        context_cfg=ContextConfig(),
        refusal_min_score=0.0,
        on_uncited="retry",
        answer_prompt=("answer", "1.0.0"),
        models={"llm": "m2", "embedder": "e1", "reranker": "r1", "judge": "j1"},
        judge_prompt_versions={"judge_faithfulness": "1.0.0"},
        golden_hash=gh1,
    )
    # Change golden file
    gpath.write_text(
        json.dumps(
            {
                "id": "q-0001",
                "question": "Q1 changed?",
                "reference_answer": "A1.",
                "type": "factual",
                "source_uris": ["d.pdf"],
                "evidence_spans": ["A1."],
                "provenance": "handwritten",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gh2 = golden_file_hash(gpath)
    assert base != compute_eval_config_hash(
        retrieval_cfg=RetrievalConfig(),
        rerank_cfg=RerankConfig(),
        context_cfg=ContextConfig(),
        refusal_min_score=0.0,
        on_uncited="retry",
        answer_prompt=("answer", "1.0.0"),
        models={"llm": "m1", "embedder": "e1", "reranker": "r1", "judge": "j1"},
        judge_prompt_versions={"judge_faithfulness": "1.0.0"},
        golden_hash=gh2,
    )
