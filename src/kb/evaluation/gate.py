"""Baseline persistence and CI regression gate evaluation."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kb.evaluation.runner import EvalReport, MetricSummary


class BaselineRecord(BaseModel):
    """Committed summary baseline stored in eval/baselines/<name>.json (no per-item data)."""

    name: str
    timestamp: str
    git_commit: str
    config_hash: str
    golden_hash: str
    item_count: int
    models: dict[str, str] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    summary_metrics: dict[str, MetricSummary] = Field(default_factory=dict)


class GateMetricRow(BaseModel):
    """Single row in the regression gate comparison table."""

    metric: str
    baseline: float | None
    current: float | None
    delta: float | None
    std_error: float | None
    threshold: str
    status: str  # PASS | FAIL | SKIPPED


class GateResult(BaseModel):
    """Outcome of running the regression gate against a baseline and thresholds file."""

    passed: bool
    rows: list[GateMetricRow] = Field(default_factory=list)
    hard_failures: list[str] = Field(default_factory=list)
    info_messages: list[str] = Field(default_factory=list)


def load_report(report_dir_or_file: Path) -> EvalReport:
    """Load report.json from a report directory or direct file path."""
    target = (
        report_dir_or_file / "report.json" if report_dir_or_file.is_dir() else report_dir_or_file
    )
    if not target.exists():
        raise FileNotFoundError(f"Evaluation report not found: {target}")
    return EvalReport.model_validate_json(target.read_text(encoding="utf-8"))


def save_baseline(
    report_dir: Path,
    *,
    name: str = "default",
    baselines_dir: Path = Path("eval/baselines"),
) -> Path:
    """Promote a validated, non-exploratory report to eval/baselines/<name>.json."""
    report = load_report(report_dir)
    if report.exploratory:
        raise ValueError("Cannot promote an exploratory report (--include-unreviewed) to baseline.")
    if report.invalid_metrics:
        raise ValueError(
            f"Cannot promote report with invalid judge metrics: {report.invalid_metrics}"
        )

    baseline = BaselineRecord(
        name=name,
        timestamp=report.timestamp,
        git_commit=report.git_commit,
        config_hash=report.config_hash,
        golden_hash=report.golden_hash,
        item_count=report.item_count,
        models=report.models,
        config=report.config,
        summary_metrics=report.summary_metrics,
    )
    baselines_dir.mkdir(parents=True, exist_ok=True)
    out_path = baselines_dir / f"{name}.json"
    out_path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def evaluate_gate(
    *,
    report_path: Path,
    baseline_path: Path,
    thresholds_path: Path,
    allow_golden_change: bool = False,
) -> GateResult:
    """Evaluate hard gates and threshold regressions for a report against a baseline."""
    report = load_report(report_path)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    hard_cfg = thresholds.get("hard_gates", {})

    hard_failures: list[str] = []
    info_messages: list[str] = []
    rows: list[GateMetricRow] = []

    # 1. Hard gates (no baseline needed)
    access_leaks_ms = report.summary_metrics.get("access_leaks")
    access_leaks = int(access_leaks_ms.mean or 0) if access_leaks_ms else 0
    if access_leaks > int(hard_cfg.get("max_access_leaks", 0)):
        hard_failures.append(f"HARD GATE FAILED: access_leaks={access_leaks} > 0")

    answer_leaks_ms = report.summary_metrics.get("answer_leaks")
    answer_leaks = int(answer_leaks_ms.mean or 0) if answer_leaks_ms else 0
    if answer_leaks > int(hard_cfg.get("max_answer_leaks", 0)):
        hard_failures.append(f"HARD GATE FAILED: answer_leaks={answer_leaks} > 0")

    if report.exploratory:
        hard_failures.append("HARD GATE FAILED: exploratory reports cannot pass the gate.")

    # 2. Load baseline & compare hashes
    if not baseline_path.exists():
        hard_failures.append(f"Baseline file not found: {baseline_path}")
        return GateResult(passed=False, rows=rows, hard_failures=hard_failures)

    baseline = BaselineRecord.model_validate_json(baseline_path.read_text(encoding="utf-8"))

    if report.golden_hash != baseline.golden_hash and not allow_golden_change:
        hard_failures.append(
            f"Golden hash changed ({baseline.golden_hash[:8]} -> {report.golden_hash[:8]}); "
            "pass --allow-golden-change to override."
        )
    if report.config_hash != baseline.config_hash:
        info_messages.append(
            f"INFO: config_hash changed ({baseline.config_hash[:8]} -> {report.config_hash[:8]})."
        )

    # 3. Check max_drop (higher is better) & max_rise (lower is better)
    eps = 1e-12
    for metric_name, max_drop in thresholds.get("max_drop", {}).items():
        curr_ms = report.summary_metrics.get(metric_name)
        base_ms = baseline.summary_metrics.get(metric_name)

        if curr_ms is not None and curr_ms.skipped:
            rows.append(
                GateMetricRow(
                    metric=metric_name,
                    baseline=base_ms.mean if base_ms else None,
                    current=None,
                    delta=None,
                    std_error=None,
                    threshold=f"-{max_drop:.3f}",
                    status="SKIPPED",
                )
            )
            continue

        if curr_ms is not None and curr_ms.invalid:
            hard_failures.append(f"HARD GATE FAILED: gated metric '{metric_name}' is invalid.")
            rows.append(
                GateMetricRow(
                    metric=metric_name,
                    baseline=base_ms.mean if base_ms else None,
                    current=curr_ms.mean,
                    delta=None,
                    std_error=curr_ms.std_error,
                    threshold=f"-{max_drop:.3f}",
                    status="FAIL",
                )
            )
            continue

        if curr_ms is None or curr_ms.mean is None:
            hard_failures.append(f"MISSING METRIC: '{metric_name}' missing from current report.")
            rows.append(
                GateMetricRow(
                    metric=metric_name,
                    baseline=base_ms.mean if base_ms else None,
                    current=None,
                    delta=None,
                    std_error=None,
                    threshold=f"-{max_drop:.3f}",
                    status="FAIL",
                )
            )
            continue

        if base_ms is None or base_ms.mean is None:
            rows.append(
                GateMetricRow(
                    metric=metric_name,
                    baseline=None,
                    current=curr_ms.mean,
                    delta=None,
                    std_error=curr_ms.std_error,
                    threshold=f"-{max_drop:.3f}",
                    status="SKIPPED",
                )
            )
            continue

        delta = curr_ms.mean - base_ms.mean
        status = "PASS" if delta >= (-float(max_drop) - eps) else "FAIL"
        rows.append(
            GateMetricRow(
                metric=metric_name,
                baseline=base_ms.mean,
                current=curr_ms.mean,
                delta=delta,
                std_error=curr_ms.std_error,
                threshold=f"-{float(max_drop):.3f}",
                status=status,
            )
        )

    for metric_name, max_rise in thresholds.get("max_rise", {}).items():
        curr_ms = report.summary_metrics.get(metric_name)
        base_ms = baseline.summary_metrics.get(metric_name)

        if curr_ms is None or curr_ms.mean is None:
            hard_failures.append(f"MISSING METRIC: '{metric_name}' missing from current report.")
            rows.append(
                GateMetricRow(
                    metric=metric_name,
                    baseline=base_ms.mean if base_ms else None,
                    current=None,
                    delta=None,
                    std_error=None,
                    threshold=f"+{max_rise:.3f}",
                    status="FAIL",
                )
            )
            continue

        if base_ms is None or base_ms.mean is None:
            rows.append(
                GateMetricRow(
                    metric=metric_name,
                    baseline=None,
                    current=curr_ms.mean,
                    delta=None,
                    std_error=curr_ms.std_error,
                    threshold=f"+{max_rise:.3f}",
                    status="SKIPPED",
                )
            )
            continue

        delta = curr_ms.mean - base_ms.mean
        status = "PASS" if delta <= (float(max_rise) + eps) else "FAIL"
        rows.append(
            GateMetricRow(
                metric=metric_name,
                baseline=base_ms.mean,
                current=curr_ms.mean,
                delta=delta,
                std_error=curr_ms.std_error,
                threshold=f"+{float(max_rise):.3f}",
                status=status,
            )
        )

    any_metric_fail = any(r.status == "FAIL" for r in rows)
    passed = (not hard_failures) and (not any_metric_fail)
    return GateResult(
        passed=passed,
        rows=rows,
        hard_failures=hard_failures,
        info_messages=info_messages,
    )
