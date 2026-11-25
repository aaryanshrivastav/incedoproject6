"""Refusal threshold calibration sweep over evaluation report scores."""

from pathlib import Path

from pydantic import BaseModel, Field

from kb.evaluation.gate import load_report


class CalibrationPoint(BaseModel):
    """Metrics at one candidate REFUSAL_MIN_SCORE threshold."""

    threshold: float
    false_refusal_rate: float
    missed_refusal_rate: float
    balanced_accuracy: float


class CalibrationResult(BaseModel):
    """Full sweep result and recommended threshold."""

    recommended_threshold: float | None = None
    recommended_balanced_accuracy: float | None = None
    max_false_refusal_rate: float
    reranked: bool
    points: list[CalibrationPoint] = Field(default_factory=list)
    caveat: str


def calibrate_report(
    report_dir: Path,
    *,
    max_false_refusal_rate: float = 0.10,
) -> CalibrationResult:
    """Sweep candidate REFUSAL_MIN_SCORE thresholds from observed top_score distribution.

    Writes calibration.md into report_dir and never modifies config files or environment.
    """
    report = load_report(report_dir)
    reranked = bool(report.config.get("rerank", {}).get("enabled", False))
    score_kind = "rerank_score" if reranked else "vector_similarity"
    caveat = (
        f"Note: top_score in this report is {score_kind}-based (rerank.enabled={reranked}). "
        "The recommended threshold is only valid for the configuration that produced this report."
    )

    answerable = [r for r in report.items if not r.expected_refusal]
    should_refuse = [r for r in report.items if r.expected_refusal]

    observed_scores = sorted(
        {round(r.top_score, 6) for r in report.items if r.top_score is not None}
    )
    candidates = [0.0] + [s for s in observed_scores if s > 0.0]
    if observed_scores:
        candidates.append(round(observed_scores[-1] + 0.05, 6))

    points: list[CalibrationPoint] = []
    for thresh in candidates:
        # Simulate gating: item is refused if top_score is None or top_score < thresh
        # (for thresh == 0.0, gating is disabled and only empty context refuses)
        if answerable:
            false_ref = sum(
                1
                for r in answerable
                if r.top_score is None or (thresh > 0.0 and r.top_score < thresh)
            ) / len(answerable)
        else:
            false_ref = 0.0

        if should_refuse:
            missed_ref = sum(
                1
                for r in should_refuse
                if r.top_score is not None and (thresh == 0.0 or r.top_score >= thresh)
            ) / len(should_refuse)
        else:
            missed_ref = 0.0

        bal_acc = 1.0 - 0.5 * (false_ref + missed_ref)
        points.append(
            CalibrationPoint(
                threshold=thresh,
                false_refusal_rate=false_ref,
                missed_refusal_rate=missed_ref,
                balanced_accuracy=bal_acc,
            )
        )

    feasible = [p for p in points if p.false_refusal_rate <= max_false_refusal_rate]
    best = (
        max(feasible, key=lambda p: (p.balanced_accuracy, -p.false_refusal_rate, p.threshold))
        if feasible
        else None
    )

    result = CalibrationResult(
        recommended_threshold=best.threshold if best else None,
        recommended_balanced_accuracy=best.balanced_accuracy if best else None,
        max_false_refusal_rate=max_false_refusal_rate,
        reranked=reranked,
        points=points,
        caveat=caveat,
    )

    out_dir = report_dir if report_dir.is_dir() else report_dir.parent
    md_lines = [
        "# Refusal Threshold Calibration",
        "",
        f"> {caveat}",
        "",
        f"- **Constraint**: `false_refusal_rate <= {max_false_refusal_rate:.2f}`",
        (
            f"- **Recommended `REFUSAL_MIN_SCORE`**: `{best.threshold:.4f}` "
            f"(balanced_accuracy={best.balanced_accuracy:.4f})"
            if best
            else "- **Recommended `REFUSAL_MIN_SCORE`**: None satisfied the constraint"
        ),
        "",
        "| Threshold | False Refusal Rate | Missed Refusal Rate | Balanced Accuracy |",
        "|---|---|---|---|",
    ]
    for pt in points:
        md_lines.append(
            f"| {pt.threshold:.4f} | {pt.false_refusal_rate:.4f} | "
            f"{pt.missed_refusal_rate:.4f} | {pt.balanced_accuracy:.4f} |"
        )
    (out_dir / "calibration.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return result
