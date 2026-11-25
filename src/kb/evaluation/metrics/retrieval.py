"""Pure retrieval evaluation metrics (no I/O)."""

import math
from collections.abc import Sequence
from uuid import UUID

from kb.evaluation.golden import span_in_text


def recall_at_k(
    retrieved_texts: Sequence[str],
    evidence_spans: Sequence[str],
    k: int,
) -> float:
    """Fraction of evidence spans covered by at least one chunk in the top-k retrieved texts."""
    if not evidence_spans:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = retrieved_texts[:k]
    covered = sum(1 for span in evidence_spans if any(span_in_text(span, txt) for txt in top_k))
    return covered / len(evidence_spans)


def precision_at_k(
    retrieved_ids: Sequence[UUID],
    relevant_ids: set[UUID],
    k: int,
) -> float:
    """Fraction of top-k slots that are relevant (missing ranks count as non-relevant)."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / k


def mrr(
    retrieved_ids: Sequence[UUID],
    relevant_ids: set[UUID],
) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant chunk (1-based), or 0.0 if none."""
    for idx, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[UUID],
    relevant_ids: set[UUID],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k with binary relevance."""
    if k <= 0 or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, cid in enumerate(top_k, start=1) if cid in relevant_ids
    )
    ideal_count = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evidence_in_context(
    context_texts: Sequence[str],
    evidence_spans: Sequence[str],
) -> float:
    """Fraction of evidence spans present in the assembled context chunks."""
    if not evidence_spans:
        return 1.0
    covered = sum(
        1 for span in evidence_spans if any(span_in_text(span, txt) for txt in context_texts)
    )
    return covered / len(evidence_spans)


def compute_item_retrieval_metrics(
    *,
    retrieved_ids: Sequence[UUID],
    retrieved_texts: Sequence[str],
    context_texts: Sequence[str],
    relevant_ids: set[UUID],
    evidence_spans: Sequence[str],
    ks: Sequence[int] = (1, 3, 5, 8),
) -> dict[str, float]:
    """Compute all retrieval metrics for a single answerable item."""
    metrics: dict[str, float] = {
        "mrr": mrr(retrieved_ids, relevant_ids),
        "evidence_in_context": evidence_in_context(context_texts, evidence_spans),
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_texts, evidence_spans, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    return metrics
