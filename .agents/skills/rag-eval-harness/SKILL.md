---
name: rag-eval-harness
description: Conventions for the golden dataset, retrieval and generation metrics, baselines, and the CI regression gate for the RAG pipeline. Use whenever working on evaluation, golden sets, RAGAS or DeepEval, recall, precision, MRR, nDCG, faithfulness, answer relevance, judge models, benchmarks, or regression gating.
---

# RAG evaluation harness

## Golden set (`eval/golden/*.jsonl`)

One JSON object per line:

```json
{"id": "q-001", "question": "...", "reference_answer": "...",
 "source_document_ids": ["..."], "source_chunk_ids": ["..."],
 "type": "factual|multi_hop|table|unanswerable|access_restricted",
 "user_tags": [], "hub_id": "...", "reviewed": false, "notes": ""}
```

- Validate with a pydantic model on load. Reject malformed lines with line numbers.
- At least 10% `unanswerable` (expected refusal) and some `access_restricted` (expected refusal for a user without the tag).
- Synthetic items start `reviewed=false`. Only `reviewed=true` items count toward the gate.
- Chunk IDs can change when a document is re-chunked. Prefer document IDs plus an answer-bearing text span for matching, and keep chunk IDs as a convenience.

## Metrics

- Retrieval: recall@k, precision@k, MRR, nDCG. Implement directly and unit-test against hand-computed fixtures.
- Generation: faithfulness, answer relevance, context precision, context recall via RAGAS or DeepEval behind our wrapper.
- Behavior: refusal accuracy (should refuse vs did refuse), citation validity rate.

## Judge model

Configurable and recorded in every report. Use a different model from the generator where possible (D-011). Judges are noisy: report per-metric means with sample counts, and re-run borderline failures before concluding a regression.

## Reports and baselines

- `kb eval run` writes JSON and markdown to `eval/reports/` including: commit, config hash (retrieval params, prompt version, models), per-type breakdown, and failing examples.
- Baselines live in `eval/baselines/` and are updated deliberately in their own commit.

## CI gate

Fail when any gated metric drops by more than its threshold versus baseline. Thresholds live in config, not code. Prove the gate works with a test that degrades the config (for example k=1) and expects failure.
