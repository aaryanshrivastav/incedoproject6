---
name: rag-eval-harness
description: Conventions for the golden dataset, retrieval and generation metrics, baselines, and the CI regression gate for the RAG pipeline. Use whenever working on evaluation, golden sets, RAGAS or DeepEval, recall, precision, MRR, nDCG, faithfulness, answer relevance, judge models, benchmarks, or regression gating.
---

# RAG evaluation harness

## Golden set (`eval/golden/*.jsonl`)

One JSON object per line matching `kb.domain.evaluation.GoldenItem`:

```json
{"id": "q-0001", "question": "...", "reference_answer": "...",
 "type": "factual|multi_hop|table|unanswerable|access_restricted",
 "hub": "default", "user_tags": [], "source_uris": ["..."],
 "evidence_spans": ["..."], "restriction_kind": null, "control_for": null,
 "provenance": "manifest|llm|handwritten", "reviewed": false,
 "reviewed_by": null, "reviewed_at": null, "notes": "", "tags": []}
```

- Validate with `kb.evaluation.golden.load_golden`. Collect and report all malformed/invalid lines with line numbers via `GoldenFormatError`.
- At least 10% `unanswerable` (expected refusal, empty `source_uris`, `evidence_spans`, `reference_answer`) and `access_restricted` (`acl_tag` and `other_hub`, each paired with an authorized control item via `control_for`).
- Synthetic/generated items start `reviewed=false`. Only `reviewed=true` items count toward the gate. Items with `provenance="llm"` can never be bulk-accepted.
- Chunk and document UUIDs can change when a document is re-ingested or re-chunked. Items reference relative `source_uris` and exact `evidence_spans`, resolved at runtime.

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
