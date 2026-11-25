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

- Retrieval (`src/kb/evaluation/metrics/retrieval.py`): `recall@k` (span coverage in top-k), `precision@k` (relevant/k, missing ranks non-relevant), `mrr`, `ndcg@k` (binary relevance, ideal min(k, |rel|)), and `evidence_in_context` for `k in {1,3,5,8}` over answerable items only.
- Behavior (`src/kb/evaluation/metrics/behavior.py`): `refusal_accuracy`, `false_refusal_rate`, `missed_refusal_rate`, `citation_validity_rate`, `access_leaks` (via pure-Python `access_oracle.py`), and `answer_leaks`.
- Generation (`src/kb/evaluation/metrics/judge.py`): `faithfulness`, `answer_relevance`, `context_precision`, `context_recall` behind `JudgeMetric` using `LLMClient` (I7) and versioned prompts in `prompts/judge_*.v1.0.0.md` (I9). Malformed JSON retries once, then records `judge_failure`; >10% failure marks the metric `invalid`.

## Judge model

Configurable via `JUDGE_PROVIDER`, `JUDGE_MODEL`, and `JUDGE_BASE_URL` and recorded in every report (`build_judge`). Use a different model from the generator (D-011); same-model use emits a warning.

## Reports and baselines

- `kb eval run` writes `eval/reports/<UTC>-<confighash8>/report.json` and `report.md` (summary mean/n/std_error, per-type breakdown, 10 worst items per metric, leaks, warnings, config & golden hashes, commit, models).
- `kb eval baseline set REPORT_DIR --name default` writes `eval/baselines/<name>.json` (refuses `exploratory` or `invalid` reports). Updating a baseline is a deliberate separate commit.

## CI gate & Calibration

- `kb eval gate` checks `eval/thresholds.json` (`max_drop`, `max_rise`, hard gates `access_leaks==0` & `answer_leaks==0`, invalid judge metric check, and golden hash match). `--no-judge` reports show `SKIPPED` for judge metrics.
- `kb eval calibrate` sweeps observed `top_score` values, writes `calibration.md` into the report directory, and recommends a `REFUSAL_MIN_SCORE` without modifying config files.
