# Phase 2 audit

- Audited commit: `b502e95ec32d042c76cae91caa7b7464cb63f3b4`
- Date: `2026-11-25`
- Auditor session: fresh, read-only
- Verdict: **PASS**

## Standard checks

| Check | Command | Result |
|---|---|---|
| Tests | `uv run --no-sync pytest tests/unit -q` | PASS (60 passed in 9.63s) |
| Integration Tests | `uv run --no-sync pytest tests/integration -v` | PASS (35 passed, 4 warnings in 252.11s against Supabase + live Ollama) |
| Lint | `uv run --no-sync ruff check .` | PASS (All checks passed!) |
| Format | `uv run --no-sync ruff format --check .` | PASS (106 files already formatted) |
| Types | `uv run --no-sync mypy src` | PASS (Success: no issues found in 50 source files; strict mode enabled) |

## Exit criteria

| ID | Criterion (short) | Status | Evidence |
|---|---|---|---|
| E2.1 | Hybrid retrieval fuses vector and full-text ranks (RRF) and returns top-k with scores | PASS | `src/kb/retrieval/hybrid.py:166-251` implements RRF (`1/(rrf_k+rank)`). `tests/unit/test_retrieval_unit.py::test_rrf_hand_computed_scores_and_tie_break` and `tests/integration/test_hybrid.py::test_hybrid_retrieval_determinism` pass. |
| E2.2 | Access invariant I1 enforced in SQL (ACL tags, hub isolation, global scope, active status) | PASS | `src/kb/retrieval/filters.py:17-69` (`eligible_chunk_conditions`) applied in SQL in `hybrid.py:108,143` and `parents.py:41`. All 9 tests in `tests/integration/test_retrieval_access.py`, `test_parent_expansion_unauthorized_not_substituted`, and `test_access_isolated_rag_query` pass. |
| E2.3 | Filtered queries return k results when >=k eligible chunks exist (ANN filtering check) | PASS | `src/kb/retrieval/hybrid.py:88-89` sets `SET LOCAL hnsw.iterative_scan = relaxed_order` and `hnsw.ef_search`. `tests/integration/test_hybrid.py::test_iterative_scan_assert_active_and_returns_k_eligible_under_filter` passes (returns 5/5 eligible chunks despite 150 closer distractors). |
| E2.4 | Reranker sits behind Protocol and can be toggled for ablation | PASS | `RerankerProtocol` in `src/kb/domain/protocols.py:86-91`; `RerankStage` in `src/kb/retrieval/rerank.py:47-56` bypasses reranker when `enabled=False`. `test_reranker_ablation_disabled` passes and `kb ask ... --no-rerank --json` executed cleanly. |
| E2.5 | Context assembly respects token budget, dedupes, orders deterministically, and expands parents | PASS | `src/kb/retrieval/context.py:73-188`. `tests/unit/test_rerank_and_context.py` (`test_context_deduplication`, `test_context_budget_greedy_fill_no_truncation`, `test_context_deterministic_ordering_shuffled`) and `test_parent_expansion_authorized_substituted` pass. |
| E2.6 | Answers use `[[chunk:<id>]]`; unknown IDs stripped & flagged; resolver returns title, URI, page, section, snippet | PASS | `src/kb/generation/citations.py:60-180`. `test_invented_citation_stripped_and_flagged` and `test_snippet_is_exact_word_boundary_prefix` pass; live `kb ask --json` resolved full citation metadata. |
| E2.7 | Refusal path (I2): below-threshold or insufficient context yields `refusal=true`, no citations | PASS | `src/kb/rag/service.py:57-93` and `src/kb/generation/generator.py:95-131,186-201`. `test_empty_context_refuses_without_calling_llm`, `test_low_relevance_refuses_without_calling_llm`, `test_uncited_answer_retried_then_refused`, `test_model_declared_refusal_passes_through`, and live out-of-corpus `kb ask` pass. |
| E2.8 | `kb ask "<q>" --hub <id> --user-tag <t>` prints answer and resolved sources | PASS | `src/kb/cli.py:214-369`. `tests/unit/test_ask_cli.py` passes; live `kb ask "What are the employee benefits?" --hub audit-hub --user-tag hr` printed grounded answer with `[[chunk:...]]` and `[1] Northwind Labs 2026 Benefits Guide ...`. |

## Invariants checked

| ID | Status | Evidence |
|---|---|---|
| I1 | PASS | `git grep -n "ChunkModel" src/` confirms every retrieval/parent query applies `*eligible_chunk_conditions(access)` in SQL (`hybrid.py:108,143`, `parents.py:41`). |
| I2 | PASS | `Generator` (`generator.py:136-205`) strips unverified chunk IDs, retries/refuses uncited answers, and resolves only context-present IDs. |
| I3 | PASS | `render_context` (`context_prompt.py:10-46`) neutralizes `</chunk>`, `</context>`, `<context>` inside chunk text; `prompts/answer.v1.0.0.md:12` marks `<context>` as untrusted data. |
| I4 | PASS | `git grep -n -E "os\.(environ|getenv)" src/` returned 0 matches; `.env.example` covers all 33 `Settings` fields; `hybrid.py:255` & `rerank.py:93` avoid logging raw query/chunk text at INFO. |
| I6 | PASS | `git grep -n -E "import (openai\|anthropic\|cohere\|google\|ollama\|sentence_transformers)" src/` returned 0 matches outside `llm/` (adapters isolated in `src/kb/llm/`). |
| I7 | PASS | `Generator` invokes the model exclusively through `LLMClient` (`src/kb/generation/generator.py:64,84,150`). |
| I9 | PASS | System prompt loaded from `prompts/answer.v1.0.0.md` with semver frontmatter via `src/kb/generation/prompts.py:29-90`; `prompt_version` recorded on `AnswerResult`. |
| I10 | PASS | `mypy --strict src` clean (50 files); `ruff` clean; 95 total tests (60 unit + 35 integration) pass with negative-first security and edge-case coverage. |

## Adversarial probes

| Probe | Result | Notes |
|---|---|---|
| Hub A queries content existing only in Hub B | PASS | `test_hub_isolation_never_returns_other_hub_documents` returned 0 chunks for Hub A. |
| User without `finance` tag asks question only in `finance` doc | PASS | `test_user_lacking_tag_never_receives_tagged_chunk` & `test_access_isolated_rag_query` returned `no_context` refusal with `len(fake_llm.history) == 0`. Live `kb ask --user-tag eng` on `hr`-tagged doc also returned `Refusal (no_context)`. |
| Deactivated / superseded chunk | PASS | `test_inactive_and_superseded_chunks_never_returned` returned 0 chunks. |
| Fewer than k eligible chunks & >k with strict filter | PASS | `test_fewer_than_k_chunks_returns_available_without_error` and `test_iterative_scan_assert_active_and_returns_k_eligible_under_filter` both pass. |
| Model cites ID not in context | PASS | `test_invented_citation_stripped_and_flagged` and Probe B confirmed invented + malformed UUIDs stripped from answer and recorded in `invalid_citations`. |
| Out-of-corpus question | PASS | Live `kb ask "What is the capital of Mars?" --hub audit-hub --user-tag hr` returned `Refusal (model_declined)` with 0 citations. |
| Determinism across repeated runs & shuffled inputs | PASS | `test_hybrid_retrieval_determinism` and `test_context_deterministic_ordering_shuffled` (10 random seeds) pass. |
| Delimiter breakout injection (`</chunk></context><context>...`) | PASS | Probe A confirmed nested tags inside chunk text are escaped (`&lt;/context&gt;`, `&lt;/chunk&gt;`, `&lt;context&gt;`), leaving exactly 1 outer `<context>` and 1 `</context>`. |
| Token budget smaller than every candidate chunk | PASS | Probe D confirmed oversize chunk dropped with `reason="budget"`, `total_tokens=0`, and `top_score` preserved. |
| Markdown code fences + preamble around model JSON | PASS | Probe C confirmed `parse_model_output` extracts valid JSON cleanly. |

## Findings

| # | Severity | Title | Reproduction | Suggested fix | Resolution |
|---|---|---|---|---|---|
| F1 | minor | `docs/PHASES.md` Phase 2 status row still reads `not started` | Inspect `docs/PHASES.md:9` | Update to `audit passed` during `/phase-closeout 2` | open |
| F2 | minor | `REFUSAL_MIN_SCORE` defaults to `0.0` | Inspect `src/kb/config.py:40-43` (prints warning on `kb ask`) | Calibrate threshold during Phase 3 evaluation using the golden set | open |

## Summary

Phase 2 (RAG core with citations) satisfies all exit criteria (E2.1–E2.8) and all relevant invariants (I1–I4, I6, I7, I9, I10). Filtered hybrid retrieval with pgvector iterative scan, protocol-backed reranking with ablation, deterministic budgeted context assembly with access-checked parent expansion, versioned file prompts, citation validation/stripping, and `kb ask` CLI all pass both unit/integration tests (95/95 passing) and live end-to-end verification against Supabase and Ollama. There are zero blocker or major findings; the verdict is **PASS**.
