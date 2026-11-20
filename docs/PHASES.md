# Phases and exit criteria

This file is the source of truth for audits. Each exit criterion (E#.#) must be demonstrable with a command, test, or inspectable artifact. Status values: `not started`, `in progress`, `built`, `audit failed`, `audit passed`.

| Phase | Name | Build prompts | Status |
|---|---|---|---|
| 0 | Setup | 1 | not started |
| 1 | Ingestion and repository | 3 | not started |
| 2 | RAG core with citations | 3 | not started |
| 3 | Evaluation | 2 | not started |
| 4 | API and UI | 3 | not started |
| 5 | Guardrails | 3 | not started |
| 6 | Observability and prompt optimisation | 2 | not started |
| 7 | Hardening and delivery | 2 | not started |

---

## Phase 0: Setup

Deliverables: skeleton per `AGENTS.md` layout, `pyproject.toml` (uv), `docker-compose.yml`, Makefile, `config.py`, CI workflow, Protocols with fakes.

- E0.1 `docker compose up -d` brings up Postgres (with pgvector extension available) and Redis, both healthy.
- E0.2 `pytest`, `ruff check`, `ruff format --check`, and `mypy --strict src` all pass, with at least one real test.
- E0.3 Config loads from environment via pydantic-settings; a missing required variable fails fast with a clear message; `git grep` finds no secrets; `.env.example` covers every variable read by `config.py`.
- E0.4 CI workflow runs lint, type check, and tests on push.
- E0.5 `LLM`, `Embedder`, `Reranker` Protocols exist in `domain/` with deterministic fake implementations used by tests (I6, I7).

## Phase 1: Ingestion and repository

Prompts: 1.1 schema, migrations, repositories. 1.2 parsers and chunkers. 1.3 embedding, indexing, pipeline CLI.

- E1.1 Alembic migration creates all tables from `docs/ARCHITECTURE.md` including pgvector extension, HNSW and GIN indexes; both `upgrade head` and `downgrade base` succeed on a clean database.
- E1.2 Parsers for PDF, DOCX, HTML, and CSV/XLSX, plus OCR fallback for scanned PDFs, each return a normalized `ParsedDocument` with page/section info and table handling. A fixture per type is tested.
- E1.3 Chunker is deterministic, respects token limits, preserves page and section metadata, and never splits a table row.
- E1.4 Idempotency (I5): ingesting the same file twice creates zero new chunks; a modified file creates a new document version and deactivates old chunks.
- E1.5 Every active chunk has an embedding of the configured dimension, a document_id, ordinal, content hash, and page/section where the source has them.
- E1.6 Documents carry hub membership, scope, and `acl_tags` (I8). CLI accepts `--hub`, `--scope`, `--acl-tag`.
- E1.7 A corrupt or unsupported file is recorded as failed with a reason and does not abort the batch.
- E1.8 Running ingestion over the sample corpus (>= 30 mixed files) prints a summary: documents, chunks, skipped, failed.

## Phase 2: RAG core with citations

Prompts: 2.1 filtered hybrid retrieval. 2.2 rerank and context assembly. 2.3 cited generation and refusal.

- E2.1 Hybrid retrieval fuses vector and full-text ranks (RRF) and returns top-k with scores.
- E2.2 Access invariant I1 is enforced in SQL. Tests prove: a user lacking an ACL tag never receives a restricted chunk; a query on hub A never returns hub-only docs of hub B; global docs are visible from every hub; inactive chunks are never returned.
- E2.3 When at least k eligible chunks exist, filtered queries return k results (ANN filtering check).
- E2.4 Reranker sits behind the Protocol and can be toggled for ablation.
- E2.5 Context assembly respects a token budget, dedupes overlapping chunks, orders deterministically, and can expand to parent chunks.
- E2.6 Answers use `[[chunk:<id>]]` citations. Cited IDs not in the context are stripped and flagged. Resolver returns title, URI, page, section, snippet.
- E2.7 Refusal path (I2): below-threshold retrieval or insufficient context yields `refusal=true`, no citations. Tested with out-of-corpus questions.
- E2.8 `kb ask "<q>" --hub <id> --user-tag <t>` prints the answer and resolved sources.

## Phase 3: Evaluation

Prompts: 3.1 golden set tooling. 3.2 metrics runner and CI gate.

- E3.1 Golden set JSONL schema is validated by code. At least 50 items, each with question, reference answer, source document/chunk IDs, and type (factual, multi-hop, table, unanswerable, access-restricted). At least 10% unanswerable.
- E3.2 Synthetic generation tool exists; items have `reviewed` flag; only reviewed items count toward the gate.
- E3.3 Retrieval metrics (recall@k, precision@k, MRR, nDCG) are implemented and unit-tested against hand-computed fixtures.
- E3.4 Generation metrics (faithfulness, answer relevance, context precision, context recall) run through our wrapper; judge model is configurable and recorded in the report.
- E3.5 `kb eval run` writes JSON and markdown reports containing a config hash; a baseline is stored in `eval/baselines/`.
- E3.6 CI gate fails when a gated metric drops beyond threshold versus baseline, demonstrated by a test using a deliberately degraded config.

## Phase 4: API and UI

Prompts: 4.1 endpoints and streaming. 4.2 auth, rate limits, errors. 4.3 UI.

- E4.1 OpenAPI lists every endpoint in the API surface of `docs/ARCHITECTURE.md`.
- E4.2 Query response includes answer, citations[], conversation_id, usage, prompt_version, refusal flag.
- E4.3 SSE streams tokens and ends with a final event carrying resolved citations.
- E4.4 JWT required on all `/v1` routes. User identity, org, and ACL tags come from the token only (I1).
- E4.5 Rate limiting returns 429 with `Retry-After`.
- E4.6 Uniform error envelope; no stack traces or internal paths leak.
- E4.7 Follow-up questions use the rewrite step with conversation history; a test covers an elliptical follow-up.
- E4.8 UI renders clickable citations that show the source snippet, and thumbs up/down feedback is stored.
- E4.9 Integration tests cover every endpoint including unauthorized and cross-hub attempts.

## Phase 5: Guardrails

Prompts: 5.1 input guard. 5.2 output guard. 5.3 red-team suite.

- E5.1 Input guard performs injection screening and PII redaction before the LLM and before logging; blocked requests get a safe message.
- E5.2 Retrieved chunks are delimited as untrusted data in prompts (I3); poisoned-document tests pass.
- E5.3 Output guard checks grounding of claims against context, scans for PII, and enforces topic policy.
- E5.4 Red-team suite of at least 40 cases across: direct injection, indirect injection via documents, system-prompt exfiltration, PII leakage, out-of-scope requests, cross-hub and ACL probes. Pass rate: 100% for access probes, at least 90% for the rest. Runs in CI.
- E5.5 Guardrail decisions are logged with reason codes and without raw PII.
- E5.6 Latency overhead of guardrails is measured and reported.

## Phase 6: Observability and prompt optimisation

Prompts: 6.1 tracing and metrics. 6.2 prompt registry and A/B.

- E6.1 Every request yields a trace with: redacted query, rewrite, retrieved chunk IDs and scores, prompt version, model, tokens, cost, per-stage latency, guardrail decisions, hub_id (I7).
- E6.2 Metrics exported: latency p50/p95, error rate, cost per request, refusal rate, feedback score. Dashboard definition is committed.
- E6.3 Alert rules defined for latency, error rate, cost anomaly, and faithfulness drop.
- E6.4 Prompts in `prompts/` carry a semver header and changelog and are loaded through a registry; traces record the version (I9).
- E6.5 `kb prompts compare A B` runs both versions on the golden set and reports metric deltas; promotion requires no gated regression.
- E6.6 Online sampling scores a configurable fraction of live traffic with the faithfulness judge and stores results.
- E6.7 Optional: DSPy experiment documented. Not required.

## Phase 7: Hardening and delivery

Prompts: 7.1 resilience, caching, tests. 7.2 CI/CD and documentation.

- E7.1 LLM calls have timeouts, retries with backoff, and a fallback path; fault-injection tests cover each.
- E7.2 Embedding and query caches invalidate on document version change.
- E7.3 Load test results documented with target concurrency and p95 latency.
- E7.4 Coverage at least 80%; an end-to-end test covers ingest, ask, and cite.
- E7.5 CI runs lint, types, tests, eval gate, red-team suite, and container build.
- E7.6 README quickstart works from a fresh clone, verified in a clean environment.
- E7.7 Docs complete: architecture, eval report, known limitations, security notes.
