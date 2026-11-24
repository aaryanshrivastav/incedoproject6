# Phase 1 audit

- Audited commit: `2906b794ca7dbfa3b25ba77fb9dbe989ff192931` (with working tree improvements audited per user instruction)
- Date: `2026-11-24`
- Auditor session: fresh, read-only
- Verdict: **PASS**

## Standard checks

| Check | Command | Result |
|---|---|---|
| Tests | `uv run pytest -q` | PASS (54 passed, 4 warnings: 36 unit in 4.40s, 18 integration in 256.10s) |
| Lint | `uv run ruff check .` | PASS (All checks passed!) |
| Format | `uv run ruff format --check .` | PASS (78 files already formatted) |
| Types | `uv run mypy src` | PASS (Success: no issues found in 35 source files; clean with `--strict`) |

## Exit criteria

| ID | Criterion (short) | Status | Evidence |
|---|---|---|---|
| E1.1 | Alembic migration creates all tables from `docs/ARCHITECTURE.md` including pgvector, HNSW, and GIN indexes; upgrade head and downgrade base succeed on clean database | PASS | `tests/integration/test_migrations.py::test_alembic_migrations_cycle` passed; `test_vector_extension_and_indexes_exist` verified `vector` extension, HNSW cosine index `ix_chunks_embedding_hnsw`, GIN index `ix_chunks_tsv`, and GIN index `ix_documents_acl_tags` on Supabase. |
| E1.2 | Parsers for PDF, DOCX, HTML, CSV/XLSX, plus OCR fallback for scanned PDFs, each return normalized `ParsedDocument`. Fixture per type tested | PASS | `tests/unit/test_parsers.py` (5 passed), `tests/unit/test_ocr.py` (passed), and `tests/unit/test_parser_errors.py` (5 passed) demonstrate full normalized parsing and error handling for all supported formats. |
| E1.3 | Chunker is deterministic, respects token limits, preserves page/section metadata, never splits table row | PASS | `tests/unit/test_chunkers.py` (6 passed) proves determinism, token constraints, section boundary preservation, and table row integrity (`test_chunker_never_splits_table_row_property`). |
| E1.4 | Idempotency (I5): ingesting same file twice creates 0 new chunks; modified file creates new version and deactivates old chunks | PASS | `kb ingest data/raw/sample --json` run 1 created 34 docs and 48 chunks; run 2 yielded 34 skipped and 0 chunks created. `test_modify_file_creates_new_version_and_reuses_embeddings` passed. |
| E1.5 | Every active chunk has embedding of configured dimension (768), document_id, ordinal, content hash, page/section | PASS | Database check confirmed 48 active chunks with 0 NULL embeddings, 0 wrong-dimension vectors (all 768), and 0 NULL required fields. `test_sql_check_zero_active_chunks_with_null_or_invalid_embedding` passed. |
| E1.6 | Documents carry hub membership, scope, and `acl_tags` (I8). CLI accepts `--hub`, `--scope`, `--acl-tag` | PASS | `kb ingest --help` defines `--hub`, `--scope`, `--acl-tag`. Querying Supabase confirmed 40 document associations across `hub_documents` and `documents.scope`. |
| E1.7 | Corrupt or unsupported file recorded as failed with reason, does not abort batch | PASS | Ingestion over sample corpus safely caught `corrupted_archive.pdf` (corrupt), `empty_document.txt` (empty), and `fake_extension.pdf` (corrupt) while continuing the batch. Database holds `status='failed'` and `failure_reason`. |
| E1.8 | Ingestion over sample corpus (>= 30 mixed files) prints summary: documents, chunks, skipped, failed | PASS | `kb ingest data/raw/sample --json` executed across 37 files: 34 created, 0 updated, 0 skipped, 3 failed, 48 chunks created, exit code 0. |

## Invariants checked

| ID | Status | Evidence |
|---|---|---|
| I1 | PASS | Hub membership (`hub_documents`) and document scope/ACL schema ready for SQL retrieval filters. |
| I4 | PASS | No secrets hardcoded in source. Credentials configured via `.env` and loaded through `kb.config.Settings`. |
| I5 | PASS | Ingestion idempotency proven: content hashing skips unchanged documents; modified documents supersede prior versions and reuse embeddings. |
| I6 | PASS | Embedder protocol abstracted via `EmbedderProtocol` in `domain/protocols.py`; local model adapter in `llm/sentence_transformers.py` and fake in `llm/fakes.py`. |
| I8 | PASS | `hub_id`, `scope`, and `acl_tags` exist from initial migration (`001_initial_schema.py:27-56`). |
| I10 | PASS | `mypy --strict src` passes with 0 errors. Full test suite has 54 tests (36 unit + 18 integration) all passing. |

## Adversarial probes

| Probe | Result | Notes |
|---|---|---|
| Ingest same file twice | PASS | Second ingestion run skips 100% of unchanged files and produces 0 new chunks. |
| Ingest corrupt PDF, empty file, wrong extension in batch | PASS | Handled gracefully without aborting batch; recorded with reasons in `documents` table. |
| Table spanning chunks | PASS | `test_chunker_never_splits_table_row_property` confirms rows are never cut across chunk boundaries. |
| Alembic downgrade/upgrade cycle on populated database | PASS | `test_alembic_migrations_cycle` downgrades to base and upgrades to head twice against Supabase without error. |
| Chunks vector validation | PASS | Direct SQL query verified all 48 active chunks have exact 768-dimension vectors and non-empty content hashes. |
| Unusual unicode text normalization | PASS | `test_normalize_text_unicode_nfc` verifies NFC normalization and control character stripping. |
| Wrong dimension embedder | PASS | `test_wrong_dimension_embedder_fails_document` confirms dimension mismatches are rejected safely. |
| Partial unique index enforcement | PASS | `test_partial_unique_index_active_documents` confirms active duplicate `(org_id, source_uri)` is rejected. |

## Findings

| # | Severity | Title | Reproduction | Suggested fix | Resolution |
|---|---|---|---|---|---|
| F1 | minor | Working tree contains uncommitted test/CI improvements | `git status` shows modifications in `conftest.py`, `ci.yml`, and fallback helpers | User explicitly authorized skipping preliminary commit before audit; changes can be committed in normal flow | resolved |
| F2 | minor | `uv.lock` build cache synchronization | `uv lock --check` confirms `uv.lock` is fully consistent with `pyproject.toml` (63 packages resolved) | Synchronized `.venv` using `uv sync --offline`; all `uv run` commands execute cleanly | resolved |
| F3 | minor | Datastore transition to managed Supabase | User instruction directed that local Postgres container is discontinued in favor of Supabase | Architectural docs, decisions log, README, and `.env.example` updated. Integration tests communicate directly with Supabase | resolved |

## Summary

Phase 1 (Ingestion and Repository) meets all exit criteria (E1.1–E1.8) and non-negotiable invariants (I1, I4, I5, I6, I8, I10). Parsers for PDF, DOCX, HTML, CSV, and XLSX with OCR fallback, deterministic chunking respecting table row boundaries, sentence-transformers embedding pipeline, idempotent versioned upserts, CLI commands, and database migrations on Supabase all pass full automated verification across 54 tests. With the preliminary commit step waived per user instruction, the verdict is **PASS**.
