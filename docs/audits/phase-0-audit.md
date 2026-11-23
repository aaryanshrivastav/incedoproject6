# Phase 0 audit

- Audited commit: `a4a1e754fd0ff8f8deaf89120c8dc6288a68c3b6`
- Date: `2026-11-23`
- Auditor session: fresh, read-only
- Verdict: **PASS**

## Standard checks

| Check | Command | Result |
|---|---|---|
| Tests | `uv run pytest -q` | PASS (15 passed in 2.77s, 98% coverage) |
| Lint | `uv run ruff check .` | PASS (All checks passed!) |
| Format | `uv run ruff format --check .` | PASS (44 files already formatted) |
| Types | `uv run mypy src` | PASS (Success: no issues found in 17 source files; also clean with `--strict`) |

## Exit criteria

| ID | Criterion (short) | Status | Evidence |
|---|---|---|---|
| E0.1 | `docker compose up -d` brings up Postgres (pgvector) and Redis healthy | PASS | `docker compose ps`: `kb-postgres` Up (healthy), `kb-redis` Up (healthy). `docker exec kb-postgres psql -U kb -d kb -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"` -> `vector 0.8.7`. |
| E0.2 | `pytest`, `ruff check`, `ruff format --check`, and `mypy --strict src` all pass, with >=1 real test | PASS | `15 passed in 2.77s` across `tests/unit/test_{cli,client,config,fakes,health}.py`. All linter and type checks pass. |
| E0.3 | Config loads from env via pydantic-settings, fails fast on missing required vars, no secrets, `.env.example` covers all vars | PASS | `src/kb/config.py:9-72` validated by `tests/unit/test_config.py`. `git grep` for hardcoded secrets returned exit code 1 (no matches). `.env.example` matches all fields in `Settings.model_fields`. |
| E0.4 | CI workflow runs lint, type check, and tests on push | PASS | `.github/workflows/ci.yml:1-39` triggers on `push` and `pull_request` to `main`, executing `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy --strict src`, and `uv run pytest -q`. |
| E0.5 | `LLM`, `Embedder`, `Reranker` Protocols exist in `domain/` with deterministic fake implementations used by tests (I6, I7) | PASS | Defined in `src/kb/domain/protocols.py:39-86`. Deterministic implementations in `src/kb/llm/fakes.py:19-139`. Single entrypoint `LLMClient` in `src/kb/llm/client.py:20-80`. Unit tested in `tests/unit/test_fakes.py` and `tests/unit/test_client.py`. |

## Invariants checked

| ID | Status | Evidence |
|---|---|---|
| I4 | PASS | No hardcoded secrets in code; `git grep -n -i secret` showed only config/test keys (`change-me-locally`, `test-secret-key-12345`). Zero reads of `os.environ` or `os.getenv` outside config/tests. |
| I6 | PASS | Protocols defined in `src/kb/domain/protocols.py`; fakes and client in `src/kb/llm/`. No vendor SDK imports (`openai`, `anthropic`, `cohere`, etc.) anywhere in the repository. |
| I7 | PASS | `LLMClient` in `src/kb/llm/client.py:20-80` wraps `LLMProtocol`, instrumenting calls with latency measurement and `_before_call` / `_after_call` extension hooks for guardrails/traces. |
| I9 | PASS | `prompts/` directory created. No prompt text embedded in `src/` (only function argument names). |
| I10 | PASS | `mypy --strict src` passes cleanly. 15 unit tests assert behavior across configuration validation, CLI, health checks, domain protocols, and fakes with 98% line coverage. |

## Adversarial probes

| Probe | Result | Notes |
|---|---|---|
| Missing required env vars (`DATABASE_URL`, `REDIS_URL`, `JWT_SECRET`, `EMBEDDING_DIM`) | PASS | Startup/Settings instantiation fails fast with `ValidationError` specifying the missing field. |
| Empty / whitespace strings for required settings | PASS | Field validator rejects whitespace-only values (`"   "`) with clear `Value error, <field> is required and cannot be empty`. |
| Invalid `EMBEDDING_DIM` (`0`, `-10`, non-integer) | PASS | Rejected by validator: `EMBEDDING_DIM is required and must be greater than 0`. |
| `git grep` for hardcoded secrets / credentials | PASS | Exit code 1; no hardcoded API keys or secrets in repository code. |
| CI failure on test / type breaks | PASS | `uv run pytest` exits with code 1 (or 5 if no tests collected), causing GitHub Actions step failure on error. |
| `FakeEmbedder` edge cases (empty strings, variable text lengths) | PASS | Produces deterministic unit-normalized vectors (norm = 1.0000) for all inputs including empty strings. |
| `FakeReranker` edge cases (empty candidate lists, zero-overlap candidates) | PASS | Returns empty list for empty candidates; returns score 0.0 with original indices for non-matching candidates. |

## Findings

| # | Severity | Title | Reproduction | Suggested fix | Resolution |
|---|---|---|---|---|---|
| F1 | minor | `migrations/` and `ui/` directories from `AGENTS.md` skeleton layout are not present on disk | `Test-Path migrations, ui` returns `False` | Add empty placeholder directories with `.gitkeep` if full skeleton is strictly required ahead of Phase 1 and Phase 4. | open |

## Summary

Phase 0 setup is verified and complete. The repository skeleton, uv configuration, linting, formatting, strict typing, docker compose datastores (PostgreSQL 16 with pgvector extension and Redis 7), domain protocols, deterministic fakes, and unified LLM client are properly implemented and thoroughly tested with 98% coverage. All exit criteria E0.1–E0.5 pass with full evidence.
