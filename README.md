# Enterprise GenAI Knowledge Assistant

A production-grade, retrieval-augmented generation (RAG) knowledge assistant. It ingests structured and unstructured documents, builds an embedding repository in Supabase (PostgreSQL with pgvector), answers queries through a FastAPI gateway with verified citation grounding, and ships with guardrails, automated evaluation, and observability.

---

## Architecture & Data Flow

Two decoupled pipelines that meet only at the knowledge repository:

```
ONLINE (latency-sensitive query path)
User / Client (Streamlit UI) -> FastAPI Gateway (JWT auth, rate limits, SSE)
  -> Input Guard (PII redaction, injection check)
  -> Query Rewrite (history disambiguation)
  -> Retrieval (filtered hybrid search: vector + full-text with RRF fusion)
  -> Reranker -> Context Assembly (token budget, parent expansion)
  -> Generation (cited answer via versioned prompt, refusal path)
  -> Output Guard (grounding verification, PII scan)
                             |
                             v
                  KNOWLEDGE REPOSITORY
                (Supabase Postgres 16 + pgvector)
                             ^
                             |
OFFLINE (throughput-sensitive ingestion path)
Sources -> Parse / OCR -> Chunk (table-aware) -> Embed -> Index & Upsert (idempotent, versioned)
```

---

## Key Invariants

- **I1 - Access Filter in SQL**: Hub scope and ACL filters are strictly applied inside database queries. User identity and ACL tags are extracted solely from verified JWT tokens.
- **I2 - Grounded & Cited**: Every non-refusal answer cites valid chunk IDs (`[[chunk:<id>]]`). Insufficient context returns a refusal (`refusal=true`).
- **I3 - Untrusted Content**: Retrieved chunks and external inputs are treated as untrusted data and strictly delimited.
- **I4 - Zero Secrets in Code**: Configuration is loaded through `kb.config.Settings` from `.env` via `pydantic-settings`.
- **I5 - Idempotent Ingestion**: Content hashing prevents duplicate chunks; updates create new document versions.
- **I6 & I7 - Single LLM Call Path & Provider Isolation**: All model access is abstracted behind `src/kb/domain/protocols.py` and routed through `src/kb/llm/client.py`.
- **I8 - Hub-Ready Schema**: Multi-hub membership, scope, and ACL tags exist from the first schema migration.
- **I9 - Prompts as Files**: Prompts live under `prompts/` with semantic versioning headers.
- **I10 - Typed and Tested**: Strict typing (`mypy --strict`) and high test coverage.

---

## Prerequisites

- **Python 3.12** (managed automatically via `uv`)
- **[uv](https://docs.astral.sh/uv/)** (package and virtualenv manager)
- **Supabase PostgreSQL 16 + pgvector** (cloud datastore configured via `DATABASE_URL` in `.env`)
- **Docker & Docker Compose** (for Redis local queue/cache)
- **Tesseract OCR binary** (required for OCR fallback on scanned PDFs)
  - Ubuntu / Debian: `sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-eng`
  - macOS: `brew install tesseract`
  - Windows: `winget install UB-Mannheim.TesseractOCR` (or via WSL: `sudo apt-get install -y tesseract-ocr`)

---

## Quickstart

### 1. Environment Setup
Copy the environment template and adjust configuration values as needed:
```powershell
copy .env.example .env
```

### 2. Start Infrastructure
Launch Redis (and optional local services) in detached mode (Postgres with `pgvector` is hosted on Supabase):
```powershell
docker compose up -d
docker compose ps
```

### 3. Install Dependencies
Install all project and development dependencies in an isolated virtual environment (`.venv`):
```powershell
uv sync
```

---

## Developer Commands

All core tasks can be run using `uv run` or via `make` targets:

| Task | Command | Makefile Target |
|---|---|---|
| **Run Unit Tests** | `uv run pytest -v` | `make test` |
| **Run Linter** | `uv run ruff check .` | `make lint` |
| **Format Code** | `uv run ruff format .` | `make format` |
| **Format Check** | `uv run ruff format --check .` | `make format-check` |
| **Type Check** | `uv run mypy --strict src` | `make typecheck` |
| **Run All Checks** | `make check` | `make check` |
| **CLI Version** | `uv run kb version` | - |
| **CLI Help** | `uv run kb --help` | - |
| **CLI Ask Query** | `uv run kb ask "What is the policy?" --hub default` | - |
| **Start API Gateway** | `uv run uvicorn kb.api.app:app --reload --port 8000` | - |

---

## Asking Questions with `kb ask`

The `kb ask` CLI provides a trusted local development interface for testing end-to-end RAG answer generation, citation grounding, and refusal gating over ingested documents:

```powershell
# Basic query against the default hub
uv run kb ask "What are the database connection guidelines?" --hub default

# Query with access tags (simulating user ACL permissions)
uv run kb ask "Summarize Q3 financial results" --hub default --user-tag finance

# Inspect the retrieved and assembled context bundle along with answers
uv run kb ask "How does iterative scan work?" --hub default --show-context

# Output the complete AnswerResult structure as JSON
uv run kb ask "Explain parent expansion" --hub default --json
```

---

---

## Evaluation

The Phase 3 evaluation harness measures retrieval (`recall@k`, `precision@k`, `mrr`, `ndcg@k`), behavior (`refusal_accuracy`, `citation_validity_rate`, `access_leaks`, `answer_leaks`), and judge-based generation quality (`faithfulness`, `answer_relevance`, `context_precision`, `context_recall`).

```powershell
# 1. Setup evaluation corpus (idempotent)
uv run kb eval setup --plan eval/golden/corpus_plan.json --root data/raw/sample

# 2. Run evaluation over reviewed golden items
uv run kb eval run --golden eval/golden/golden.jsonl --out eval/reports

# 3. Promote a baseline (deliberate, separate commit)
uv run kb eval baseline set eval/reports/<timestamp>-<hash> --name default

# 4. Run regression gate against eval/thresholds.json
uv run kb eval gate --report eval/reports/<timestamp>-<hash> --baseline eval/baselines/default.json --thresholds eval/thresholds.json

# 5. Calibrate REFUSAL_MIN_SCORE without modifying config
uv run kb eval calibrate --report eval/reports/<timestamp>-<hash>
```

### CI Evaluation Secrets (for `workflow_dispatch` in `.github/workflows/eval.yml`)
- `EVAL_DATABASE_URL`, `EVAL_REDIS_URL`, `EVAL_JWT_SECRET`
- `EVAL_EMBEDDING_PROVIDER`, `EVAL_EMBEDDING_MODEL`
- `EVAL_LLM_PROVIDER`, `EVAL_LLM_MODEL`, `EVAL_LLM_API_KEY`
- `EVAL_RERANKER_PROVIDER`
- `EVAL_JUDGE_PROVIDER`, `EVAL_JUDGE_MODEL`, `EVAL_JUDGE_API_KEY`

> Note: Always-on PR/push CI runs `pytest -m eval_fixture` with local fake providers and requires **no secrets**. Updating `eval/baselines/*.json` must be a deliberate, separate commit.

---

## Git Workflow & Conventions

Follow the rules in `.agents/rules/05-git-and-workflow.md`:

### Commit Convention
Commits follow the format: `<phase>.<step>: <imperative summary>`
- Examples:
  - `0.1: repository skeleton, tooling, infra, and test doubles`
  - `0.close: phase 0 complete`
  - `1.1: alembic migrations, pgvector extension, and repositories`

### Common Git Commands
```powershell
# Check working tree status
git status

# Stage changes
git add <files>

# Commit with conventional message
git commit -m "<phase>.<step>: <message>"

# Push to origin
git push origin main
```

---

## Phased Build Status

Tracked in [docs/PHASES.md](docs/PHASES.md):

- [x] **Phase 0: Setup** (`audit passed`) — Skeleton, tooling (`uv`), Postgres/pgvector & Redis, protocols, fakes, CI.
- [ ] **Phase 1: Ingestion & Repository** (`not started`) — Migrations, parsers (PDF, DOCX, HTML, CSV/XLSX), chunking, embeddings pipeline.
- [ ] **Phase 2: RAG Core with Citations** (`not started`) — Filtered hybrid search (RRF), reranking, context assembly, cited generation.
- [ ] **Phase 3: Evaluation** (`not started`) — Golden set tooling, retrieval and generation metrics, CI gate.
- [ ] **Phase 4: API & UI** (`not started`) — FastAPI endpoints (REST + SSE streaming), JWT auth, Streamlit UI.
- [ ] **Phase 5: Guardrails** (`not started`) — Input/output guards, Presidio PII, adversarial red-team suite.
- [ ] **Phase 6: Observability** (`not started`) — Langfuse tracing, Prometheus metrics, prompt registry.
- [ ] **Phase 7: Hardening & Delivery** (`not started`) — Resilience, caching, load testing, docs.
