# Enterprise GenAI Knowledge Assistant

A production-grade, retrieval-augmented generation (RAG) knowledge assistant. It ingests structured and unstructured documents, builds an embedding repository in PostgreSQL with pgvector, answers queries through a FastAPI gateway with verified citation grounding, and ships with guardrails, automated evaluation, and observability.

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
                (Postgres 16 + pgvector)
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
- **Docker & Docker Compose** (for PostgreSQL 16 + pgvector and Redis)

---

## Quickstart

### 1. Environment Setup
Copy the environment template and adjust configuration values as needed:
```powershell
copy .env.example .env
```

### 2. Start Infrastructure
Launch PostgreSQL (with `pgvector` extension) and Redis in detached mode:
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
| **Start API Gateway** | `uv run uvicorn kb.api.app:app --reload --port 8000` | - |

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
