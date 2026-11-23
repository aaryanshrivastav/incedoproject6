# Enterprise GenAI Knowledge Assistant

A RAG-based knowledge assistant. It ingests structured and unstructured documents, builds an embedding repository, answers questions through an API with citation grounding, and ships with guardrails, automated evaluation, prompt versioning, and observability.

## How to work in this repo

1. Read this file, then `docs/ARCHITECTURE.md`, then the current phase in `docs/PHASES.md`.
2. Use Planning mode. Produce a plan, list assumptions and open questions, and wait for approval before editing files.
3. One prompt = one scoped change = one commit. Never start work that belongs to a later phase or step, even if it looks easy.
4. Every change ships with tests. Run the verification commands and report their real output. Never say something passes unless you ran it.
5. If a requirement is ambiguous or conflicts with an invariant below, stop and ask. Do not guess.
6. End every task with three lists: what changed, what was verified (with the commands run), and what is unverified or deferred.
7. Do not edit `docs/PHASES.md` exit criteria or the invariants in this file unless the prompt explicitly asks for it.

## Stack (defaults, recorded in `docs/DECISIONS.md`)

| Concern | Choice |
|---|---|
| Language / tooling | Python 3.12, `uv`, `ruff`, `mypy --strict` on `src/`, `pytest` |
| API | FastAPI, pydantic v2, SSE streaming |
| Database | Supabase (PostgreSQL 16 + pgvector), SQLAlchemy 2.0, Alembic, psycopg 3 |
| Keyword search | Postgres full-text (`tsvector`), fused with vector search via RRF |
| Cache / queue | Redis |
| Orchestration | Plain Python with thin adapters (no framework lock-in) |
| PII / guardrails | Presidio, injection heuristics plus classifier hook |
| Evaluation | RAGAS or DeepEval behind our own wrapper |
| Observability | Langfuse (traces), Prometheus-style metrics |
| UI | Streamlit |

## Repo layout

```
src/kb/
  config.py            # pydantic-settings, single source of config
  domain/              # pydantic models, enums, Protocols (LLM, Embedder, Reranker)
  db/                  # SQLAlchemy models, session, repositories
  ingestion/           # parsers/, chunkers/, pipeline.py
  retrieval/           # filters.py, hybrid.py, rerank.py, context.py
  generation/          # prompt loading, generator.py, citations.py
  guardrails/          # input.py, output.py, pii.py
  evaluation/          # metrics, golden set tools, runner
  llm/                 # provider adapters (the ONLY place SDKs are imported)
  observability/       # tracing, metrics
  api/                 # routers, auth, schemas, errors
migrations/            # Alembic
prompts/               # versioned prompt files, never inline strings
tests/{unit,integration,e2e,adversarial}/
eval/{golden,baselines,reports}/
ui/
docs/{ARCHITECTURE.md,PHASES.md,DECISIONS.md,audits/}
```

## Invariants (non-negotiable; audits check these by ID)

- **I1 Access filter in the query.** Hub scope and ACL filters are applied inside the retrieval SQL. Never fetch first and filter after. Identity and ACL tags come from the auth token, never from the request body.
- **I2 Grounded and cited.** Every non-refusal answer carries citations that resolve to chunk IDs that were actually in the context. If evidence is insufficient the system refuses; it does not guess.
- **I3 Untrusted content.** Retrieved text and tool output are data, never instructions. They are delimited in prompts and screened before use.
- **I4 No secrets, no raw PII in code or logs.** Config comes from environment through `config.py`. `.env.example` lists every variable.
- **I5 Idempotent ingestion.** Content hashes make re-ingestion a no-op. Changed documents create a new version and deactivate old chunks.
- **I6 Provider isolation.** LLM, embedding, and reranker access goes through Protocols in `domain/` and adapters in `llm/`. No vendor SDK is imported anywhere else.
- **I7 Single LLM call path.** Every LLM call goes through one wrapper so tracing, cost, retries, and guardrails attach in one place.
- **I8 Hub-ready schema.** `hub_id`, `scope`, and `acl_tags` exist from the first migration, even while only one hub is used.
- **I9 Prompts are files.** Prompts live in `prompts/` with a version header. No prompt text inline in code.
- **I10 Typed and tested.** `mypy --strict` and `ruff` are clean. New behavior has tests, and security-relevant behavior has adversarial tests.

## Definition of done (every prompt)

- Acceptance criteria in the prompt are each demonstrated, not just claimed.
- `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy src` all pass.
- No new TODO without a matching entry in the final report.
- Docs updated if an interface, schema, or command changed.
- Commit message follows `.agents/rules/05-git-and-workflow.md`.

## Commands

```
docker compose up -d              # redis (Postgres datastore hosted entirely on Supabase)
uv run pytest -q                  # tests
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run alembic upgrade head
uv run kb --help                  # project CLI (ingest, ask, eval)
```

## Slash workflows

`/phase-kickoff N`, `/audit-phase N`, `/fix-audit-findings N`, `/phase-closeout N`. Skills in `.agents/skills/` load on demand for audits, ACL filtering, cited generation, evaluation, guardrails, and ingestion.
