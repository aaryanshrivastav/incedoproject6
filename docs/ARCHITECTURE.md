# Architecture

Two paths that meet only at the knowledge repository.

```
ONLINE (latency-sensitive)
Client (Streamlit) -> FastAPI gateway (auth, rate limit, SSE)
  -> Input guard -> Rewrite -> Retrieve (hybrid + rerank) -> Generate (cited) -> Output guard
                                  |
                                  v
                        KNOWLEDGE REPOSITORY (Supabase PostgreSQL 16 + pgvector)
                                  ^
                                  |
OFFLINE (throughput-sensitive)
Sources -> Parse/OCR -> Chunk -> Embed -> Index/upsert (idempotent, versioned)

CROSS-CUTTING: prompt registry, automated evaluation, observability
```

## Components

| Component | Responsibility | Module |
|---|---|---|
| Gateway | Auth (JWT), rate limiting, streaming, error envelope | `api/` |
| Input guard | Injection screening, PII redaction, topic restriction | `guardrails/input.py` |
| Rewrite | Turn follow-ups into standalone queries using conversation history | `retrieval/rewrite.py` |
| Retrieve | Filtered hybrid search (vector + full-text, RRF fusion), rerank | `retrieval/` |
| Context assembly | Token budget, dedupe, ordering, parent expansion | `retrieval/context.py` |
| Generate | Cited answer from versioned prompt, refusal path | `generation/` |
| Output guard | Grounding check, PII scan, policy check | `guardrails/output.py` |
| Ingestion | Parse, chunk, embed, upsert | `ingestion/` |
| Evaluation | Golden set, retrieval and generation metrics, CI gate | `evaluation/` |
| Observability | Traces, metrics, cost, alerts | `observability/` |

## Data model (first migration; see invariants I5, I8)

```
hubs(id, org_id, name, created_at)
documents(id, org_id, source_uri, title, content_hash, version,
          status VARCHAR CHECK ('active','superseded','failed'),
          failure_reason TEXT NULL,
          scope VARCHAR CHECK ('hub','global'), owner_hub_id NULL, acl_tags TEXT[],
          created_at, updated_at)
hub_documents(hub_id, document_id)                       -- many-to-many
chunks(id, document_id, ordinal, parent_chunk_id NULL, text, page NULL, section NULL,
       token_count, content_hash, metadata JSONB, is_active BOOL,
       embedding VECTOR(EMBEDDING_DIM), tsv TSVECTOR)
conversations(id, hub_id, user_id, created_at)
messages(id, conversation_id, role, content, citations JSONB,
         prompt_version, trace_id, created_at)
feedback(id, message_id, rating, comment, created_at)
```

Indexes:
- HNSW on `chunks.embedding` (cosine, `m=16, ef_construction=64`), GIN on `chunks.tsv`
- Btree on `chunks(document_id, is_active)` and `chunks(content_hash)`
- Unique on `chunks(document_id, ordinal)`
- Unique on `documents(org_id, source_uri, version)`
- Partial unique on `documents(org_id, source_uri)` WHERE `status = 'active'`
- Btree on `documents(org_id, status, scope)` and GIN on `documents.acl_tags`
- Btree on `hub_documents(document_id)` and Unique on `hubs(org_id, name)`

## Retrieval access invariant (I1)

Every retrieval query must restrict candidates to chunks that satisfy all of:

1. `chunks.is_active` and `documents.status = 'active'`
2. `documents.org_id = :org_id`
3. `documents.scope = 'global'` OR a row exists in `hub_documents` for `(:hub_id, document_id)`
4. `documents.acl_tags = '{}'` OR `documents.acl_tags && :user_tags`

`:org_id`, `:user_tags` come from the verified token. `:hub_id` comes from the path and is checked against the user's hub access. Query plans may be optimized, but these semantics may not change. With HNSW, filtered queries can return fewer than k rows; verify behavior and use iterative scans or an over-fetch strategy so that k eligible results are returned when they exist.

## Retrieval architecture

Retrieval is structured in `src/kb/retrieval/`:
- `filters.py`: Single source of truth for constructing the SQL access filter (`eligible_chunk_conditions(access: AccessContext)`). Enforces Invariant I1 across all search queries.
- `hybrid.py`: `HybridRetriever` executes vector cosine search (`<=>`) with pgvector iterative scan (`hnsw.iterative_scan = relaxed_order`) and full-text search against `chunks.tsv` with `ts_rank_cd`, fusing candidate lists via Reciprocal Rank Fusion ($k=60$).
- `rerank.py`: `RerankStage` scores candidates via `RerankerProtocol`, sorts by `rerank_score` descending (tie-breaking on `fused_score` then `chunk_id`), and truncates to `top_n`. Can be disabled for ablation (`final_score = fused_score`).
- `parents.py`: `get_eligible_parent_chunks` looks up parent chunks for hierarchical expansion strictly bounded by `eligible_chunk_conditions(access)` (Invariant I1).
- `context.py`: `ContextAssembler` enforces token budgets via `TokenCounter`, deduplicates identical `content_hash` across documents, performs access-controlled parent expansion, and applies deterministic hierarchical document/ordinal ordering without trimming overlap (D-021). Computes semantic `top_score` (D-022).
- `retriever.py`: `Retriever` unifies `HybridRetriever` and `RerankStage` under `retrieve(query, access)`.

## Citation contract (I2)

- Context chunks are presented to the model with stable IDs: `<chunk id="...">...</chunk>`.
- The model cites with `[[chunk:<id>]]`. The parser rejects any ID not present in the context.
- The resolver maps each valid ID to `{document_title, source_uri, page, section, snippet}`.
- Refusal response has `refusal=true` and no citations.

## API surface (target by Phase 4)

```
POST   /v1/hubs
POST   /v1/hubs/{hub_id}/documents          scope: hub | global
PATCH  /v1/documents/{id}                   change scope or hub membership
POST   /v1/hubs/{hub_id}/query              JSON or SSE; optional conversation_id
GET    /v1/hubs/{hub_id}/conversations
POST   /v1/feedback
GET    /health
```

## Multi-hub position

One physical index. Hubs are a logical layer (membership + scope + ACL). The schema supports many hubs from day one; the MVP uses one default hub. Multi-tenant isolation with hard guarantees would require per-tenant namespaces and is out of scope.
