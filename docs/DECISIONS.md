# Decisions log

Status: `proposed` until the owner confirms. Agents must treat `proposed` items as defaults and flag them in reports, not silently change them.

| ID | Decision | Phase | Status | Notes |
|---|---|---|---|---|
| D-001 | Postgres + pgvector as the single store (vectors, full-text, metadata, ACLs) | Phase 0 | proposed | Swap-out path: retrieval behind a repository interface |
| D-002 | Plain Python orchestration with thin adapters, no LangChain/LlamaIndex | Phase 0 | proposed | Keeps control of filters, citations, tracing |
| D-003 | Hybrid search via RRF fusion of vector and `tsvector` ranks | Phase 2 | proposed | Parameter constants locked during retrieval prompt 2.1 |
| D-004 | Hub-ready schema from migration 1 (I8) | Phase 1 | accepted | |
| D-005 | Citation format `[[chunk:<id>]]` with server-side validation | Phase 2 | accepted | |
| D-006 | LLM provider and model | Phase 2 | **needs input** | Set `LLM_PROVIDER`, `LLM_MODEL`; Phase 0 uses `FakeLLM` |
| D-007 | Embedding model and dimension | Phase 1 | **needs input** | Set `EMBEDDING_MODEL`, `EMBEDDING_DIM`; needed for migration 1 |
| D-008 | Reranker (cross-encoder, local or hosted) | Phase 2 | **needs input** | Toggled behind Protocol in prompt 2.2; Phase 0 uses `FakeReranker` |
| D-009 | Hosted vs fully local models | Phase 1-2 | **needs input** | Affects OCR (Phase 1), embedder (Phase 1), judge model (Phase 3) |
| D-010 | Auth method (JWT issued by whom) | Phase 4 | proposed | Dev-issued JWT for MVP, pluggable verifier |
| D-011 | Evaluation judge model differs from the generation model | Phase 3 | proposed | Reduces self-preference bias |
| D-012 | Sample corpus location and size (>= 30 files, mixed types) | Phase 1 | **needs input** | Needed for E1.8 batch ingestion test |

## Log

- 2026-11-20: Mapped decisions to execution phases so work proceeds on an as-we-go basis. Phase 0 requires only config schemas and fake protocols (I6, I7, E0.5), with no external model API blockers.
