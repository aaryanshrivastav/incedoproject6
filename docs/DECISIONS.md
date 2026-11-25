# Decisions log

Status: `proposed` until the owner confirms. Agents must treat `proposed` items as defaults and flag them in reports, not silently change them.

| ID | Decision | Phase | Status | Notes |
|---|---|---|---|---|
| D-001 | Supabase (PostgreSQL 16 + pgvector) as the single store (vectors, full-text, metadata, ACLs) | Phase 0-1 | accepted | Cloud-hosted Supabase PostgreSQL 16 with pgvector extension enabled. Local Postgres container discontinued in favor of managed Supabase ($0 free tier) |
| D-002 | Plain Python orchestration with thin adapters, no LangChain/LlamaIndex | Phase 0 | accepted | Established via domain protocols and unified LLM client ($0 cost) |
| D-003 | Hybrid search via RRF fusion of vector and `tsvector` ranks | Phase 2 | accepted | Reciprocal Rank Fusion ($k=60$) fused vector and full-text ranks into deterministic candidate pools ($0 cost) |
| D-004 | Hub-ready schema from migration 1 (I8) | Phase 1 | accepted | In-engine SQL schema with multi-hub and ACL support |
| D-005 | Citation format `[[chunk:<id>]]` with server-side validation | Phase 2 | accepted | Parsing and context validation contract |
| D-006 | LLM provider and model | Phase 2 | accepted | **100% Local / $0 budget**: Local LLM via Ollama (`llama3.2:3b` primary, `qwen2.5:3b` secondary). Models stored on drive D: (`OLLAMA_MODELS=D:\ollama\models`) leveraging 262 GB free space. Tuned for 12-core CPU/16GB RAM. |
| D-007 | Embedding model and dimension | Phase 1 | accepted | **$0 budget local package**: `BAAI/bge-base-en-v1.5` (`sentence-transformers`), `EMBEDDING_DIM = 768` |
| D-008 | Reranker (cross-encoder, local or hosted) | Phase 2 | accepted | **100% Local / $0 budget**: Local cross-encoder (`ms-marco-MiniLM-L-6-v2`) or token overlap |
| D-009 | Hosted vs fully local models | Phase 1-2 | accepted | **100% Fully Local / $0 budget**: OCR via `wsl tesseract` (v5.3.4), local embeddings and generation |
| D-010 | Auth method (JWT issued by whom) | Phase 4 | proposed | Dev-issued HMAC-SHA256 JWT for MVP, pluggable verifier ($0 cost) |
| D-011 | Evaluation judge model differs from the generation model | Phase 3 | accepted | **$0 budget**: Secondary local model via Ollama / heuristic offline evaluators |
| D-012 | Sample corpus location and size (>= 30 files, mixed types) | Phase 1 | accepted | `data/sample_corpus/` containing >= 30 open/synthetic files ($0 cost) |
| D-013 | Schema constraints and enums as VARCHAR with CHECK constraints | Phase 1 | accepted | Enums stored as VARCHAR with CHECK constraints (not PG enums) for easier migration; partial UNIQUE index for active documents |
| D-014 | Default Organization UUID for single-org MVP | Phase 1 | accepted | Fixed UUID constant `00000000-0000-0000-0000-000000000001` in Settings for single-org phase |
| D-015 | Approximate token counter, tokenizer swappable | Phase 1 | accepted | Ingest pipeline uses `ApproxTokenCounter` (`ceil(chars / 4)`) behind `TokenCounter` Protocol, swappable for model-specific tokenizers without framework lock-in |
| D-016 | Parsing libraries and permissive licenses | Phase 1 | accepted | Permissive licenses only: `pdfplumber` (MIT), `pytesseract` (Apache 2.0), `python-docx` (MIT), `beautifulsoup4` (MIT), `lxml` (BSD), `openpyxl` (MIT), dev `reportlab` (BSD), `pillow` (HPND). Strictly no `pandas` or `tiktoken` |
| D-017 | Embedding reuse by chunk hash and model | Phase 1 | accepted | Ingestion queries existing chunk embeddings matching `content_hash` and `metadata["embedding_model"]`, avoiding redundant inference during document updates |
| D-018 | Source URI identity rules | Phase 1 | accepted | `source_uri` is normalized relative path when `--source-root` is provided, otherwise canonical `file://` URI. Ensures stable document versioning across runs |
| D-019 | RRF constants and tie-break rules | Phase 2 | accepted | Reciprocal Rank Fusion uses constant $k=60$ ($score = \sum 1 / (60 + rank)$ with 1-based ranks). Deterministic tie-break: descending `fused_score`, ascending `best_rank`, ascending `chunk_id` string |
| D-020 | pgvector iterative scan and minimum version | Phase 2 | accepted | Vector retrieval requires pgvector >= 0.8.0 for `hnsw.iterative_scan = relaxed_order` to prevent under-fetching under restrictive access filters |
| D-021 | Chunk overlap preserved untrimmed in context assembly | Phase 2 | accepted | Adjacent chunks keep their natural overlap text; do NOT trim overlap in context assembly because snippets must exactly match stored chunk text for citation verification |
| D-022 | Relevance top_score definition and RRF separation | Phase 2 | accepted | `top_score` is the best semantic score available (highest rerank_score if reranking ran, else highest vector_similarity; None if neither). RRF scores cannot serve as absolute relevance thresholds because their magnitude varies with candidate pool sizes and relative rank distributions |
| D-023 | Output JSON contract and retry policy | Phase 2 | accepted | Generation model must output strict JSON schema. Malformed JSON triggers exactly 1 retry before refusing (`invalid_model_output`). Non-refusal answers with 0 valid citations trigger 1 retry or immediate refusal per `ON_UNCITED`, preventing hallucinations from returning as confident answers |
| D-024 | Refusal reasons and short-circuit pre-checks | Phase 2 | accepted | Pipeline short-circuits WITHOUT calling the LLM when context is empty (`no_context`) or top score is below `REFUSAL_MIN_SCORE` (`low_relevance`). Model-reported insufficiency yields `model_declined`. All refusals carry `citations=[]` and human-readable reasons |
| D-025 | Golden items reference `source_uris` + `evidence_spans` instead of database UUIDs | Phase 3 | accepted | Document and chunk UUIDs are assigned per database/ingestion and change across environments or re-chunking. Golden items store relative `source_uris` and verbatim `evidence_spans` inside a single chunk, resolved dynamically at evaluation runtime |
| D-026 | Golden composition targets and review attestation model | Phase 3 | accepted | Target candidate set >= 64 items across all 5 types (`factual >= 24`, `table >= 12`, `multi_hop >= 10`, `unanswerable >= 10`, `access_restricted >= 8` with controls). Only `reviewed=True` items are gated; `manifest` provenance may be bulk-accepted with `--yes` attestation after `check-corpus`, whereas `llm` provenance always requires individual human review |
| D-027 | Native `JudgeMetric` Protocol implementation instead of heavyweight eval frameworks | Phase 3 | accepted | Implemented 4 RAGAS-style metrics natively behind `JudgeMetric` using `LLMClient` (I7) and versioned prompt files (I9); third-party adapters (RAGAS/DeepEval) remain deferred behind the same Protocol |
| D-028 | Independent pure-Python access oracle for leak detection | Phase 3 | accepted | `src/kb/evaluation/access_oracle.py` evaluates org, status, scope, hub membership, and ACL tag overlap in pure Python without importing `kb.retrieval.filters`, catching any SQL filter regressions |
| D-029 | Config-driven CI gate semantics and provisional thresholds | Phase 3 | accepted | Thresholds live in `eval/thresholds.json` (`max_drop` / `max_rise` plus zero-tolerance hard gates for `access_leaks` and `answer_leaks`, and >10% judge failure invalidation); provisional until the first real calibrated baseline |
| D-030 | Threshold calibration recommends without mutating config | Phase 3 | accepted | `kb eval calibrate` sweeps observed `top_score` values, writes `calibration.md` in the report directory, and prints a caveat on rerank vs vector score semantics without altering `.env` or `config.py` |

## Log

- 2026-11-20: Mapped decisions to execution phases so work proceeds on an as-we-go basis. Phase 0 requires only config schemas and fake protocols (I6, I7, E0.5), with no external model API blockers.
- 2026-11-23: Phase 0 closeout. Accepted D-001 (Postgres + pgvector) and D-002 (Plain Python orchestration) following successful infrastructure setup, implementation, and passing audit.
- 2026-11-23: Constraint established: $0 budget for entire project. Adopted 100% local, open-source stack for D-006 (Ollama local LLM), D-008 (local cross-encoder), D-009 (100% local with WSL tesseract OCR), D-011 (local judge), and D-012 (synthetic/public corpus). D-007 (embedding model & dimension) currently in discussion.
- 2026-11-23: Phase 1.1 implementation. Accepted D-007 (EMBEDDING_DIM = 768), D-013 (schema constraints with partial unique index), and D-014 (DEFAULT_ORG_ID constant).
- 2026-11-23: Phase 1.2 implementation. Accepted D-015 (approximate token counter Protocol) and D-016 (permissive document parsing stack).
- 2026-11-23: Phase 1.3 implementation. Accepted D-017 (chunk hash embedding reuse) and D-018 (source_uri identity rules).
- 2026-11-23: Infrastructure decision update. Migrated datastore from local Docker Postgres to Supabase cloud-hosted PostgreSQL 16 with pgvector. All tables, extensions, and migrations run directly against Supabase. Local Docker Postgres is discontinued.
- 2026-11-24: Hardware-tailored inference plan: Selected `llama3.2:3b` as primary generation model for 12-core CPU / 16GB RAM, with models directed to drive D: (`OLLAMA_MODELS=D:\ollama\models`, 262 GB free).
- 2026-11-24: Phase 2.1 implementation. Accepted D-019 (RRF parameters and tie-breaker) and D-020 (pgvector 0.8.0+ iterative scan).
- 2026-11-24: Phase 2.2 implementation. Accepted D-021 (overlap kept untrimmed) and D-022 (top_score definition and rejection of RRF for relevance thresholds).
- 2026-11-24: Phase 2.3 implementation. Accepted D-023 (output JSON contract and retry policy) and D-024 (refusal reasons and short-circuit pre-checks). Real Ollama adapter implemented in `src/kb/llm/ollama.py`.
- 2026-11-25: Phase 2 closeout. Accepted D-003 (Hybrid search via RRF fusion). Audit passed with zero blocker or major findings (95/95 tests passing, live Ollama inference verified).
