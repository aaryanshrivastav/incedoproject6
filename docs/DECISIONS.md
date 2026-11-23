# Decisions log

Status: `proposed` until the owner confirms. Agents must treat `proposed` items as defaults and flag them in reports, not silently change them.

| ID | Decision | Phase | Status | Notes |
|---|---|---|---|---|
| D-001 | Postgres + pgvector as the single store (vectors, full-text, metadata, ACLs) | Phase 0 | accepted | Validated in Phase 0 via docker-compose and pgvector extension ($0 cost) |
| D-002 | Plain Python orchestration with thin adapters, no LangChain/LlamaIndex | Phase 0 | accepted | Established via domain protocols and unified LLM client ($0 cost) |
| D-003 | Hybrid search via RRF fusion of vector and `tsvector` ranks | Phase 2 | proposed | Parameter constants locked during retrieval prompt 2.1 ($0 cost) |
| D-004 | Hub-ready schema from migration 1 (I8) | Phase 1 | accepted | In-engine SQL schema with multi-hub and ACL support |
| D-005 | Citation format `[[chunk:<id>]]` with server-side validation | Phase 2 | accepted | Parsing and context validation contract |
| D-006 | LLM provider and model | Phase 2 | accepted | **100% Local / $0 budget**: Local LLM via Ollama (e.g. `llama3.2:3b` / `qwen2.5:3b`) |
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

## Log

- 2026-11-20: Mapped decisions to execution phases so work proceeds on an as-we-go basis. Phase 0 requires only config schemas and fake protocols (I6, I7, E0.5), with no external model API blockers.
- 2026-11-23: Phase 0 closeout. Accepted D-001 (Postgres + pgvector) and D-002 (Plain Python orchestration) following successful infrastructure setup, implementation, and passing audit.
- 2026-11-23: Constraint established: $0 budget for entire project. Adopted 100% local, open-source stack for D-006 (Ollama local LLM), D-008 (local cross-encoder), D-009 (100% local with WSL tesseract OCR), D-011 (local judge), and D-012 (synthetic/public corpus). D-007 (embedding model & dimension) currently in discussion.
- 2026-11-23: Phase 1.1 implementation. Accepted D-007 (EMBEDDING_DIM = 768), D-013 (schema constraints with partial unique index), and D-014 (DEFAULT_ORG_ID constant).
- 2026-11-23: Phase 1.2 implementation. Accepted D-015 (approximate token counter Protocol) and D-016 (permissive document parsing stack).
