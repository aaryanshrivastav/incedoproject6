# RAG conventions

- Chunk IDs are stable UUIDs. Citations use `[[chunk:<id>]]` and are validated against the context set (invariant I2).
- Prompts live in `prompts/<name>.vX.Y.Z.md` with a header (name, version, purpose, changelog). Code loads them through the registry; no inline prompt strings (I9).
- Retrieval returns structured results (`chunk_id`, `document_id`, `score`, `source`, `page`, `section`). Do not pass bare strings between stages.
- Fusion uses RRF with documented constants. Any change to retrieval parameters (k, thresholds, chunk size) must be justified by eval results and recorded in `docs/DECISIONS.md`.
- Refusal is a first-class outcome with its own response shape, not an error.
- Context assembly is deterministic: same inputs, same ordering and truncation.
- Every pipeline stage takes and returns typed objects so it can be traced and tested alone.
