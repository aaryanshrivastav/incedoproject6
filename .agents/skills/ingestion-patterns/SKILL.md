---
name: ingestion-patterns
description: Patterns for document parsing, OCR, chunking, embedding, and idempotent versioned upserts into the knowledge repository. Use whenever working on ingestion, parsers for PDF, DOCX, HTML, CSV, or Excel, OCR, chunking strategy, embeddings, indexing, deduplication, document versioning, or the ingest CLI.
---

# Ingestion patterns

## Pipeline

`source -> parse -> normalize -> chunk -> embed -> upsert`. Each stage takes and returns typed objects.

- **Parse:** one parser per format behind a common interface returning `ParsedDocument` (title, blocks with type, page, section, table structure). Scanned PDFs fall back to OCR when extracted text is empty or too sparse per page.
- **Normalize:** fix encoding, collapse whitespace, keep structural markers (headings, table boundaries).
- **Chunk:** recursive splitting on structure first (headings, paragraphs), then sentences, with token limits and small overlap. Never split a table row. Keep page and section on each chunk. Optionally create parent chunks for context expansion.
- **Embed:** batch calls through the `Embedder` Protocol, cache by content hash, verify output dimension equals `EMBEDDING_DIM`.
- **Upsert:** see below.

## Idempotency and versioning (I5)

- Document `content_hash` is computed from normalized content. Same hash for the same source means skip.
- Changed content creates a new document version. Old chunks are marked `is_active=false` in the same transaction that activates the new ones, so queries never see a mix.
- Chunk `content_hash` lets unchanged chunks reuse existing embeddings.
- Ingestion of a batch is per-document transactional. One failure does not roll back the others.

## Failure handling

Record failed documents with a reason (`unsupported`, `corrupt`, `encrypted`, `empty`, `parse_error`). Continue the batch. Print a summary: processed, skipped (unchanged), failed (with reasons), chunks created.

## Metadata (I8)

Every document gets hub membership, scope (`hub` or `global`), `acl_tags`, `source_uri`, and title. The CLI exposes `--hub`, `--scope`, and `--acl-tag`.

## Structured data

CSV and Excel rows are serialized into readable text with column names preserved, grouped into chunks without splitting a row, with the sheet or table name as section. Keep the raw table available for a future text-to-SQL path.

## Tests

Fixtures per format; double-ingest is a no-op; modified file versions correctly; corrupt file fails alone; embeddings have the right dimension; table rows never split.
