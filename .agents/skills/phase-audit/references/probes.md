# Adversarial probes by phase

Use these as a starting list. Add your own.

## Phase 0
- Remove a required env var: does startup fail with a clear message?
- `git grep -nE "(api[_-]?key|secret|password)\s*=\s*['\"][^'\"]+"` for hardcoded secrets.
- Does CI fail when a test fails or when types are broken (introduce a temporary error in a scratch branch, do not commit)?

## Phase 1
- Ingest the same file twice, then a file with one changed paragraph. Count chunks and active chunks.
- Ingest a corrupt PDF, an empty file, a password-protected file, and a file with a wrong extension. Batch must continue.
- Ingest a PDF with a table spanning pages. Are rows split mid-row?
- Downgrade then upgrade the migration on a populated database.
- Check that every chunk has an embedding with the right dimension. Query for NULL or wrong-length vectors.
- Very large file and a file with unusual unicode.

## Phase 2
- Hub A user queries for content that exists only in hub B. Must return nothing from B.
- User without tag `finance` asks a question whose only answer lives in a `finance`-tagged doc. Must refuse, and the chunk must not appear in logs or context.
- Deactivated (superseded) chunk is still returned?
- Query where fewer than k eligible chunks exist, and where more than k exist but few pass the filter.
- Model output cites an ID that was not in context. Is it stripped and flagged?
- Out-of-corpus question. Does it refuse or hallucinate?
- Same inputs twice: is context ordering identical?

## Phase 3
- Hand-compute recall@k, MRR, and nDCG for a tiny fixture and compare.
- Degrade retrieval (set k=1) and confirm the gate fails.
- Are unreviewed golden items excluded from the gate?
- Does the report record the judge model and config hash?

## Phase 4
- Call every `/v1` route with no token, an expired token, and a token for another org.
- Put `user_tags` or `org_id` in the request body. Must be ignored.
- Trigger an exception. Does the response leak a stack trace or path?
- Exceed the rate limit. Check 429 and `Retry-After`.
- Disconnect an SSE client mid-stream. Resource leak?
- Elliptical follow-up without history, and with history.

## Phase 5
- Direct injection ("ignore previous instructions...") in the query.
- Injection hidden inside an ingested document, and inside a table cell.
- System prompt exfiltration attempts.
- PII in the query: does it appear in logs, traces, or the prompt?
- Cross-hub and ACL probes through the query text ("show me documents from the finance hub").
- Guardrail failure mode: if the guard errors, does the system fail closed?

## Phase 6
- Pick a random request. Can you reconstruct the whole pipeline from its trace alone?
- Do traces contain raw PII or full document text?
- Promote a prompt version that regresses the golden set. Is it blocked?
- Is the prompt version in the trace the one actually used?

## Phase 7
- Kill the LLM mid-request, add latency, return malformed output. Does the system degrade gracefully?
- Update a document, then query. Is a stale cache entry returned?
- Follow the README on a clean machine or container.
- Run the full CI command set locally and compare against CI config.
