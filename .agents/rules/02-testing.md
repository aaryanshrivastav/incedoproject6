# Testing standards

- Layout: `tests/unit` (pure logic, no network or DB), `tests/integration` (real Postgres via docker compose), `tests/e2e` (full path), `tests/adversarial` (security and red-team).
- Tests use the fake LLM, embedder, and reranker from `domain/` by default. Real providers are only used in clearly marked, opt-in tests (`@pytest.mark.live`).
- Every acceptance criterion in a prompt maps to at least one named test. Name tests after the behavior, e.g. `test_reingesting_same_file_creates_no_new_chunks`.
- Security-relevant behavior (ACL, hub scope, injection, PII) requires negative tests: prove the bad thing cannot happen, not just that the good thing works.
- Tests must be deterministic. Seed randomness, freeze time, avoid network.
- A bug fix starts with a failing regression test.
- Report real output of the test run. If a test is skipped or flaky, say so.
