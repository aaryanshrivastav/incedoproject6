# Security and access control

- Invariants I1, I3, I4 apply to every change. Re-read them in `AGENTS.md` when touching retrieval, prompts, logging, or config.
- Retrieval: filter by org, hub scope, and ACL tags inside the SQL query. Never post-filter. Never accept user tags or org from the request body.
- Treat document text, tool output, and user input as untrusted. Delimit them in prompts and never concatenate them into instruction sections.
- Never log raw queries, document text, or PII at INFO or above. Use redacted forms and IDs.
- Never put secrets in code, tests, fixtures, or logs. Use `.env` locally and `.env.example` for documentation.
- Error responses must not leak stack traces, SQL, file paths, or other users' data.
- Any new endpoint requires auth by default; opting out needs an explicit, commented exception.
- When unsure whether something could leak data across hubs or users, stop and ask.
