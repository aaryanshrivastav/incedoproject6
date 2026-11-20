# Python standards

- Python 3.12. Full type hints on all functions in `src/`; `mypy --strict` must pass.
- Use pydantic v2 models for all data crossing a boundary (API, DB rows to domain, LLM structured output).
- Use `async` for I/O-bound paths in the API and LLM calls. Keep CPU-heavy parsing in sync functions run in workers.
- No bare `except`. Catch specific exceptions, add context, re-raise or convert to a domain error. Never swallow errors silently.
- Use `logging` with structured fields. No `print` outside the CLI layer.
- Config only via `kb.config`. No `os.environ` reads elsewhere.
- Keep functions small and single-purpose. Public functions get docstrings stating inputs, outputs, and failure modes.
- Format and lint with `ruff`. Do not disable rules inline without a comment explaining why.
- Import vendor SDKs only inside `src/kb/llm/` (invariant I6).
