# Phase N audit

- Audited commit: `<hash>`
- Date: `<date>`
- Auditor session: fresh, read-only
- Verdict: **PASS | FAIL**

## Standard checks

| Check | Command | Result |
|---|---|---|
| Tests | `uv run pytest -q` | |
| Lint | `uv run ruff check .` | |
| Format | `uv run ruff format --check .` | |
| Types | `uv run mypy src` | |

## Exit criteria

| ID | Criterion (short) | Status | Evidence |
|---|---|---|---|
| EN.1 | | PASS / FAIL / UNVERIFIED | command + output excerpt, or test name, or file:line |

## Invariants checked

| ID | Status | Evidence |
|---|---|---|
| I1 | | |

## Adversarial probes

| Probe | Result | Notes |
|---|---|---|

## Findings

| # | Severity | Title | Reproduction | Suggested fix | Resolution |
|---|---|---|---|---|---|
| F1 | blocker/major/minor | | steps | | open |

## Summary

Two to four sentences: what is solid, what is not, what must change before re-audit.
