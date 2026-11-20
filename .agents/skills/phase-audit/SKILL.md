---
name: phase-audit
description: Procedure for independently auditing a completed build phase of the knowledge assistant against its exit criteria and invariants. Use whenever the user runs /audit-phase, asks to audit, review, verify, validate, or sign off a phase or milestone, or asks whether a phase is done, even if they do not say "audit".
---

# Phase audit

You are the auditor, not the builder. Your job is to find what is wrong or unproven.

## Principles

- Evidence over claims. A criterion is PASS only with evidence you produced in this session: a command and its output, a test name that you ran, or a file and line you inspected.
- No evidence means UNVERIFIED. UNVERIFIED blocks a PASS verdict just like FAIL.
- Try to break it. Passing happy-path tests prove little. Use the probes in `references/probes.md` for the phase under audit, then invent at least three more.
- Read-only. Write only to `docs/audits/phase-N-audit.md`. Do not fix, refactor, or reformat anything.
- Do not rely on the builder's summaries, commit messages, or comments as evidence.

## Procedure

1. Read `AGENTS.md` invariants and the Phase N criteria in `docs/PHASES.md`.
2. Establish a clean baseline: fresh database where relevant, `git status` clean, note the commit hash audited.
3. Run the standard checks: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`. Record results.
4. Walk the criteria one by one. For each, run the specific check and record the evidence.
5. Check relevant invariants (I1-I10). For each, search the code for violations, for example `git grep` for vendor SDK imports outside `llm/` (I6), inline prompt strings (I9), `os.environ` outside config (I4), and queries on chunks without the access filter (I1).
6. Run phase-specific probes from `references/probes.md`.
7. Inspect test quality: do tests assert behavior, or just that code runs? Are negative cases covered? Are there skipped or xfail tests hiding failures?
8. Write the report using `references/report-template.md`.

## Severity

- **Blocker:** violates an invariant, leaks data, loses data, or makes a criterion impossible to satisfy.
- **Major:** a criterion fails or is unverified, or behavior is wrong in a realistic scenario.
- **Minor:** quality, clarity, or hygiene issues that do not affect criteria.

## Verdict

PASS only if every criterion is PASS and there are zero blocker or major findings. Otherwise FAIL, with the shortest list of what must change.
