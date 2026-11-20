---
description: Independent audit of a completed phase against its exit criteria and the invariants. Read-only for source code.
---

# /audit-phase N

Load the `phase-audit` skill and follow it. Summary:

1. You are an independent auditor. Assume the builder's summary may be wrong. Do not modify any file outside `docs/audits/`.
2. Read `AGENTS.md` (invariants I1-I10) and Phase N in `docs/PHASES.md` (criteria E N.x). Do not read the builder's commit messages or summaries as evidence.
3. Run the real verification commands: tests, ruff, mypy, migrations, and any CLI the criteria mention. Record exact commands and trimmed output.
4. For every exit criterion, decide PASS, FAIL, or UNVERIFIED, with evidence (command output, file and line references, or test names). No evidence means UNVERIFIED, never PASS.
5. Check each invariant relevant to this phase.
6. Run the adversarial probes listed in the skill for this phase and try at least three additional ways to break the work.
7. Write `docs/audits/phase-N-audit.md` using the template in the skill. Findings are ranked blocker, major, or minor, each with reproduction steps.
8. Verdict is PASS only if every criterion is PASS and there are no blocker or major findings.
9. Do not fix anything. Stop after writing the report.
