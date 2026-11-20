---
description: Close a phase after a PASS audit. Updates status, decisions, and tags the commit.
---

# /phase-closeout N

1. Confirm the latest `docs/audits/phase-N-audit.md` has verdict PASS and was produced after the last code change (compare timestamps or commits). If not, stop.
2. Set Phase N status to `audit passed` in `docs/PHASES.md`.
3. Add any decisions made during the phase to `docs/DECISIONS.md`, and resolve any `needs input` items that were settled.
4. Update docs that describe changed interfaces, schemas, or commands.
5. Commit as `N.close: phase N complete` and create tag `phase-N`.
6. Summarize what is now available for the next phase, including module names and entry points the next prompts should build on.
