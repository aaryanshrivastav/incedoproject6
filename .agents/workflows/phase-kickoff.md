---
description: Start a phase. Verifies the previous phase passed audit and produces a plan before any code is written.
---

# /phase-kickoff N

1. Read `AGENTS.md`, `docs/ARCHITECTURE.md`, and Phase N in `docs/PHASES.md`.
2. If N > 0, confirm Phase N-1 has status `audit passed` in `docs/PHASES.md` and a matching `docs/audits/phase-(N-1)-audit.md` with verdict PASS. If not, stop and report.
3. List any `needs input` items in `docs/DECISIONS.md` that block this phase. If any, stop and ask.
4. Inspect the current codebase and note what already exists that this phase will build on (module names, schemas, signatures).
5. Produce a plan artifact: the build prompts for this phase in order, files each will touch, risks, and open questions.
6. Do not write or edit any code. Stop and wait for approval.
