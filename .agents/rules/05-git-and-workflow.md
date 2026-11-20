# Git and workflow

- One prompt = one commit (or a short series if the prompt asks for it). Never mix unrelated changes.
- Commit format: `<phase>.<step>: <imperative summary>`, e.g. `1.2: add recursive chunker with table-safe splitting`. Body lists tests added and commands run.
- Do not commit generated reports under `eval/reports/` (except baselines in `eval/baselines/`).
- Do not rewrite history on shared branches. Do not force push.
- Do not modify `docs/PHASES.md` exit criteria or `AGENTS.md` invariants unless the prompt says so. Status fields in `docs/PHASES.md` are updated only by `/phase-closeout`.
- Audits never modify source code. They write only to `docs/audits/`.
- Record meaningful design choices in `docs/DECISIONS.md` as you make them.
