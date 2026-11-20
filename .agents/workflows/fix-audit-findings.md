---
description: Fix blocker and major findings from a phase audit, one finding per commit, with regression tests.
---

# /fix-audit-findings N

1. Read `docs/audits/phase-N-audit.md`. Work only on findings marked blocker or major, in severity order, unless the prompt names others.
2. For each finding: write a failing regression test that reproduces it, fix the cause (not the symptom), confirm the test passes, run the full suite, and commit as `N.fix: <finding id> <summary>`.
3. Do not refactor unrelated code or expand scope. If a finding needs a design change, stop and ask.
4. Update the audit report's finding rows with `fixed in <commit>` and the test name. Do not change any verdict or mark criteria PASS yourself; only a fresh `/audit-phase N` can do that.
5. Finish with the three-list report from `AGENTS.md` and recommend re-running the audit in a fresh session.
