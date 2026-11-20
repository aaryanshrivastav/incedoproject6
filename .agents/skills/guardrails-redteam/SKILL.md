---
name: guardrails-redteam
description: Patterns for input and output guardrails, prompt injection defense, PII handling, and the red-team test suite. Use whenever working on guardrails, prompt injection, indirect injection through documents, PII redaction, Presidio, moderation, topic restriction, jailbreaks, adversarial tests, or security testing of the assistant.
---

# Guardrails and red-team suite

## Layers

1. **Input guard:** screen the user query for injection patterns, redact PII before anything is logged or sent to the model, enforce topic policy. Fail closed: if the guard errors, block the request with a safe message.
2. **Context hygiene (I3):** retrieved text is delimited as untrusted data. Optionally screen chunks at ingestion and query time for instruction-like content and flag or drop them.
3. **Output guard:** check that claims are grounded in the context, scan for PII and policy violations, verify citations. Fail closed.
4. **Logging:** record decision, reason code, and rule ID. Never raw PII or full document text (I4).

## Red-team suite (`tests/adversarial/`)

Each case is data: `{id, category, input, setup, expected}`. Categories and minimum coverage:

- direct injection ("ignore previous instructions", role-play, encoding tricks)
- indirect injection (instructions hidden in ingested documents, table cells, metadata)
- system prompt exfiltration
- PII leakage (query containing PII; document containing PII; model asked to reveal it)
- out-of-scope requests (topic policy)
- cross-hub and ACL probes (asking for other hubs' or restricted documents by name or by trick)

At least 40 cases total. Pass-rate targets: 100% on access probes, at least 90% elsewhere. Failures are listed with the case ID in the report. The suite runs in CI.

## Principles

- Defense in depth: no single layer is trusted. Access control (I1) is enforced in SQL and does not depend on the model behaving.
- A blocked request returns a generic, safe message with no hint about which rule fired.
- Measure latency added by guards and report it.
- Do not weaken a test to make it pass. Fix the guard or record a justified exception.
