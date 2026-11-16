# Security model

| Threat | Defence | Where | Tested |
|---|---|---|---|
| A user reads documents above their role | ACL filter before scoring; role from the bearer token only; body `role` ignored (and only allowed on loopback with an explicit flag) | `store.can_see`, `server` | `test_acl_*`, `test_role_in_body_cannot_escalate_privileges`, eval `acl_leaks = 0` |
| Restricted text leaks into the prompt | the prompt is built from already-filtered hits | `agent._context` | `test_restricted_text_never_reaches_the_model_for_wrong_role` |
| Prompt injection planted in a document | scrub instruction-like lines; context labelled as data; model has no action tools; answers must cite chunks and pass a verifier | `guardrails.sanitize`, `agent` | `test_poisoned_document_is_scrubbed...`, eval `inject OK` |
| Prompt injection in the question | rejected before any LLM call | `guardrails.looks_like_injection` | `test_injection_in_question_is_blocked...` |
| Runaway cost / loops | per-request caps on LLM calls, tokens, cost; one revision max | `guardrails.Budget`, `agent` | `test_budget_caps_calls_and_cost` |
| Credential or ID leakage in answers | output redaction | `guardrails.redact_output` | `test_output_redaction` |
| Provider outage | any model/network failure ends in a recorded `refused` status, never a 500 | `agent.ask` | `test_llm_outage_is_a_200...` |
| Unauthenticated access | bearer tokens (constant-time compare) or Cloud Run IAM (`--no-allow-unauthenticated`); no keys in the image, service account on Cloud Run | `server`, `deploy/cloudrun.sh` | `test_auth_required...` |
| Oversized / malformed input | body and question length limits, JSON validation, concurrency cap with 429 | `server` | `test_validation_and_limits`, `test_concurrency_cap...` |

## Residual risks (read these)

- Injection scrubbing is a regex heuristic; a determined attacker can phrase around it. The structural defences (read-only retrieval tool, no side-effects, citation + verifier) are what limit the damage.
- The mock verifier checks lexical support, which a real model's verifier does better but not perfectly; treat the verifier as a filter, not a proof.
- ACLs are only as good as the front matter on each document; there is no document-level provenance check.
- Retrieval-time ACLs do not stop a user inferring that a restricted document exists from a refusal's timing or wording. The refusal text is identical for "not found" and "not allowed" by design, but the machine-readable `reason` field can differ, so strip it for untrusted callers if that matters to you.
- Static bearer tokens are a stopgap. Use IAP or ID-token verification and group-to-role mapping for real deployments.
- No TLS in-process (Cloud Run terminates TLS); no per-user rate limiting.
