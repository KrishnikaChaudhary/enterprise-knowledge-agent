# Technical discovery (template: fill from a real conversation)

A forward-deployed engineer starts with the customer's problem, not the model. Run this with a real stakeholder (an HR ops lead, an IT service-desk owner, a support manager). Do not fill it in from imagination; leave it blank if you have not done the conversation.

**Who / when:** [role, team, date, length]

## 1. Problem and outcome
- What question do people ask repeatedly, and where do they ask it today?
- Volume per week, and what a wrong or slow answer costs.
- Success in one sentence, and the number that proves it (deflection rate, time to answer, escalation rate).

## 2. Data readiness
- Where does the knowledge live (wiki, PDFs, tickets, spreadsheets, a database)? Who owns it? How stale is it?
- Which sources are authoritative when two disagree?
- Structured vs unstructured split; export formats; update frequency.

## 3. Access and risk
- Who may see what? Where is that policy written down? Is there an existing group/role system to reuse?
- What must never be answered (legal, HR cases, salaries)? What is the cost of a leak vs a refusal?
- Regulatory constraints: data residency, retention, audit logging.

## 4. Integration and operations
- Channel (Slack, intranet, ticketing), auth (SSO), latency expectations, expected peak load.
- Who operates it after launch, and what do they monitor?

## 5. Evaluation
- Ask for 30-50 real questions with the right answer and the right source. These become `data/eval/questions.jsonl`.
- Which failure is worse: refusing a good question or answering a bad one?

## Decisions this should change (fill after the call)
| Finding | Design decision it changed |
|---|---|
| [e.g. "salary questions must never be answered"] | [e.g. restricted-tier ACL + refusal test] |
