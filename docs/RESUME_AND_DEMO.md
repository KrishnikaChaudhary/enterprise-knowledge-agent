# Resume bullets, demo script, interview notes

## What you can honestly claim today (offline run)
- Built an enterprise knowledge agent with hybrid retrieval (dense + BM25 + reciprocal-rank fusion) over structured (CSV) and unstructured (Markdown) sources, with per-document access control enforced before scoring.
- Designed a guarded answer pipeline (router, query-rewrite reflection, citation audit, verifier with one revision, refusal on weak support) and a 60-question evaluation harness with ablations across 6 configurations: 0 access-control leaks, 100% refusal and injection-resistance, 100% cited answers on the synthetic set.
- Shipped it as a service: bearer-token roles, per-request cost/token budgets, Prometheus metrics, JSON logs, graceful shutdown; Gemini/Vertex AI clients and a Cloud Run deploy script; 76 automated tests.

Say "synthetic corpus, offline mock model" if asked. Do NOT quote latency or cost from the offline table.

## Claim only AFTER you run it
- Gemini retrieval/answer quality: `python -m kagent.evalrun --llm gemini --embed gemini` then quote the real numbers, including where real embeddings beat BM25 (or do not).
- "Deployed on Cloud Run with Vertex AI Gemini": only after `deploy/cloudrun.sh` succeeds and you have the URL.
- Technical discovery: only after a real conversation, filled into `docs/DISCOVERY.md`.

## 3-minute demo
1. `kagent ask "...annual leave..."`: answer, citation, Metrics line (calls, tokens, cost).
2. Same salary question as employee (refused) and as `--role hr` (answered): access control in retrieval.
3. Ask about reception hours: the planted "ignore previous instructions" line in the wiki is scrubbed and the answer is still correct.
4. `python -m kagent.evalrun`: walk the ablation table and the failing-questions list; say what the hash embedder cannot do and what you would measure with Gemini embeddings.
5. Open a trace (`--trace out.json`): request -> router -> retrieve -> answer -> verify spans with tokens and cost.

## Questions to be ready for
- *Why hybrid retrieval?* Dense for paraphrase, BM25 for identifiers/numbers; RRF avoids calibrating two score scales. On my tiny corpus BM25 alone matched it, which is why the harness exists.
- *How do you stop a user reading restricted data?* Filter before scoring; role from the token; the prompt is built only from visible chunks. Tested end to end.
- *How do you know it is not hallucinating?* Citations are audited deterministically; a verifier checks each sentence against its chunks; weak support -> refuse. And the eval has unanswerable questions.
- *What is mocked?* The model and embeddings (offline). Retrieval, ACLs, parsing, tracing, budgets, HTTP clients and the server are real and tested.
- *What next?* Real-embedding eval, Vertex Vector Search behind `VectorStore`, a prebuilt index, group-to-role mapping from IAP, an LLM-as-judge for answer quality.
