# Architecture

## Components

```mermaid
flowchart TD
    subgraph Ingest [ingestion - offline or at startup]
      F[md / txt / csv + acl] --> CH[chunking] --> EM[embedder] --> ST[(SQLite vector store)]
    end
    subgraph Serve [serving]
      API[HTTP server: auth -> role] --> AG[agent pipeline]
      AG --> RT[retriever: dense + BM25 + RRF + ACL]
      RT --> ST
      AG --> LLM[LLM client: mock | Gemini]
      AG --> TR[tracer: spans, tokens, TTFT, cost]
      API --> MX[Prometheus metrics]
    end
```

## Why these stages

| Stage | Reason |
|---|---|
| guard | cheap, deterministic rejection of empty / oversized / instruction-like questions before spending tokens |
| router (small model) | greetings and out-of-scope questions never touch retrieval or the large model |
| hybrid retrieval | dense catches paraphrase, BM25 catches identifiers and numbers; RRF fuses ranks without calibrating scores |
| support gate | an IDF-weighted share of question terms found in the best chunk; below `min_support` the agent refuses instead of guessing |
| rewrite (self-reflection on retrieval) | weak first retrieval -> reformulate once, merge, retry |
| citation audit | deterministic: at least one citation, all indices valid; otherwise revise once, then refuse |
| verifier (self-check) | a second pass checks every sentence against the chunks it cites; failure -> one revision -> refuse |
| redact | credentials and ID-like numbers never leave the service |

## Observability

One trace per request. Spans: `request` > `router` / `retrieve` / `rewrite` / `answer` / `verify` (+ `sanitize`, `embed_chunks` at ingest). Each LLM span carries tokens in/out, TTFT, latency, cost, tier and status; `summary()` derives tokens/sec and cost-per-request. The JSON maps 1:1 onto OpenTelemetry spans (swap `Tracer.export` for an OTLP or Cloud Trace exporter).

## Cost levers (and how they are measured)

Model routing (small tier for router/rewrite/verify), hard per-request budgets (calls, tokens, cost), context cap per chunk, support gate (refusing early skips the large model). The ablation table in the README toggles each stage; real cost numbers need `KB_PRICE_*` and a real model.

## Scaling path

| Limit | Next step |
|---|---|
| exact search over SQLite (~10^5 chunks) | implement `VectorStore` on Vertex AI Vector Search or AlloyDB pgvector; keep the ACL filter as a metadata filter |
| per-instance index | build the index in a pipeline (Cloud Run job / Cloud Build) and load it read-only; today it ingests at startup |
| in-process concurrency cap | Cloud Run concurrency + max instances; add a queue for long jobs |
| role from token map | verify Google ID tokens / IAP headers in a gateway and map groups to roles |
