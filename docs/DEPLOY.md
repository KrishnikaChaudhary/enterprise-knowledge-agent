# Deploy to Google Cloud Run (Vertex AI Gemini, no API keys)

1. `gcloud auth login` and `gcloud config set project <id>`.
2. Pick model ids from the Vertex AI model garden and prices from the pricing page. Export `KB_MODEL_SMALL`, `KB_MODEL_LARGE`, `KB_EMBED_MODEL`, `KB_PRICE_SMALL`, `KB_PRICE_LARGE`.
3. `PROJECT_ID=<id> REGION=asia-south1 ./deploy/cloudrun.sh`. It enables the APIs, creates a runtime service account with `roles/aiplatform.user`, and deploys from source with `--no-allow-unauthenticated`.
4. Call it with an identity token: `curl -X POST "$URL/v1/ask" -H "Authorization: Bearer $(gcloud auth print-identity-token)" -d '{"question":"..."}'`.
5. Check Cloud Logging for the JSON logs (`severity`, `message`, fields) and scrape `/metrics` with Managed Service for Prometheus if wanted.

Notes
- The index is built at container start (`KB_DB=/tmp/kb.db`). With the real embedder this costs embedding calls on every cold start; for production build the index once and ship it, or move to a managed vector store.
- Cloud Run IAM authenticates callers but does not tell the app *who* they are, so every caller gets `KB_DEFAULT_ROLE`. For role-based access, set `KB_TOKENS` or put a gateway in front that maps identities to roles.
- This script and the Dockerfile have not been run by the author's build environment. Run once, read the output, and fix anything your project's org policy blocks (for example, domain-restricted sharing or VPC-SC).
