#!/usr/bin/env bash
# Deploys the service to Cloud Run with Vertex AI (Gemini) and no API keys: the runtime service account is the identity.
# Usage: PROJECT_ID=my-project REGION=asia-south1 KB_MODEL_SMALL=<id> KB_MODEL_LARGE=<id> KB_EMBED_MODEL=<id> \
#        KB_PRICE_SMALL="in,out" KB_PRICE_LARGE="in,out" ./deploy/cloudrun.sh
# NOT RUN by the author's build environment (no gcloud login): read it, then run it once yourself.
set -euo pipefail
: "${PROJECT_ID:?set PROJECT_ID}" "${KB_MODEL_SMALL:?}" "${KB_MODEL_LARGE:?}" "${KB_EMBED_MODEL:?}" "${KB_PRICE_SMALL:?}" "${KB_PRICE_LARGE:?}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-knowledge-agent}"
SA="${SERVICE}-runtime@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud services enable run.googleapis.com aiplatform.googleapis.com cloudbuild.googleapis.com --project "$PROJECT_ID"
gcloud iam service-accounts describe "$SA" --project "$PROJECT_ID" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${SERVICE}-runtime" --project "$PROJECT_ID" --display-name "Knowledge agent runtime"
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$SA" --role roles/aiplatform.user --condition=None >/dev/null

gcloud run deploy "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --source . \
  --service-account "$SA" \
  --no-allow-unauthenticated \
  --concurrency 8 --max-instances 3 --memory 512Mi --cpu 1 --timeout 60 \
  --set-env-vars "KB_LLM=gemini,KB_GEMINI_BACKEND=vertex,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},KB_MODEL_SMALL=${KB_MODEL_SMALL},KB_MODEL_LARGE=${KB_MODEL_LARGE},KB_EMBED_MODEL=${KB_EMBED_MODEL},KB_PRICE_SMALL=${KB_PRICE_SMALL},KB_PRICE_LARGE=${KB_PRICE_LARGE}"

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format 'value(status.url)')"
echo "Deployed: $URL"
echo "Try it:   curl -s -X POST \"$URL/v1/ask\" -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" -H 'Content-Type: application/json' -d '{\"question\":\"How many days of annual leave do employees get?\"}'"
