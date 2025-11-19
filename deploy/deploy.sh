#!/usr/bin/env bash
set -euo pipefail

# Robust, idempotent deploy for Streamlit → Cloud Run, with Hosting proxy verification

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
SERVICE="${SERVICE:-cal-rag-agent}"
REGION="${REGION:-us-central1}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
IMAGE_BASE="${IMAGE_BASE:-gcr.io/${PROJECT_ID}/${SERVICE}}"
IMAGE_TAG="${IMAGE_BASE}:${STAMP}"

MODEL_CHOICE="${MODEL_CHOICE:-gpt-4.1-mini}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "[fail] PROJECT_ID not set and no gcloud default project configured." >&2
  exit 2
fi
if ! command -v gcloud >/dev/null 2>&1; then
  echo "[fail] gcloud CLI not found. Install the Google Cloud SDK first." >&2
  exit 2
fi

gcloud auth list --quiet >/dev/null || true
if ! gcloud auth print-identity-token >/dev/null 2>&1; then
  echo "[fail] You are not authenticated to gcloud. Run: gcloud auth login" >&2
  exit 2
fi

gcloud config set project "${PROJECT_ID}" >/dev/null

logs() {
  gcloud run services logs tail "${SERVICE}" --region "${REGION}" --stream
}

# Hint for secret
if ! gcloud secrets describe OPENAI_API_KEY --project "${PROJECT_ID}" >/dev/null 2>&1; then
  echo "[hint] Secret 'OPENAI_API_KEY' not found. Create it once, then re-run deploy:" >&2
  echo "  echo -n 'sk-…' | gcloud secrets create OPENAI_API_KEY --data-file=- --project ${PROJECT_ID}" >&2
fi

echo "[info] Building image: ${IMAGE_TAG}"
gcloud builds submit --tag "${IMAGE_TAG}" . --quiet

SECRET_FLAG=()
if gcloud secrets describe OPENAI_API_KEY --project "${PROJECT_ID}" >/dev/null 2>&1; then
  SECRET_FLAG+=(--set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest)
else
  echo "[warn] Skipping OPENAI_API_KEY binding (secret not found)." >&2
fi

echo "[info] Deploying service: ${SERVICE} (${REGION})"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE_TAG}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --min-instances 1 \
  --max-instances 10 \
  --cpu 1 \
  --memory 1Gi \
  --set-env-vars MODEL_CHOICE=${MODEL_CHOICE} \
  "${SECRET_FLAG[@]}" \
  --quiet

SERVICE_URL=$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')
if [[ -z "${SERVICE_URL}" ]]; then
  echo "[fail] Could not retrieve service URL." >&2
  exit 3
fi

echo "[ok] Cloud Run URL: ${SERVICE_URL}"

echo -n "[probe] Health check … "
if ! curl -fsS -m 20 "${SERVICE_URL}" >/dev/null; then
  echo "FAIL" >&2
  echo "[hint] Tail logs with:" >&2
  echo "  gcloud run services logs tail '${SERVICE}' --region '${REGION}' --stream" >&2
  exit 4
fi
echo "OK"

echo
echo "[verify] Open Hosting URL: https://cal-rag-agent.web.app"
echo "If you see 403, grant Invoker to the Hosting service agent:"
cat <<'IAM'
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
gcloud run services add-iam-policy-binding "${SERVICE}" \
  --region "${REGION}" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-firebaseapphosting.iam.gserviceaccount.com" \
  --role="roles/run.invoker"
IAM

echo
echo "[done] Deployed ${SERVICE} in ${REGION}. Use logs() helper if needed."
