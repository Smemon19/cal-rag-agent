#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[error] line $LINENO: $BASH_COMMAND" >&2' ERR

# Runbook on failure (prints on any non-zero exit)
on_exit() {
  code=$?
  if [ "$code" -ne 0 ]; then
    echo "[runbook] Deployment failed (exit $code). Next steps:" >&2
    cat <<'RB' >&2
Quick triage

Tail logs (live):
gcloud run services logs tail "$SERVICE" --region "$REGION" --stream

Ensure secret accessor IAM (uses actual service account):

SA=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(spec.template.spec.serviceAccountName)')
gcloud secrets add-iam-policy-binding openai-api-key \
  --member="serviceAccount:$SA" \
  --role="roles/secretmanager.secretAccessor"

Verify image includes chroma_db/ and that CHROMA_DIR=/tmp/.calrag/chroma is set.

If timeouts: try lower latency
gcloud run services update "$SERVICE" --region "$REGION" --min-instances=1
RB
  fi
}
trap 'on_exit' EXIT

# Idempotent one-and-done deploy to Cloud Run (managed)
# Cloud Shell-friendly; requires gcloud configured

# -------- Inputs & Defaults --------
PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}
REGION=${REGION:-us-central1}
SERVICE=${SERVICE:-cal-rag-agent}
ART_REPO=${ART_REPO:-cal-rag-repo}
OPENAI_KEY=${OPENAI_KEY:-}
CPU=${CPU:-1}
MEMORY=${MEMORY:-1Gi}
CONCURRENCY=${CONCURRENCY:-80}
TIMEOUT=${TIMEOUT:-900s}
MIN_INSTANCES=${MIN_INSTANCES:-0}
MAX_INSTANCES=${MAX_INSTANCES:-3}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-}

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is not set and no gcloud default project configured." >&2
  exit 2
fi
if [[ -z "${OPENAI_KEY}" ]]; then
  echo "OPENAI_KEY is required (export OPENAI_KEY=...)" >&2
  exit 2
fi

gcloud config set project "$PROJECT_ID" >/dev/null

# -------- Enable Required APIs (idempotent) --------
echo "[info] Ensuring required APIs are enabled..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com --quiet

# -------- Secret Manager (idempotent) --------
echo "[info] Ensuring Secret Manager secret exists and is updated..."
SECRET_NAME=openai-api-key
if gcloud secrets describe "$SECRET_NAME" --quiet >/dev/null 2>&1; then
  printf "%s" "$OPENAI_KEY" | gcloud secrets versions add "$SECRET_NAME" --data-file=- --quiet >/dev/null
else
  printf "%s" "$OPENAI_KEY" | gcloud secrets create "$SECRET_NAME" --replication-policy=automatic --data-file=- --quiet >/dev/null
fi

# -------- Artifact Registry (idempotent) --------
echo "[info] Ensuring Artifact Registry repo exists..."
if ! gcloud artifacts repositories describe "$ART_REPO" --location="$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$ART_REPO" --repository-format=docker --location="$REGION" --quiet
fi

# -------- Preflight: ChromaDB presence --------
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -d "$ROOT_DIR/chroma_db" || -z $(ls -A "$ROOT_DIR/chroma_db" 2>/dev/null) ]]; then
  echo "[fail] chroma_db/ missing or empty. Ensure the vector store is present before deploy." >&2
  exit 3
fi

# -------- Build & Tag --------
TS=$(date +%Y%m%d-%H%M%S)
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$ART_REPO/$SERVICE:$TS"
echo "[info] Building image: $IMAGE"
gcloud builds submit --tag "$IMAGE" "$ROOT_DIR" --quiet

# -------- Deploy to Cloud Run --------
echo "[info] Deploying to Cloud Run service: $SERVICE"
COMMON_ARGS=(
  --region="$REGION"
  --platform=managed
  --allow-unauthenticated
  --cpu="$CPU"
  --memory="$MEMORY"
  --concurrency="$CONCURRENCY"
  --timeout="$TIMEOUT"
  --max-instances="$MAX_INSTANCES"
  --min-instances="$MIN_INSTANCES"
  --set-env-vars=CHROMA_DIR=/tmp/.calrag/chroma,FIREBASE_APP_HOSTING=true
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest
)
if [[ -n "$SERVICE_ACCOUNT" ]]; then
  COMMON_ARGS+=(--service-account="$SERVICE_ACCOUNT")
fi

gcloud run deploy "$SERVICE" --image "$IMAGE" "${COMMON_ARGS[@]}" --quiet

# -------- Ensure IAM for secret access --------
echo "[info] Ensuring service account has secret accessor role..."
SA=$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(template.spec.serviceAccountName)')
SA=${SA:-"$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"}
if ! gcloud secrets get-iam-policy "$SECRET_NAME" --format=json | grep -q "$SA"; then
  gcloud secrets add-iam-policy-binding "$SECRET_NAME" \
    --member="serviceAccount:$SA" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet >/dev/null
fi

# -------- Output URL & Smoke Check --------
URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')
echo "[ok] Deployed: $URL"

echo -n "[probe] /?healthz=1 -> "
HC=$(curl -s -o /dev/null -w "%{http_code}" "$URL/?healthz=1" || true)
echo "$HC"
if [[ "$HC" != "200" ]]; then
  echo "[fail] Health probe returned $HC. Check logs and ensure OPENAI_KEY is valid." >&2
  echo "Runbook:"
  echo "  - gcloud run services describe $SERVICE --region $REGION"
  echo "  - gcloud run services logs read $SERVICE --region $REGION --limit 200"
  echo "  - Verify Secret Manager: gcloud secrets versions list openai-api-key"
  exit 4
fi

# -------- Optional: Firebase Hosting front door --------
if [[ -f "$ROOT_DIR/firebase.json" ]] && command -v firebase >/dev/null 2>&1; then
  CHECK=$(python3 - <<'PY'
import json, os
fn = os.path.join(os.environ.get('ROOT_DIR','.'),'firebase.json')
svc=os.environ.get('SERVICE','')
region=os.environ.get('REGION','')
try:
    with open(fn,'r') as f:
        j=json.load(f)
    rew=(j.get('hosting') or {}).get('rewrites') or []
    ok=any(((r.get('run') or {}).get('serviceId')==svc and (r.get('run') or {}).get('region')==region) for r in rew)
    print('OK' if ok else 'MISS')
except Exception:
    print('MISS')
PY
)
  if [[ "$CHECK" == "OK" ]]; then
    echo "[info] Deploying Firebase Hosting front door..."
    if firebase deploy --only hosting --project "$PROJECT_ID" --non-interactive; then
      HOSTING_SITE=$(firebase hosting:sites:list --project "$PROJECT_ID" --json | python3 - <<'PY'
import json,sys
j=json.load(sys.stdin)
sites=j.get('result') or j.get('sites') or []
site=None
for s in sites:
    sid = s.get('site') or s.get('name') or s.get('siteId') or ''
    if sid:
        sid = sid.split('/')[-1] if '/' in sid else sid
        if not site:
            site = sid
        if sid == sys.argv[1]:
            site = sid
            break
print(site or '')
PY
 "$PROJECT_ID")
      if [[ -n "$HOSTING_SITE" ]]; then
        echo "[ok] Firebase Hosting: https://$HOSTING_SITE.web.app"
      else
        echo "[warn] Firebase deploy succeeded but could not detect Hosting site."
      fi
    else
      echo "[warn] Firebase Hosting deploy failed; Cloud Run remains deployed. Try: firebase deploy --only hosting --project $PROJECT_ID" >&2
    fi
  else
    echo "[skip] Firebase Hosting: missing or mismatched rewrite for service=$SERVICE region=$REGION"
  fi
else
  if [[ ! -f "$ROOT_DIR/firebase.json" ]]; then echo "[skip] Firebase Hosting: firebase.json not found"; fi
  if ! command -v firebase >/dev/null 2>&1; then echo "[skip] Firebase Hosting: CLI not installed"; fi
fi

# -------- Helpful Epilogue --------
cat <<EOF

Add a custom domain in Firebase Console → Hosting → Custom domains.

Toggle helpers (cold start vs cost):
- Set low-latency (keep one warm instance):
  gcloud run services update "\$SERVICE" --region "\$REGION" --min-instances=1
- Back to lowest cost:
  gcloud run services update "\$SERVICE" --region "\$REGION" --min-instances=0
- Cap spikes (example):
  gcloud run services update "\$SERVICE" --region "\$REGION" --max-instances=3

Next steps:
- Tail logs:
  gcloud run services logs read \$SERVICE --region \$REGION --follow
- Scale to 1 min instance:
  gcloud run services update \$SERVICE --region \$REGION --min-instances=1
- Update max instances:
  gcloud run services update \$SERVICE --region \$REGION --max-instances=10
- Roll back to previous revision:
  gcloud run services list-revisions \$SERVICE --region \$REGION
  gcloud run services update-traffic \$SERVICE --region \$REGION --to-revisions="REVISION=100"

EOF


