# Deploying the Phishing Detection Demo

This is a one-time setup followed by repeatable deploy commands. The app has
two pieces: a Cloud Run backend (Python ML service) and a Firebase Hosting
frontend that proxies API calls to it.

## Prerequisites

- Google Cloud account with billing enabled (Cloud Run is free-tier friendly)
- Firebase project linked to the same GCP project
- `gcloud` CLI and `firebase` CLI installed and authenticated
- Docker Desktop (for building the backend image)
- A Google Safe Browsing API key (free) —
  https://developers.google.com/safe-browsing/v4/get-started

## One-Time Setup

1. **Set the project IDs:**
   ```bash
   gcloud config set project <YOUR_GCP_PROJECT_ID>
   ```
   In `.firebaserc`, replace `REPLACE_WITH_YOUR_FIREBASE_PROJECT_ID` with your
   Firebase project ID (same as the GCP one).

2. **Enable required Google Cloud APIs:**
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
     artifactregistry.googleapis.com safebrowsing.googleapis.com
   ```

3. **Train the models locally** (only needed once, or whenever you retrain):
   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\activate              # Windows PowerShell
   pip install -r requirements-dev.txt
   cd ..
   python -m backend.train_models
   ```
   This writes `backend/models/{gb,rf,xgb,svm}.joblib`. These files get baked
   into the Docker image and are not committed to git.

## Deploy the Backend (Cloud Run)

```bash
cd backend
gcloud builds submit --tag gcr.io/<YOUR_GCP_PROJECT_ID>/phishing-demo
gcloud run deploy phishing-demo \
  --image gcr.io/<YOUR_GCP_PROJECT_ID>/phishing-demo \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 60 \
  --set-env-vars SAFE_BROWSING_API_KEY=<YOUR_API_KEY>,FRONTEND_ORIGIN=https://<YOUR_FIREBASE_PROJECT_ID>.web.app
```

After deploy, `gcloud` prints the Cloud Run URL. Verify it's healthy:

```bash
curl https://phishing-demo-XXXX.run.app/api/health
# {"status":"ok"}
```

## Deploy the Frontend (Firebase Hosting)

```bash
firebase deploy --only hosting
```

The CLI prints the public URL (e.g. `https://<project-id>.web.app`). Open it
in a browser and verify that:

- The page loads with the URL input form
- Submitting `https://github.com` returns 4 model predictions and a Safe
  Browsing verdict within ~10 seconds
- `/api/health` is reachable through Firebase (it should be proxied to
  Cloud Run via the rewrite in `firebase.json`)

## Updating

To redeploy after code changes:
- Backend changes → re-run `gcloud builds submit` and `gcloud run deploy`
- Frontend changes → re-run `firebase deploy --only hosting`
- Model changes → re-run `python -m backend.train_models`, then redeploy
  the backend

## Local Development

To run the whole app locally without deploying:

```bash
# Terminal 1 — backend
backend\.venv\Scripts\activate
python -m uvicorn backend.main:app --port 8000

# Terminal 2 — frontend
cd frontend
python -m http.server 5000
```

Then open <http://localhost:5000>. The frontend detects it's running on port
5000 and points API calls at <http://localhost:8000> directly.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| First scan times out at ~60s | Cloud Run cold start. The frontend hits `/api/health` on page load to warm it; if you skip that, the first scan is slow. |
| `safe_browsing.verdict == "unknown"` | `SAFE_BROWSING_API_KEY` env var missing or invalid on Cloud Run. |
| CORS errors in browser console | `FRONTEND_ORIGIN` env var doesn't match the Firebase Hosting URL. Update with `gcloud run services update phishing-demo --update-env-vars FRONTEND_ORIGIN=...` |
| Docker build fails on `Missing models` | Run `python -m backend.train_models` before building the image. |
