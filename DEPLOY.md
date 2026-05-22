# Deploying the Phishing Detection Demo (Firebase)

The whole app deploys through the **Firebase CLI** — no Docker, no gcloud.

- **Frontend** (`frontend/`) → Firebase Hosting
- **Backend** (Python ML API) → a Cloud Function for Firebase (`functions/`)

The Hosting config rewrites `/api/**` to the function, so the browser sees a
single origin.

## Prerequisites

- **Node.js** (https://nodejs.org) and the Firebase CLI:
  `npm install -g firebase-tools`
- **Python 3.12** (you already have it for local dev)
- A Firebase project on the **Blaze (pay-as-you-go) plan.** Cloud Functions
  require Blaze even though this demo stays within the free monthly quota.
- A **Google Safe Browsing API key** (you already have one in `.env`).

> No Docker and no gcloud CLI are needed for this path.

## One-time setup

1. **Log in and select your project:**
   ```powershell
   firebase login
   ```
   Open `.firebaserc` and replace `REPLACE_WITH_YOUR_FIREBASE_PROJECT_ID` with
   your Firebase project ID (from the Firebase console).

2. **Upgrade the project to the Blaze plan:**
   Firebase console → ⚙ → Usage and billing → modify plan → Blaze. (Free-tier
   quota covers a class demo; you only pay above it.)

3. **Give the function your Safe Browsing key.** Create `functions/.env`
   (it's gitignored) with:
   ```
   SAFE_BROWSING_API_KEY=AIzaSy...your-key...
   ```
   Firebase loads this file's variables into the deployed function.

4. **Set up the function's Python venv** (the Firebase CLI uses it to discover
   the function during deploy):
   ```powershell
   cd functions
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   deactivate
   cd ..
   ```

## Every deploy

1. **Train the models** if you haven't lately (writes `backend/models/*.joblib`):
   ```powershell
   .\backend\.venv\Scripts\python.exe -m backend.train_models
   ```

2. **Bundle the backend into the function.** Firebase only uploads `functions/`,
   so this copies the backend package + models in next to `functions/main.py`:
   ```powershell
   python prepare_functions.py
   ```

3. **Deploy both pieces:**
   ```powershell
   firebase deploy
   ```
   (Or separately: `firebase deploy --only functions` /
   `firebase deploy --only hosting`.)

When it finishes, the CLI prints your live URL: `https://<project-id>.web.app`.
Open it, scan `https://www.wikipedia.org`, and you should get four model
verdicts plus a Safe Browsing result.

## Verifying

- The function's health check, through Hosting:
  `https://<project-id>.web.app/api/health` → `{"status":"ok"}`
- The first scan after a deploy can take **5-15 seconds** — the function
  cold-starts and loads the four models. Later scans are fast. The frontend
  pings `/api/health` on page load to warm it up.

## Updating

- **Backend or model change** → `python prepare_functions.py` then
  `firebase deploy --only functions`
- **Frontend change** → `firebase deploy --only hosting`

## Troubleshooting

| Symptom | Fix |
|---|---|
| `firebase deploy` fails: "Cloud Functions requires the Blaze plan" | Upgrade the project to Blaze (one-time setup step 2). |
| Deploy can't find the function / Python errors during analysis | Make sure `functions/venv` exists (one-time setup step 4) and `python prepare_functions.py` has been run so `functions/backend/` is present. |
| Runtime error: "no module named backend" | You skipped `python prepare_functions.py` — the backend bundle isn't in `functions/`. |
| `safe_browsing.verdict` is `"unknown"` on the live site | `functions/.env` is missing or has the wrong key; recreate it and redeploy. |
| Deploy rejected: unsupported runtime `python312` | Change `"runtime"` in `firebase.json` to `"python311"`, recreate `functions/venv` with Python 3.11, and redeploy. |
| First scan times out | Cold start. Reload and try again, or scan once to warm the instance before the live demo. |

## Local development (unchanged)

You don't need any of the above to run locally. From the repo root:

```powershell
# Terminal 1 - backend
.\backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000

# Terminal 2 - frontend
cd frontend
python -m http.server 5000
```

Then open <http://localhost:5000>.
