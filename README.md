# csse415-final-project

CSSE415 Final Project — Detecting Phishing Websites.

Includes the original Jupyter notebooks (Logistic Regression, Decision Tree,
kNN, Random Forest, Gradient Boosting, XGBoost, SVM, Ridge/Lasso) and a
demo web app that scans a live URL through the four best-performing models
and compares results to Google Safe Browsing.

## Running the demo locally

Tested on **Windows 11 + Python 3.11 / 3.12**. macOS/Linux notes inline.

### 1. One-time setup

Open a PowerShell terminal at the repo root.

```powershell
# Create and activate a venv
python -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1

# Install Python dependencies (~2 min, ~500 MB)
pip install -r backend\requirements-dev.txt

# Train the four models on notebooks\dataset1.csv (~2 min)
# Writes joblib files to backend\models\ — these are gitignored, so each
# teammate needs to run this once after cloning.
python -m backend.train_models
```

**If PowerShell blocks `Activate.ps1`** with an execution-policy error, run
this once in that terminal and try activation again:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

**macOS / Linux equivalents:**
```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements-dev.txt
python -m backend.train_models
```

### 2. (Optional) Get a Google Safe Browsing API key

Without a key the demo still works — but the Safe Browsing panel will show
`verdict: "unknown"`. To get a free key:

1. <https://console.cloud.google.com/> → create a project
2. Enable the Safe Browsing API:
   <https://console.cloud.google.com/apis/library/safebrowsing.googleapis.com>
3. Create an API key at <https://console.cloud.google.com/apis/credentials>
4. Restrict it to the Safe Browsing API

Then copy the template and paste your key in:
```powershell
copy .env.example .env
# Open .env in your editor, set:
#   SAFE_BROWSING_API_KEY=AIzaSy...
```

`.env` is gitignored — your key won't be committed.

### 3. Run the demo (two terminals)

**Terminal 1 — backend** (from the repo root, with the venv activated):
```powershell
python -m uvicorn backend.main:app --port 8000
```
Wait until you see `Uvicorn running on http://127.0.0.1:8000`. Leave it running.

**Terminal 2 — frontend** (new PowerShell window, repo root):
```powershell
cd frontend
python -m http.server 5000
```
Wait until you see `Serving HTTP on :: port 5000`. Leave it running.

**Open <http://localhost:5000> in your browser.**

Paste any URL, click **Scan**. Results appear in 5-10 seconds.

### 4. Sample URLs to try

| URL | What it demonstrates |
|---|---|
| `https://www.wikipedia.org` | Clean baseline — all models say legitimate |
| `https://testsafebrowsing.appspot.com/s/phishing.html` | Google's own test URL — Safe Browsing flags it |
| `http://192.168.1.1@bit.ly/freegift?login=admin` | Multiple URL-level phishing signals at once |
| `https://paypal-secure-login.com` | Hyphen in domain + suspicious naming |

### 5. Stopping

Ctrl-C in each terminal.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'uvicorn'` | You're running system Python instead of the venv. Activate the venv, or use `.\backend\.venv\Scripts\python.exe -m uvicorn ...` |
| Docker build fails in `backend/Dockerfile` with "Missing models" | Run `python -m backend.train_models` first — the build expects the joblib files to exist. |
| `safe_browsing.verdict` is always `"unknown"` | Either no API key in `.env`, or backend started before `.env` was saved. Restart uvicorn. |
| Scan times out at ~30 seconds | Target site is blocking scrapers or very slow. Pick a different URL. |
| First scan after backend start is slow | Models load on the first request. Subsequent scans are faster. |

## Project structure

```
backend/                     FastAPI service
├── main.py                  /api/health, /api/predict
├── feature_extractor.py     URL → 30-feature dict
├── safe_browsing.py         Google Safe Browsing v4 client
├── train_models.py          One-time training script
├── extractors/              Per-feature extractors (URL-only, HTML, SSL, WHOIS, DNS)
├── models/                  joblib pickles (gitignored, regenerate locally)
└── tests/                   76 pytest cases
├── api.py                   Framework-agnostic prediction core (shared)
frontend/                    Static HTML/JS/CSS served by Firebase Hosting
functions/                   Firebase Cloud Function wrapper (deploy bundle)
prepare_functions.py         Bundles backend into functions/ before deploy
notebooks/                   Original Jupyter notebooks + dataset CSVs
docs/superpowers/            Design doc + implementation plan
DEPLOY.md                    How to deploy to Firebase (Hosting + Functions)
```

## Running the tests

From the `backend/` directory with the venv activated:
```powershell
cd backend
pytest
```

Expected: **76 passed**.

## Deployment

See [DEPLOY.md](DEPLOY.md) for the full Firebase walkthrough (Hosting + a
Python Cloud Function — no Docker or gcloud needed).
