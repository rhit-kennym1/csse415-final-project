# Phishing Detection Demo Web App — Design

**Status:** Draft for review
**Date:** 2026-05-18
**Authors:** Ervin Perkowski, Mark J. Kenny, Sanil Maheshwari

## 1. Goal

Build an interactive web app for the CSSE415 final-project demo that accepts a
URL, extracts the dataset-1 features live from that URL, runs predictions
through the project's four best-performing models, and shows those predictions
side-by-side with Google Safe Browsing's verdict.

The app is intended for a live in-class demo where the audience suggests URLs
and sees results in real time.

## 2. Non-Goals

- Not a production phishing-detection service. Live feature extraction is
  best-effort; some features cannot be perfectly reconstructed (see §6).
- No user accounts, no scan history persistence, no Firestore.
- No retraining or model selection from the UI — models are pre-trained and
  loaded as static artifacts.
- Not building dashboards beyond the single scan-and-compare page.

## 3. Success Criteria

- Pasting any reasonable URL returns a verdict from all four models plus a
  Google Safe Browsing result within ~10 seconds on a normal network.
- The four models match the architecture/hyperparameters documented in the
  Week 9 progress report (Gradient Boosting, Random Forest, XGBoost, SVM).
- The app is reachable at a Firebase Hosting URL and can be demoed without
  the presenter touching a terminal.
- One broken feature lookup (e.g., WHOIS rate-limited) does not crash the
  prediction; the affected feature falls back to a neutral value.

## 4. Architecture

```
Browser
  │
  ▼
Firebase Hosting  (frontend/index.html, app.js, style.css)
  │  fetch('/api/predict', {url})
  ▼
Cloud Run — FastAPI service  (backend/main.py)
  ├─► feature_extractor.py     (requests + BeautifulSoup + whois + dnspython + ssl)
  ├─► 4 joblib pipelines       (models/{gb,rf,xgb,svm}.joblib)
  └─► Google Safe Browsing v4 API
  │
  ▼
JSON response → rendered as two side-by-side panels
```

**Why Cloud Run, not Cloud Functions:**
sklearn pipelines with PolynomialFeatures can be tens of MB; cold starts on
Cloud Functions for that workload are sluggish. Cloud Run holds an instance
warm during the demo session and has no awkward memory cap for this size.

**Why static frontend, not React:**
The UI is one page with one form and one results view. A build pipeline adds
friction without payoff for a demo. Vanilla `fetch()` + DOM updates are
sufficient and easier for the group to explain to the class.

## 5. Repository Layout

```
csse415-final-project/
├── backend/
│   ├── main.py                 # FastAPI app, routes, model loading
│   ├── feature_extractor.py    # URL → 31-feature dict
│   ├── safe_browsing.py        # Google Safe Browsing v4 client
│   ├── train_models.py         # one-time training script (run locally)
│   ├── models/                 # joblib artifacts (gitignored if large)
│   │   ├── gb.joblib
│   │   ├── rf.joblib
│   │   ├── xgb.joblib
│   │   └── svm.joblib
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── firebase.json               # Hosting config + /api/** rewrite to Cloud Run
├── .firebaserc
└── DEPLOY.md                   # step-by-step deploy commands
```

## 6. Feature Extraction

Each of the 31 features falls into one of four buckets based on how it can be
recovered from a live URL:

**URL-only (no network call, all from `urllib.parse`):**
`having_IPhaving_IP_Address`, `URLURL_Length`, `Shortining_Service`,
`having_At_Symbol`, `double_slash_redirecting`, `Prefix_Suffix`,
`having_Sub_Domain`, `HTTPS_token`, `port`.

**Single HTTP fetch + HTML parse (BeautifulSoup):**
`Favicon`, `Request_URL`, `URL_of_Anchor`, `Links_in_tags`, `SFH`,
`Submitting_to_email`, `on_mouseover`, `RightClick`, `popUpWidnow`, `Iframe`,
`Redirect`.

**Network-protocol probes:**
`SSLfinal_State` (TLS handshake + cert validity),
`Domain_registeration_length`, `age_of_domain`, `Abnormal_URL` (all via
`python-whois`), `DNSRecord` (via `dnspython`).

**Approximated (set to neutral 0):**
`web_traffic` (Alexa deprecated May 2022), `Page_Rank` (Google API deprecated
2016), `Google_Index` (Google blocks scraping; Custom Search API is paid),
`Links_pointing_to_page` (requires paid backlink data), `Statistical_report`
(blocklist data — overlaps with the separate Safe Browsing call).

Each extractor:
- Catches its own exceptions and returns `0` on any failure
- Uses a 5-second hard timeout on every network call
- Returns only `-1`, `0`, or `1` to match the dataset's encoding

The list of approximated feature names is included in the API response so
the UI can show "5 features approximated" as a footnote.

## 7. Models

Trained by `train_models.py`, run locally once before deploy. The script
mirrors the feature engineering and hyperparameters reported in Week 9:

| Model | Hyperparameters | Reported accuracy |
|---|---|---|
| Gradient Boosting | `PolynomialFeatures(degree=2, interaction_only=True)` + tuned GB | 0.9774 |
| Random Forest | `max_depth=30, max_features=10, n_estimators=125`, same poly FE | 0.9747 |
| XGBoost | Same poly FE, GridSearchCV-tuned | 0.9738 |
| SVM | `C=10, gamma=0.1` (RBF), same poly FE | 0.9738 |

Each model is wrapped in a single `sklearn.pipeline.Pipeline` that mirrors
the preprocessing used in the corresponding notebook
(`GradientBoosting.ipynb`, `RandForest.ipynb`, `415XGBoost.ipynb`,
`SVM.ipynb`) — typically `PolynomialFeatures(degree=2, interaction_only=True)`
followed by the estimator, with `StandardScaler` added for SVM. The full
pipeline is saved with `joblib.dump`, so at inference time the backend
calls `pipeline.predict_proba([feature_vector])[0]` with no manual
preprocessing in `main.py`.

Models that don't expose `predict_proba` natively (e.g., default SVC) will
be trained with `probability=True`.

## 8. Backend API

**`POST /api/predict`**

Request body:
```json
{"url": "https://example.com"}
```

Response body:
```json
{
  "url": "https://example.com",
  "predictions": [
    {"model": "Gradient Boosting", "verdict": "phishing", "probability": 0.94},
    {"model": "Random Forest",     "verdict": "phishing", "probability": 0.91},
    {"model": "XGBoost",           "verdict": "legitimate", "probability": 0.18},
    {"model": "SVM",               "verdict": "phishing", "probability": 0.88}
  ],
  "safe_browsing": {
    "verdict": "safe",
    "threat_types": []
  },
  "features_meta": {
    "approximated": ["web_traffic", "Page_Rank", "Google_Index",
                     "Links_pointing_to_page", "Statistical_report"],
    "extraction_seconds": 4.2
  }
}
```

- `verdict` per model: `"phishing"` if `probability >= 0.5`, else `"legitimate"`
- `probability` is always the probability of the phishing class (mapped from
  the dataset's `1` label, regardless of the model's internal class ordering)
- Feature extraction and the Safe Browsing call run concurrently via
  `asyncio.gather` so latency is `max(extract_time, safe_browsing_time)`,
  not the sum

**`GET /api/health`** — returns `{"status": "ok"}`. Used by Cloud Run health
checks and as a smoke test after deploy.

**Errors:**
- Malformed URL → `400 {"error": "invalid_url", "message": "..."}`
- Target host unreachable / DNS fails entirely → `200` with all extractable
  features filled in best-effort, network features set to `0`, and
  `features_meta.approximated` extended to include them. We don't return
  `5xx` for "couldn't reach the site" — that should still produce a verdict.
- Safe Browsing API key missing / quota exceeded → `predictions` still
  returned; `safe_browsing` becomes `{"verdict": "unknown", "error": "..."}`.

**CORS:** restricted to the Firebase Hosting domain plus `localhost` for
development.

## 9. Frontend

Single `index.html` with three states swapped in/out via JS:

**Idle:**
- Title + one-sentence project blurb
- URL input + "Scan" button
- A small disclaimer that this is a class demo, not a security product

**Loading:**
- Spinner
- Status text: "Extracting features and querying models..."

**Results:** two columns side-by-side
- **Left column — "Our Models":** four rows, each with model name, a
  red/green verdict badge, and a horizontal probability bar (0-100%)
- **Right column — "Google Safe Browsing":** one verdict badge; if flagged,
  the matching threat type(s) listed below
- Below both columns: agreement indicator
  ("3 of 4 models agree with Google Safe Browsing")
- "Scan another URL" button to reset to idle

All requests go to `/api/predict` (same origin thanks to Firebase Hosting
rewrites — no separate API origin to configure on the client).

## 10. Deployment

**Backend (Cloud Run):**
1. `gcloud builds submit --tag gcr.io/<project>/phishing-demo backend/`
2. `gcloud run deploy phishing-demo --image gcr.io/<project>/phishing-demo --region <region> --allow-unauthenticated --set-env-vars SAFE_BROWSING_API_KEY=<key>`
3. Verify with `curl <cloud-run-url>/api/health`

**Frontend (Firebase Hosting):**
1. `firebase init hosting` (one-time)
2. Edit `firebase.json` to add a rewrite mapping `/api/**` to the Cloud Run
   service (using the `run` rewrite type, which proxies to Cloud Run by
   service name and region)
3. `firebase deploy --only hosting`

The Safe Browsing API key is stored as a Cloud Run environment variable, not
checked into the repo. `DEPLOY.md` documents the full sequence.

## 11. Testing Strategy

- **Unit tests** for `feature_extractor.py`: each extractor function tested
  with a small set of representative inputs (known IP-based URL, known
  shortened URL, etc.). Network-dependent extractors are tested against
  mocked `requests.get` / `whois.whois` responses.
- **Integration smoke test:** `POST /api/predict` with a few well-known URLs
  (the project's GitHub repo URL, a known phishing example from PhishTank
  archives) returns a sensible response shape. Asserts shape, not specific
  verdicts (those depend on live data).
- **Manual demo dry-run** before class: run through ~10 URLs the team
  expects the audience to suggest, confirm response times and verdicts.

## 12. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| WHOIS rate-limiting during a live demo | Per-extractor `0`-on-failure fallback; the prediction still completes |
| Cloud Run cold start adds 5-10s on the first scan of the demo | "Warm up" by hitting `/api/health` on page load |
| Safe Browsing API quota exhausted mid-demo | Free tier is 10k req/day — fine for a demo, but the `safe_browsing` field degrades gracefully if it fails |
| A target site blocks our `requests` user-agent | Set a realistic browser UA; if still blocked, network-fetched features fall back to `0` |
| Models on disk are too large for the container image | If joblib files exceed ~100 MB total we move them to a Cloud Storage bucket and download at container startup |

## 13. Open Questions

None at this time — all major decisions resolved during brainstorming on
2026-05-18.
